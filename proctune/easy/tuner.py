# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""EasyTuner：表格进、表格出的极简工艺参数推荐器（面向不懂 AI 的现场人员）。

两种用法：
    from proctune.easy import EasyTuner

    # 1) 极简·自动模式：什么列都不用填，自动识别可调参数/属性/质量列
    tuner = EasyTuner("历史样本.csv")

    # 2) 专业·显式模式：正式场景 / 含文字型参数或数值型属性时更可靠
    tuner = EasyTuner("历史样本.csv", knob_cols=[...], context_cols=[...], quality_col="质量")

    tuner.train()
    tuner.recommend_to_csv("新任务.csv", "推荐结果.csv")   # 拿到系统推荐的参数

底层把表自动翻译成 BusinessProfile + 质量模型 + Recommender，全程对调用者透明。
"""
import os
import csv
import pickle
from typing import List, Dict, Optional, Union

import numpy as np

from proctune.core.abstractions import (
    BusinessProfile, KnobSpace, KnobParam, SignalSpec, FeatureSpec,
    ContextSpec, ContextField, QualitySpec, ConstraintSpec,
)
from proctune.core.interfaces import FeatureExtractor, SearchStrategy
from proctune.core.models.quality import DefectQualityModel, ScoreQualityModel
from proctune.core.models.recommender import Recommender, Constraint

# 质量列里表示「良品 / 没缺陷」的取值（大小写不敏感匹配）
GOOD_LABELS = {"ok", "good", "pass", "合格", "良", "良品", "无缺陷", "无",
               "合格品", "none", "okay", "0", ""}

# ---- 自动推断用的列名关键词（命中即优先判定，大小写不敏感、子串匹配） ----
# 质量结果列：像「质量/评分/结果/缺陷」这类
QUALITY_NAME_HINTS = ["质量", "评分", "得分", "结果", "缺陷", "良率", "良品", "合格",
                      "quality", "score", "defect", "label", "result", "yield", "pass"]
# 产品属性列（不可调、已知）：即使是数字也应归为上下文
CONTEXT_NAME_HINTS = ["材料", "牌号", "型号", "产品", "规格", "壁厚", "厚度", "直径",
                      "镀种", "克重", "尺寸", "批次", "颜色",
                      "material", "grade", "type", "product", "spec", "size",
                      "thickness", "diameter", "batch", "color"]


def _name_hit(col: str, hints) -> bool:
    lc = str(col).lower()
    return any(h.lower() in lc for h in hints)


def _auto_quality_col(rows, cols) -> str:
    """自动挑出最像「质量结果」的那一列。
    优先按列名关键词命中；否则退化为「最后一列」（常见约定质量放最后）。
    """
    for h in QUALITY_NAME_HINTS:
        for c in cols:
            if h.lower() in str(c).lower():
                return c
    return cols[-1]


def _auto_split_knob_context(rows, cols):
    """把剩余列自动分成「可调参数(knob)」和「产品属性(context)」。
    规则：文本列 → 属性；数值列默认 → 可调参数，但列名像属性(壁厚/尺寸…)的数值列仍归属性。
    """
    knobs, ctx = [], []
    for c in cols:
        if _col_is_numeric(rows, c):
            if _name_hit(c, CONTEXT_NAME_HINTS):
                ctx.append(c)      # 数字但明显是属性（如壁厚/克重）
            else:
                knobs.append(c)    # 数字且可调（温度/压力/时间…）
        else:
            ctx.append(c)          # 文本一律当属性（材料/镀种…）
    return knobs, ctx


# --------------------------- 表格读写与类型推断（不依赖 pandas） ---------------------------

def _read_table(path_or_rows) -> List[Dict]:
    """接受 CSV 路径或 list[dict]，统一返回 list[dict]。"""
    if isinstance(path_or_rows, str):
        with open(path_or_rows, newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    if isinstance(path_or_rows, dict):
        return [path_or_rows]
    return list(path_or_rows)


def _to_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _col_is_numeric(rows, col) -> bool:
    for r in rows:
        if r.get(col) in (None, ""):
            continue
        if _to_float(r[col]) is None:
            return False
    return True


def _build_encoder(rows, col) -> Dict[str, float]:
    vals = sorted({r[col] for r in rows if r.get(col) not in (None, "")})
    return {v: float(i + 1) for i, v in enumerate(vals)}


def _infer_quality_kind(rows, col) -> str:
    for r in rows:
        if r.get(col) in (None, ""):
            continue
        if _to_float(r[col]) is None:
            return "defect"
    return "score"


def merge_csv(new_path: str, history_path: str) -> str:
    """把新一批数据（new_path）追加进历史样本表（history_path）。

    两表列结构应一致；新表里多出的列会被忽略、缺少的列补空。
    追加后写回 history_path，方便下次一次性全量重训。
    """
    with open(history_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        existing = list(reader)
    new_rows = _read_table(new_path)
    with open(history_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in existing:
            w.writerow(r)
        for r in new_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    return history_path


# --------------------------- 内建适配器：把「表的一行」翻译成质量模型输入 ---------------------------

class _TableExtractor(FeatureExtractor):
    """无信号业务的内建特征提取器：旋钮 + 上下文 直接拼成向量。"""

    def __init__(self, knob_cols, context_cols, knob_enc, ctx_enc):
        self.knob_cols = knob_cols
        self.context_cols = context_cols
        self.knob_enc = knob_enc
        self.ctx_enc = ctx_enc

    def extract_features(self, signal):
        return {}

    def build_quality_input(self, setting, context, feats, env=None):
        vec = []
        for k in self.knob_cols:
            v = setting[k]
            if k in self.knob_enc:           # 离散旋钮：类别 → 数值
                v = self.knob_enc[k].get(v, 0.0)
            vec.append(float(v))
        for c in self.context_cols:
            v = context[c]
            if c in self.ctx_enc:            # 离散上下文：类别 → 数值
                v = self.ctx_enc[c].get(v, 0.0)
            vec.append(float(v))
        return np.array(vec, dtype=float)


class _TableSearch(SearchStrategy):
    """连续旋钮的搜索边界（来自历史样本推断的范围）。"""

    def __init__(self, bounds):
        self._bounds = bounds

    def bounds(self):
        return self._bounds

    def initial_candidates(self, context):
        return [{k: (lo + hi) / 2.0 for k, (lo, hi) in self._bounds.items()}]


# --------------------------- 极简接口 ---------------------------

class EasyTuner:
    """给不懂 AI 的人用的工艺参数推荐器。

    两种用法
    --------
    1) 极简（自动推断，什么列都不用填）：
           tuner = EasyTuner("历史样本.csv")
       系统自动识别：像「质量/评分/缺陷」的列当质量结果；数值列当可调参数；
       文本列（及壁厚/尺寸等明显属性的数值列）当产品属性。

    2) 专业（显式指定列，推荐用于正式场景）：
           tuner = EasyTuner("历史样本.csv",
                             knob_cols=[...], context_cols=[...], quality_col="质量")
       当数值列里既有「可调参数」又有「不可调属性」、无法靠列名区分时，
       请显式指定，结果最可靠。

    参数
    ----
    data         : 历史样本 CSV 路径，或 list[dict]。每行 = 一次生产记录。
    knob_cols    : 可调工艺参数列名（系统要帮你「反推」出来的东西，如温度/压力）。缺省则自动推断。
    context_cols : 产品属性列名（已知、不可调，如材料/厚度）。缺省则自动推断。
    quality_col  : 质量结果列名。可以是缺陷名（"OK"/"短射"...）或连续评分（0~100）。缺省则自动推断。
    quality_kind : 'defect' / 'score'，默认自动推断（文本→defect，数字→score）。
    score_scale  : 评分模式下的满分值，默认自动（按数据最大值）。
    name         : 业务名称（仅用于标识）。
    """

    def __init__(self, data, knob_cols=None, context_cols=None, quality_col=None,
                 quality_kind: Optional[str] = None, score_scale: Optional[float] = None,
                 name: str = "my_business", n_trees: int = 200):
        self._rows = _read_table(data)
        if not self._rows:
            raise ValueError("历史样本为空，请检查 data 路径或内容。")

        # 列定义：缺省的部分自动推断（未填的才推断，已填的完全尊重用户）
        all_cols = list(self._rows[0].keys())
        self.auto_inferred = (knob_cols is None or context_cols is None or quality_col is None)
        if quality_col is None:
            quality_col = _auto_quality_col(self._rows, all_cols)
        remaining = [c for c in all_cols if c != quality_col]
        if knob_cols is None and context_cols is None:
            knob_cols, context_cols = _auto_split_knob_context(self._rows, remaining)
        elif knob_cols is None:
            context_cols = list(context_cols)
            knob_cols = [c for c in remaining if c not in context_cols]
        elif context_cols is None:
            knob_cols = list(knob_cols)
            context_cols = [c for c in remaining if c not in knob_cols]

        self.knob_cols = list(knob_cols)
        self.context_cols = list(context_cols)
        self.quality_col = quality_col
        self.quality_kind = quality_kind
        self.score_scale = score_scale
        self.name = name
        self.n_trees = n_trees

        # 训练产物（train 后填充）
        self._ctx_enc: Dict[str, Dict] = {}
        self._knob_enc: Dict[str, Dict] = {}
        self._knob_defs: List[tuple] = []      # (列名, 'continuous'/'categorical', (lo,hi) 或 [类别])
        self._quality_kind: Optional[str] = None
        self._defect_labels: List[str] = []
        self._score_scale_used: float = 100.0
        self.quality = None
        self._profile = None
        self._recommender = None
        self._trained = False

    # ---------------- 从 MySQL 构造（数据源切换） ----------------
    @classmethod
    def from_mysql(cls, conn: Dict, table: str, knob_cols, context_cols,
                   quality_col, quality_kind: Optional[str] = None,
                   score_scale: Optional[float] = None, name: str = "my_business",
                   n_trees: int = 200):
        """从一张 MySQL 表直接构造（表里的列名需与 knob/context/quality 对应）。

        conn 示例：{"host": "127.0.0.1", "user": "root", "password": "xxx",
                   "database": "factory", "port": 3306}
        依赖 pymysql 或 mysql-connector-python（任选其一）。
        """
        from proctune.easy.db import read_mysql_table
        rows = read_mysql_table(table=table, **conn)
        if not rows:
            raise ValueError("MySQL 表为空，请检查连接信息与表名。")
        return cls(rows, knob_cols, context_cols, quality_col,
                   quality_kind=quality_kind, score_scale=score_scale,
                   name=name, n_trees=n_trees)

    # ---------------- 训练 ----------------
    def train(self):
        rows = self._rows

        # 1) 上下文编码（文本列自动编号）
        for c in self.context_cols:
            if not _col_is_numeric(rows, c):
                self._ctx_enc[c] = _build_encoder(rows, c)

        # 2) 旋钮类型与范围
        self._knob_defs = []
        self._knob_enc = {}
        for k in self.knob_cols:
            if _col_is_numeric(rows, k):
                vals = [_to_float(r[k]) for r in rows if _to_float(r[k]) is not None]
                lo, hi = min(vals), max(vals)
                pad = (hi - lo) * 0.02 + 1e-6
                self._knob_defs.append((k, "continuous", (lo - pad, hi + pad)))
            else:
                cats = sorted({r[k] for r in rows if r.get(k) not in (None, "")})
                self._knob_enc[k] = {v: float(i + 1) for i, v in enumerate(cats)}
                self._knob_defs.append((k, "categorical", cats))

        # 3) 构造训练矩阵 X = [旋钮..., 上下文...]
        X = np.array([self._row_vector(r) for r in rows], dtype=float)

        # 4) 质量列
        self._quality_kind = self.quality_kind or _infer_quality_kind(rows, self.quality_col)
        if self._quality_kind == "defect":
            labels = sorted({str(r[self.quality_col]).strip()
                             for r in rows
                             if str(r.get(self.quality_col, "")).strip().lower() not in GOOD_LABELS})
            self._defect_labels = labels
            Y = np.zeros((len(rows), len(labels)), dtype=int)
            for i, r in enumerate(rows):
                lab = str(r.get(self.quality_col, "")).strip()
                if lab.lower() in GOOD_LABELS:
                    continue
                if lab in labels:
                    Y[i, labels.index(lab)] = 1
            self.quality = DefectQualityModel(len(labels), n_estimators=self.n_trees)
            self.quality.fit(X, Y)
        else:
            y = np.array([float(r[self.quality_col]) for r in rows], dtype=float)
            mmax = float(np.max(y)) if len(y) else 100.0
            scale = self.score_scale or (1.0 if mmax <= 1 else (10.0 if mmax <= 10 else 100.0))
            self._score_scale_used = scale
            self.quality = ScoreQualityModel(scale, n_estimators=self.n_trees)
            self.quality.fit(X, y)

        self._build_engine()
        self._trained = True
        return self

    def _row_vector(self, r) -> List[float]:
        vec = []
        for (k, kind, _) in self._knob_defs:
            v = r.get(k)
            if kind == "categorical":
                v = self._knob_enc[k].get(v, 0.0)
            vec.append(float(v))
        for c in self.context_cols:
            v = r.get(c)
            if c in self._ctx_enc:
                v = self._ctx_enc[c].get(v, 0.0)
            vec.append(float(v))
        return vec

    def _build_engine(self):
        knob_params = []
        for (k, kind, info) in self._knob_defs:
            if kind == "continuous":
                knob_params.append(KnobParam(k, info[0], info[1], kind="continuous"))
            else:
                knob_params.append(KnobParam(k, 0, len(info) - 1, kind="categorical",
                                             categories=list(info)))
        ctx_fields = []
        for c in self.context_cols:
            if c in self._ctx_enc:
                ctx_fields.append(ContextField(c, "categorical", 0.0, self._ctx_enc[c]))
            else:
                ctx_fields.append(ContextField(c, "numeric", 0.0))

        self._profile = BusinessProfile(
            name=self.name,
            description="自动从历史样本构建",
            knob_space=KnobSpace(params=knob_params),
            signal=SignalSpec(n_channels=0),
            features=FeatureSpec(names=[]),
            context=ContextSpec(fields=ctx_fields),
            quality=QualitySpec(kind=self._quality_kind,
                                defect_labels=self._defect_labels,
                                scale=self._score_scale_used),
            constraints=ConstraintSpec(window={}),
        )
        self._extractor = _TableExtractor(self.knob_cols, self.context_cols,
                                          self._knob_enc, self._ctx_enc)
        cont_bounds = {k: info for (k, kind, info) in self._knob_defs if kind == "continuous"}
        self._search = _TableSearch(cont_bounds)
        self._recommender = Recommender(self._profile, None, self._extractor,
                                        self.quality, Constraint({}), self._search)

    # ---------------- 推荐 ----------------
    def _encode_context(self, row) -> Dict[str, float]:
        out = {}
        for c in self.context_cols:
            v = row.get(c)
            if c in self._ctx_enc:
                v = self._ctx_enc[c].get(v, 0.0)
            out[c] = float(v)
        return out

    def recommend_one(self, context: Dict, top_k: int = 1) -> List[Dict]:
        """对单个新任务（属性 dict）返回 top_k 个推荐参数。"""
        if not self._trained:
            raise RuntimeError("请先调用 train()。")
        ctx_num = self._encode_context(context)
        recs = self._recommender.recommend(ctx_num, top_k=top_k)
        out = []
        for rank, rec in enumerate(recs, 1):
            item = dict(context)
            for k, v in rec["setting"].items():
                item[f"推荐_{k}"] = v
            item["预测良率"] = rec["score"]
            item["排名"] = rank
            out.append(item)
        return out

    def recommend(self, new_data, top_k: int = 1) -> List[Dict]:
        """对新任务表（CSV 路径 / list[dict]）逐行推荐，返回结果行列表。"""
        rows = _read_table(new_data)
        results = []
        for r in rows:
            results.extend(self.recommend_one(r, top_k=top_k))
        return results

    def recommend_to_csv(self, new_data, out_path: str, top_k: int = 1):
        """把推荐结果写成 CSV（新任务属性 + 推荐参数 + 预测良率）。"""
        rows = self.recommend(new_data, top_k=top_k)
        if not rows:
            raise RuntimeError("没有可输出的推荐结果。")
        cols = list(rows[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        return out_path

    # ---------------- 追加新数据 / 重训 ----------------
    def add_data(self, path_or_rows):
        """把新样本（CSV 路径或 list[dict]）追加进内存，暂不训练。

        建议攒够一批再 retrain() 一次，比每条都重训更高效。
        """
        self._rows.extend(_read_table(path_or_rows))
        self._trained = False
        return self

    def retrain(self):
        """用当前全部样本（含 add_data 追加的）重新训练。"""
        return self.train()

    @property
    def n_samples(self) -> int:
        """当前已纳入的样本总数。"""
        return len(self._rows)

    # ---------------- 模型保存 / 加载（训练一次，下次直接复用） ----------------
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: str) -> "EasyTuner":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError("文件不是 EasyTuner 模型。")
        return obj

    # ---------------- 给普通人看的信息 ----------------
    def summary(self) -> str:
        if not self._trained:
            return "尚未训练。"
        mode = "离散枚举择优" if any(k[1] == "categorical" for k in self._knob_defs) else "参数寻优"
        q = "缺陷分类" if self._quality_kind == "defect" else f"评分回归(满分{self._score_scale_used})"
        src = "列自动推断" if getattr(self, "auto_inferred", False) else "列手动指定"
        return (f"业务[{self.name}] 已就绪（{src}）：可调参数={self.knob_cols}；"
                f"产品属性={self.context_cols}；质量列={self.quality_col}({q})；"
                f"推荐模式={mode}；样本数={len(self._rows)}")
