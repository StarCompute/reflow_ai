# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""模型3 · 通用推荐引擎。

两种推荐模式，由旋钮空间自动选择：
  - 含离散旋钮（如「选哪座炉」）→ 离散枚举择优（黄铜高炉）。
  - 全连续旋钮（如 8 温区 + 链速）→ 贝叶斯优化（GP+UCB）在代理+质量上搜（回流焊/注塑）。

目标 = 最大化 goodness(质量) − K·约束罚分；硬约束越界 100% 拦截。
连续模式复用 SearchStrategy 提供边界与初始候选（业务可注入先验加速收敛）。
"""
import warnings
import itertools
from typing import Dict, List, Optional

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from ..abstractions import BusinessProfile
from ..interfaces import SignalSimulator, FeatureExtractor, SearchStrategy


class Constraint:
    """工艺窗口：越界返回罚分（>0 表示有越界）。"""

    def __init__(self, window: Dict[str, tuple]):
        self.w = window

    def penalty(self, feats: Dict[str, float]) -> float:
        p = 0.0
        for k, (lo, hi) in self.w.items():
            v = feats.get(k)
            if v is None:
                continue
            if v < lo:
                p += (lo - v)
            elif v > hi:
                p += (v - hi)
        return p


class Recommender:
    def __init__(self, profile: BusinessProfile,
                 simulator: Optional[SignalSimulator],
                 extractor: FeatureExtractor,
                 quality,
                 constraint: Constraint,
                 search: Optional[SearchStrategy] = None,
                 n_points: int = 180):
        self.profile = profile
        self.sim = simulator
        self.ext = extractor
        self.quality = quality
        self.constraint = constraint
        self.search = search
        self.n_points = n_points

    # ---- 批量评估：把「代理预测 + 特征 + 质量」合并成矩阵运算，单次推理 ----
    def _evaluate(self, settings: List[Dict[str, float]],
                  context: Dict[str, float], env: Dict[str, float] = None):
        m = len(settings)
        if m == 0:
            return np.array([]), []
        feats_list, Xmat = [], []
        for s in settings:
            if self.sim is not None:
                signal = self.sim.predict(s)
                feats = self.ext.extract_features(signal)
            else:
                feats = {}
            X = self.ext.build_quality_input(s, context, feats, env)
            feats_list.append(feats)
            Xmat.append(X)
        good = self.quality.goodness(np.array(Xmat))
        scores, pens = [], []
        for g, f in zip(good, feats_list):
            pen = self.constraint.penalty(f)
            pens.append(pen)
            scores.append(float(g) - 1000.0 * pen)
        return np.array(scores), feats_list

    def _pack(self, setting, score, feats, context, env=None):
        X = self.ext.build_quality_input(setting, context, feats, env)
        qual_report, goodness = self.quality.report(X)
        return {
            "rank": 0,
            "score": round(float(goodness), 4),
            "raw_score": round(float(score), 4),
            "setting": {k: (round(float(v), 2) if isinstance(v, (int, float)) else v)
                        for k, v in setting.items()},
            "features": {k: round(float(v), 2) for k, v in feats.items()},
            "quality": qual_report,
        }

    # ---- 模式A：离散枚举（含 categorical 旋钮）----
    def _recommend_discrete(self, context, top_k, env):
        cat = self.profile.knob_space.categorical()
        cont = self.profile.knob_space.continuous()
        combos = list(itertools.product(*[c.categories for c in cat]))
        candidates = []
        for combo in combos:
            setting = {p.name: c for p, c in zip(cat, combo)}
            for p in cont:  # 连续旋钮取中点（混合场景的简化；纯离散业务无连续旋钮）
                setting[p.name] = (p.low + p.high) / 2.0
            candidates.append(setting)
        scores, feats = self._evaluate(candidates, context, env)
        order = np.argsort(scores)[::-1][:top_k]
        return [self._pack(candidates[i], scores[i], feats[i], context, env)
                for i in order]

    # ---- 模式B：贝叶斯优化（全连续旋钮）----
    def _recommend_bo(self, context, top_k, env,
                      n_init=12, n_iter=24, n_candidates=80):
        keys = list(self.search.bounds().keys())
        lo = np.array([self.search.bounds()[k][0] for k in keys])
        hi = np.array([self.search.bounds()[k][1] for k in keys])
        rng = np.random.default_rng(42)

        # 初始候选：随机 + 业务提供的「全部先验候选」（加速收敛到合法区）
        init_list = [rng.uniform(lo, hi, size=(n_init, len(keys)))]
        for c in self.search.initial_candidates(context):
            init_list.append(np.array([c[k] for k in keys]).reshape(1, -1))
        Xs = np.vstack(init_list)
        ys = self._eval_keys(Xs, keys, context, env)

        kernel = 1.0 * Matern(length_scale=0.4, length_scale_bounds="fixed", nu=2.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(n_iter):
                gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                              alpha=1e-6, n_restarts_optimizer=0)
                gp.fit(Xs, ys)
                Xcand = rng.uniform(lo, hi, size=(n_candidates, len(keys)))
                mu, std = gp.predict(Xcand, return_std=True)
                x_next = Xcand[np.argmax(mu + 1.5 * std)]
                Xs = np.vstack([Xs, x_next.reshape(1, -1)])
                ys = np.append(ys, self._eval_keys(x_next.reshape(1, -1), keys, context, env)[0])

        best = Xs[int(np.argmax(ys))].copy()
        refine = self._refine(best, keys, lo, hi)
        if refine:
            Xs = np.vstack([Xs, np.vstack(refine)])
            ys = np.append(ys, self._eval_keys(np.vstack(refine), keys, context, env))

        order = np.argsort(ys)[::-1][:top_k]
        results = []
        for idx in order:
            setting = dict(zip(keys, Xs[idx]))
            score = ys[idx]
            if self.sim is not None:
                feats = self.ext.extract_features(self.sim.predict(setting))
            else:
                feats = {}
            results.append(self._pack(setting, score, feats, context, env))
        return results

    def _eval_keys(self, X, keys, context, env):
        settings = [dict(zip(keys, row)) for row in X]
        scores, _ = self._evaluate(settings, context, env)
        return scores

    def _refine(self, best, keys, lo, hi, span=8.0, step=2.0):
        """对最后两个连续旋钮（通常是峰值/链速等关键量）做细网格微调，
        消除高维 BO 抖动；其余维保持最优解不变。"""
        if len(keys) < 2:
            return []
        i1, i2 = -2, -1
        grid = np.arange(-span, span + 1e-9, step)
        out = []
        for d1 in grid:
            for d2 in grid:
                cand = best.copy()
                cand[i1] = float(np.clip(best[i1] + d1, lo[i1], hi[i1]))
                cand[i2] = float(np.clip(best[i2] + d2, lo[i2], hi[i2]))
                out.append(cand.reshape(1, -1))
        return out

    # ---- 统一入口 ----
    def recommend(self, context: Dict[str, float], top_k: int = 3,
                  env: Dict[str, float] = None) -> List[dict]:
        if self.profile.knob_space.categorical():
            return self._recommend_discrete(context, top_k, env)
        return self._recommend_bo(context, top_k, env)
