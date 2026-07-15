# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""通用抽象：用 5 个数据类描述「任意受控工艺业务」。

新业务只需填一份 BusinessProfile，引擎即可在其上跑训练 / 推荐 / 评估 / 安全网关。
所有"回流焊专属"的字面量都不应出现在这里，只出现在 adapters/* 里。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ------------------------- 旋钮（可调工艺参数） -------------------------
@dataclass
class KnobParam:
    name: str                 # 参数名，如 "zone1_temp" / "barrel_temp"
    low: float                # 下限
    high: float               # 上限
    unit: str = ""            # 单位，仅展示用
    kind: str = "continuous"  # 'continuous' 连续 | 'categorical' 离散
    categories: List[str] = field(default_factory=list)  # 离散取值（kind='categorical' 时）


@dataclass
class KnobSpace:
    params: List[KnobParam]

    def continuous(self) -> List[KnobParam]:
        return [p for p in self.params if p.kind == "continuous"]

    def categorical(self) -> List[KnobParam]:
        return [p for p in self.params if p.kind == "categorical"]

    def bounds(self) -> Dict[str, Tuple[float, float]]:
        """仅连续旋钮的搜索边界。"""
        return {p.name: (p.low, p.high) for p in self.continuous()}

    def default_setting(self) -> Dict[str, float]:
        """连续取中点，离散取第一个类别。"""
        s = {}
        for p in self.params:
            if p.kind == "continuous":
                s[p.name] = (p.low + p.high) / 2.0
            else:
                s[p.name] = p.categories[0] if p.categories else 0.0
        return s


# ------------------------- 信号（过程可测输出） -------------------------
@dataclass
class SignalSpec:
    n_channels: int = 0       # 通道数（如 6 路热电偶）；无信号业务填 0
    n_points: int = 0         # 每通道采样点数
    duration: float = 0.0     # 过程时长（秒）
    unit: str = "℃"           # 信号单位，仅展示


# ------------------------- 特征（信号提取出的标量） -------------------------
@dataclass
class FeatureSpec:
    names: List[str] = field(default_factory=list)  # 信号特征名，用于约束校验


# ------------------------- 上下文（被加工对象的属性） -------------------------
@dataclass
class ContextField:
    name: str
    kind: str = "numeric"            # 'numeric' | 'categorical'
    default: float = 0.0
    encode: Dict[str, float] = field(default_factory=dict)  # 离散值→数值编码


@dataclass
class ContextSpec:
    fields: List[ContextField] = field(default_factory=list)

    def encode(self, context: Dict[str, object]) -> Dict[str, float]:
        """把业务上下文（可能含字符串类别）编码成数值 dict。"""
        out = {}
        known = {f.name: f for f in self.fields}
        for name, val in context.items():
            f = known.get(name)
            if f is None:
                out[name] = float(val)
            elif f.kind == "categorical":
                out[name] = float(f.encode.get(val, 0.0))
            else:
                out[name] = float(val)
        return out


# ------------------------- 质量（缺陷标签 / 评分） -------------------------
@dataclass
class QualitySpec:
    kind: str = "defect"          # 'defect' 多标签分类 | 'score' 回归评分
    defect_labels: List[str] = field(default_factory=list)  # kind='defect' 时的标签
    scale: float = 100.0          # kind='score' 时评分归一化分母


# ------------------------- 约束（安全硬窗口） -------------------------
@dataclass
class ConstraintSpec:
    # 键为特征名（信号特征），值为 (lo, hi)；越界即拦截
    window: Dict[str, Tuple[float, float]] = field(default_factory=dict)


# ------------------------- 业务画像（唯一需要新业务提供的东西） -------------------------
@dataclass
class BusinessProfile:
    name: str
    knob_space: KnobSpace
    signal: SignalSpec
    features: FeatureSpec
    context: ContextSpec
    quality: QualitySpec
    constraints: ConstraintSpec
    description: str = ""


# ------------------------- 训练样本记录（内存态，避免强绑 DB） -------------------------
@dataclass
class Record:
    setting: Dict[str, float]               # 旋钮取值
    context: Dict[str, float]               # 已编码的数值上下文
    signal: Optional[object] = None         # 过程信号数组（无信号业务为 None）
    quality_label: Optional[str] = None     # kind='defect' 时的真实标签
    quality_score: Optional[float] = None   # kind='score' 时的真实评分
