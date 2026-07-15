# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""注塑业务画像。

特点：
  - 无过程信号；质量直接由「工艺旋钮 + 制件上下文」决定（旋钮→质量 直连）；
  - 旋钮全连续（料筒温度/注射压力/保压时间/模具温度）→ 贝叶斯优化反求；
  - 质量用多标签缺陷分类（短射/飞边/缩水/翘曲）。
与回流焊的差别在于「没有中间信号、质量直连旋钮」，证明框架对两类都能复用。
"""
from proctune.core.abstractions import (BusinessProfile, KnobSpace, KnobParam,
                               SignalSpec, FeatureSpec, ContextSpec, ContextField,
                               QualitySpec, ConstraintSpec)

MATERIAL_MAP = {"PP": 1.0, "ABS": 2.0, "PC": 3.0}

INJECTION_PROFILE = BusinessProfile(
    name="injection_molding",
    description="注塑：料筒温/注射压/保压时/模温 → 短射/飞边/缩水/翘曲；连续 BO 反求",
    knob_space=KnobSpace(params=[
        KnobParam("barrel_temp", 180.0, 260.0, "℃"),
        KnobParam("inject_pressure", 40.0, 140.0, "MPa"),
        KnobParam("holding_time", 5.0, 30.0, "s"),
        KnobParam("mold_temp", 20.0, 80.0, "℃"),
    ]),
    signal=SignalSpec(n_channels=0, n_points=0, duration=0.0, unit=""),
    features=FeatureSpec(names=[]),
    context=ContextSpec(fields=[
        ContextField("material_code", "categorical", 1.0, MATERIAL_MAP),
        ContextField("wall_thickness", "numeric", 2.0),
    ]),
    quality=QualitySpec(kind="defect",
                        defect_labels=["短射", "飞边", "缩水", "翘曲"]),
    constraints=ConstraintSpec(window={}),
)
