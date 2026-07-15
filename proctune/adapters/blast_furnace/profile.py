# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""黄铜高炉业务画像。

特点（与回流焊「类似但不同」）：
  - 无过程信号（没有炉温曲线），质量直接由「进料含量 + 选哪座炉」决定；
  - 旋钮是离散的（new / old 两座炉），推荐=枚举两座炉、择优；
  - 质量是连续评分（出料品质 %），用回归而非分类。
这正好覆盖通用框架的「离散枚举 + 评分回归」分支。
"""
from proctune.core.abstractions import (BusinessProfile, KnobSpace, KnobParam,
                               SignalSpec, FeatureSpec, ContextSpec, ContextField,
                               QualitySpec, ConstraintSpec)

FURNACE_MAP = {"new": 0.0, "old": 1.0}

BLAST_FURNACE_PROFILE = BusinessProfile(
    name="blast_furnace",
    description="黄铜选矿高炉：进料含量 + 选哪座炉 → 出料品质%；离散枚举择优",
    knob_space=KnobSpace(params=[
        KnobParam("furnace", 0.0, 1.0, "", kind="categorical",
                  categories=["new", "old"]),
    ]),
    signal=SignalSpec(n_channels=0, n_points=0, duration=0.0, unit="%"),
    features=FeatureSpec(names=[]),
    context=ContextSpec(fields=[
        ContextField("content", "numeric", 50.0),   # 进料含量 %
    ]),
    quality=QualitySpec(kind="score", scale=100.0),
    constraints=ConstraintSpec(window={}),           # 无信号特征约束
)
