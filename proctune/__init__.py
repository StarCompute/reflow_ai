# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""proctune —— 通用「受控工艺参数推荐」框架。

把"回流焊工艺优化"这类问题抽象为统一模式：

    可调工艺参数(旋钮) ─► 物理过程 ─► 可测信号 ─► 提取特征 ─► 质量结果(缺陷/良率)
       Knobs              Process      Signal        Features        Quality
            ▲                                                          │
            └──────── 推荐引擎：给定「对象上下文 + 安全约束」反求最优 Knobs ──┘

本包与具体业务无关；新业务只需提供一份 BusinessProfile + 几个适配器即可复用整套引擎。
详见 proctune/docs/USAGE.md。
"""

__version__ = "0.1.0"
