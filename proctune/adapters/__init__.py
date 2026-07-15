# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""adapters：各业务实现（平移领域逻辑，实现 core 接口）。

现有业务：
  - reflow           回流焊（信号介导 + 连续BO + 缺陷分类）
  - blast_furnace    黄铜选矿高炉（无信号 + 离散枚举择优 + 评分回归）
  - injection_molding 注塑（无信号 + 连续BO + 缺陷分类）
"""
