# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""回流焊 BO 搜索策略：提供边界与「先验候选」加速收敛。

实现 core.interfaces.SearchStrategy。先验来自 BOM 热质量（老师傅看板设温思路）。
"""
import numpy as np
from proctune.core.interfaces import SearchStrategy
from .profile import SIM_PARAMS


class ReflowSearchStrategy(SearchStrategy):
    def bounds(self):
        return {f"zone{i}_temp": (150.0, 280.0) for i in range(1, 9)} | {"chain_speed": (40.0, 120.0)}

    def initial_candidates(self, context):
        P = SIM_PARAMS
        t = float(context.get("thickness_mm", 1.6))
        c = float(context.get("copper_area_pct", 30.0))
        b = float(context.get("bga_count", 0))
        prior_peak = P["peak_base"] + P["peak_k_t"] * t + P["peak_k_c"] * c + P["peak_k_b"] * b

        lead = [170.0, 180.0, 190.0, 200.0, 212.0, 225.0]
        cands = []
        cands.append(dict(zip(
            [f"zone{i}_temp" for i in range(1, 9)] + ["chain_speed"],
            lead + [235.0, 243.0, 85.0])))
        for dp in (-12, -8, -4, 0, 4, 8, 12):
            pk = float(np.clip(prior_peak + dp, 150.0, 280.0))
            for spd in (60.0, 75.0, 90.0, 105.0):
                cands.append(dict(zip(
                    [f"zone{i}_temp" for i in range(1, 9)] + ["chain_speed"],
                    lead + [pk - 10.0, pk, spd])))
        return cands
