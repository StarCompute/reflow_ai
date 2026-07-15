# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""注塑 BO 搜索策略：边界 + 中点附近初始候选。"""
import numpy as np
from proctune.core.interfaces import SearchStrategy


class InjectionSearchStrategy(SearchStrategy):
    def bounds(self):
        return {"barrel_temp": (180.0, 260.0), "inject_pressure": (40.0, 140.0),
                "holding_time": (5.0, 30.0), "mold_temp": (20.0, 80.0)}

    def initial_candidates(self, context):
        mid = {"barrel_temp": 220.0, "inject_pressure": 90.0,
               "holding_time": 15.0, "mold_temp": 50.0}
        cands = [dict(mid)]
        for k in mid:
            for d in (-15, -8, 8, 15):
                c = dict(mid)
                lo, hi = self.bounds()[k]
                c[k] = float(np.clip(mid[k] + d, lo, hi))
                cands.append(c)
        return cands
