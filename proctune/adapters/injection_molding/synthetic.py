# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""注塑演示造数器：缺陷由「旋钮偏离该制件最优带」触发（老师傅经验规律）。

实现 core.interfaces.SyntheticGenerator。每条样本：随机制件 + 旋钮，按隐藏规律判缺陷。
最优旋钮均落在旋钮范围内，便于演示优化器能找到高良率设定。
"""
import random
import numpy as np
from proctune.core.abstractions import Record
from proctune.core.interfaces import SyntheticGenerator
from .profile import MATERIAL_MAP


class InjectionSynthetic(SyntheticGenerator):
    # 各材料理想料筒温度基准（随壁厚升高）；最优均落在 [180,260] 内
    MAT_BASE = {"PP": 195.0, "ABS": 215.0, "PC": 212.0}
    MAT_IDX = {"PP": 0, "ABS": 1, "PC": 2}

    def generate(self, n: int, seed: int = None) -> list:
        rng = random.Random(seed)
        np.random.seed(seed if seed is not None else 123)
        materials = list(MATERIAL_MAP.keys())
        records = []
        for _ in range(n):
            material = rng.choice(materials)
            wall = round(rng.uniform(0.8, 4.0), 1)
            barrel = round(rng.uniform(180, 260), 1)
            pressure = round(rng.uniform(40, 140), 1)
            hold = round(rng.uniform(5, 30), 1)
            mold = round(rng.uniform(20, 80), 1)

            ideal_barrel = self.MAT_BASE[material] + 8.0 * wall
            # 各缺陷的「偏离量」（>0 表示往缺陷方向偏）
            d_short = (ideal_barrel - barrel) + np.random.normal(0, 4)            # 料温偏低→短射
            d_flash = (pressure - (60 + 15 * wall)) + np.random.normal(0, 6)     # 压力偏高→飞边
            d_sink = ((12 - hold) + 2.0 * wall) + np.random.normal(0, 4)         # 保压短+厚壁→缩水
            d_warp = (mold - (52 + 5 * self.MAT_IDX[material])) + np.random.normal(0, 5)  # 模温偏高→翘曲

            devs = {"短射": d_short, "飞边": d_flash, "缩水": d_sink, "翘曲": d_warp}
            label = "无"
            best = max(devs.values())
            if best > 6:  # 偏离足够大才出缺陷
                label = max(devs, key=devs.get)

            setting = {"barrel_temp": barrel, "inject_pressure": pressure,
                       "holding_time": hold, "mold_temp": mold}
            context = {"material_code": float(MATERIAL_MAP[material]),
                       "wall_thickness": wall}
            records.append(Record(setting=setting, context=context,
                                  signal=None, quality_label=label))
        return records
