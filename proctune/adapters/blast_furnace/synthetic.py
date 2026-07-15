# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""高炉演示造数器：两座炉各自有随机多项式关系（无预设优劣），加噪声。

实现 core.interfaces.SyntheticGenerator。质量 = 多项式(进料含量) + 噪声，裁剪 [0,100]。
"""
import random
import numpy as np
from proctune.core.abstractions import Record
from proctune.core.interfaces import SyntheticGenerator
from .profile import FURNACE_MAP


class BlastFurnaceSynthetic(SyntheticGenerator):
    def __init__(self, seed: int = 42):
        # 固定系数，制造「随进料含量交叉优劣」：低含量新炉优、高含量旧炉优。
        # 这样推荐结果依赖 content，正好演示「离散枚举 + 内容相关择优」。
        self.coef = {
            "new": [70.0, 0.30, -0.0040],
            "old": [30.0, 1.20, -0.0060],
        }

    @staticmethod
    def _poly(x, coef):
        return coef[0] + coef[1] * x + coef[2] * (x ** 2)

    def generate(self, n: int, seed: int = None) -> list:
        rng = random.Random(seed)
        np.random.seed(seed if seed is not None else 0)
        records = []
        for _ in range(n):
            content = round(rng.uniform(20, 80), 1)
            furnace = rng.choice(["new", "old"])
            y = self._poly(content, self.coef[furnace]) + np.random.normal(0, 3)
            y = float(np.clip(y, 0, 100))
            records.append(Record(
                setting={"furnace": furnace},
                context={"content": content},
                signal=None,
                quality_score=y))
        return records
