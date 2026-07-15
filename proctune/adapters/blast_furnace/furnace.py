# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""高炉特征/输入构造：无信号，质量输入 = [进料含量, 炉编码]。"""
import numpy as np
from proctune.core.interfaces import FeatureExtractor
from .profile import FURNACE_MAP


class FurnaceFeatureExtractor(FeatureExtractor):
    def extract_features(self, signal):
        return {}   # 无过程信号

    def build_quality_input(self, setting, context, feats, env=None) -> np.ndarray:
        furnace = setting.get("furnace", "new")
        code = FURNACE_MAP.get(furnace, 0.0)
        return np.array([context.get("content", 50.0), code], dtype=float)
