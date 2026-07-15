# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""注塑特征/输入构造：无信号，质量输入 = [4 旋钮 + 壁厚 + 材料编码]。"""
import numpy as np
from proctune.core.interfaces import FeatureExtractor


class MoldingFeatureExtractor(FeatureExtractor):
    def extract_features(self, signal):
        return {}

    def build_quality_input(self, setting, context, feats, env=None) -> np.ndarray:
        return np.array([
            setting.get("barrel_temp", 220.0),
            setting.get("inject_pressure", 90.0),
            setting.get("holding_time", 15.0),
            setting.get("mold_temp", 50.0),
            context.get("wall_thickness", 2.0),
            context.get("material_code", 1.0),
        ], dtype=float)
