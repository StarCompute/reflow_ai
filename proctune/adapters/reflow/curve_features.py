# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""回流焊特征工程：6 路炉温曲线 → 8 维特征；拼接 16 维质量模型输入。

实现 core.interfaces.FeatureExtractor。
"""
import numpy as np
from proctune.core.interfaces import FeatureExtractor

FEATURE_NAMES = ["peak_temp", "tal", "ramp_up", "ramp_down",
                 "delta_t", "soak_temp", "time_above_183", "curve_duration"]


class CurveFeatureExtractor(FeatureExtractor):
    def extract_features(self, signal: np.ndarray) -> dict:
        avg = signal.mean(axis=0)
        times = np.linspace(0, 240, signal.shape[1])
        dt = np.gradient(times)
        dT = np.gradient(avg, times)
        step = float(dt.mean()) if len(dt) else 0.0
        return {
            "peak_temp": float(avg.max()),
            "tal": float(np.sum(avg >= 217) * step),
            "ramp_up": float(np.percentile(dT[:len(dT) // 3], 90)),
            "ramp_down": float(np.percentile(dT[len(dT) * 2 // 3:], 10)),
            "delta_t": float(np.ptp(signal.max(axis=1))),
            "soak_temp": float(avg[len(avg) // 4:len(avg) // 2].mean()),
            "time_above_183": float(np.sum(avg >= 183) * step),
            "curve_duration": float(times[-1] - times[0]),
        }

    def build_quality_input(self, setting, context, feats, env=None) -> np.ndarray:
        env = env or {}
        vec = [
            feats["peak_temp"], feats["tal"], feats["ramp_up"], feats["ramp_down"],
            feats["delta_t"], feats["soak_temp"], feats["time_above_183"], feats["curve_duration"],
            context.get("thickness_mm", 1.6),
            context.get("copper_area_pct", 30.0),
            context.get("bga_count", 0),
            context.get("max_bga_size_mm", 0.0),
            context.get("solder_paste", 0),
            context.get("component_density", 10.0),
            env.get("env_temp", 25.0),
            env.get("env_humidity", 50.0),
        ]
        return np.array(vec, dtype=float)
