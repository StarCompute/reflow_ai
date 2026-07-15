# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""回流焊物理基线（一阶热滞后近似），作为代理模型的物理先验。

实现了 core.interfaces.SignalSimulator：predict(setting) -> (6,180) 炉温曲线。
生产可替换为真实物理 / InfluxDB 实测曲线读取。
"""
import numpy as np
from proctune.core.interfaces import SignalSimulator


class ThermalSolver(SignalSimulator):
    def __init__(self, n_points: int = 180, n_channels: int = 6, duration_s: float = 240.0):
        self.n_points = n_points
        self.n_channels = n_channels
        self.duration_s = duration_s

    def predict(self, setting: dict) -> np.ndarray:
        zones = np.array([setting[f"zone{i}_temp"] for i in range(1, 9)], dtype=float)
        speed = float(setting["chain_speed"])
        t = np.linspace(0, 1, self.n_points)
        target = np.interp(t * (len(zones) - 1), np.arange(len(zones)), zones)
        cool = t > 0.85
        if cool.any():
            target[cool] = np.interp(np.linspace(0, 1, cool.sum()), [0, 1], [zones[-1], 130.0])

        speed_factor = 0.7 + (np.clip(speed, 40, 120) - 40) / 80.0 * 0.6   # 0.7~1.3
        base_alpha = 0.18
        out = np.zeros((self.n_channels, self.n_points))
        for c in range(self.n_channels):
            ac = base_alpha * (0.5 + 1.0 * c / max(1, self.n_channels - 1)) * speed_factor
            ch = np.full(self.n_points, target[0])
            for i in range(1, self.n_points):
                ch[i] = ch[i - 1] + ac * (target[i] - ch[i - 1])
            out[c] = ch + np.linspace(-1.5, 1.5, self.n_channels)[c]
        return out
