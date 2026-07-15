# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""模型1 · 通用工艺代理模型（物理基线 + 神经网络残差）。

- 物理基线由 SignalSimulator 注入（回流焊=热滞后模型，其他业务可换成自己的物理/经验基线）。
- 残差用 MLPRegressor 学习；生产可替换为 PyTorch PINN。
- 无信号业务（如黄铜高炉）不创建本模型，直接跳过。
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from ..interfaces import SignalSimulator


class SurrogateModel:
    def __init__(self, simulator: SignalSimulator, n_points: int = 180):
        self.sim = simulator
        self.n_points = n_points
        self.residual = MLPRegressor(hidden_layer_sizes=(64, 128, 64),
                                     max_iter=300, random_state=42)

    @staticmethod
    def _knobs_to_x(setting: dict) -> np.ndarray:
        keys = sorted(setting.keys())
        return np.array([setting[k] for k in keys], dtype=float)

    def fit(self, settings, signals):
        """settings: list[dict]; signals: (M, C, N) 过程信号。"""
        Xb = np.array([self._knobs_to_x(s) for s in settings])
        base = np.array([self.sim.predict(s) for s in settings])
        resid = np.asarray(signals, dtype=float) - base
        self.residual.fit(Xb, resid.reshape(resid.shape[0], -1))
        self._keys = sorted(settings[0].keys())

    def predict(self, setting: dict) -> np.ndarray:
        base = self.sim.predict(setting)
        x = self._knobs_to_x(setting).reshape(1, -1)
        resid = self.residual.predict(x).reshape(base.shape)
        return base + resid
