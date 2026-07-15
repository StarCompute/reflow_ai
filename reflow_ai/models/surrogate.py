"""模型1 · 工艺代理模型（Surrogate）。

设定值（温区×10 + 链速）→ 预测 6 通道炉温曲线。
实现：物理热模型基线 + 神经网络残差校正（PINN 思路）。
对应 V2 §6 模型1 / 清单 2.2。
生产可将 MLPRegressor 替换为 PyTorch PINN（物理约束网络）。
"""
import numpy as np
from sklearn.neural_network import MLPRegressor


class SimpleThermalSolver:
    """物理热模型基线：一阶滞后近似炉温沿炉长的爬升。

    时间轴以秒计（默认 240s 约 4 分钟过炉），梯度按 °C/s 计算，
    与 V2 §7 工艺窗口（ramp_up 1~3°C/s 等）单位一致。
    末段加入出板冷却尾，使 ramp_down 落在合理区间。
    """

    def __init__(self, n_points=180, n_channels=6, duration_s=240.0):
        self.n_points = n_points
        self.n_channels = n_channels
        self.duration_s = duration_s

    def predict(self, setting):
        zones = np.array(setting["zones"], dtype=float)        # (10,)
        speed = float(setting.get("chain_speed", 85.0))
        t = np.linspace(0, 1, self.n_points)
        # 温区之间线性插值 → 平滑目标剖面（避免阶梯跳变造成虚假陡坡）
        target = np.interp(t * (len(zones) - 1), np.arange(len(zones)), zones)
        # 出板冷却段：最后 15% 目标线性降至 130℃（放缓，贴近真实冷却斜率）
        cool = t > 0.85
        if cool.any():
            target[cool] = np.interp(
                np.linspace(0, 1, cool.sum()), [0, 1], [zones[-1], 130.0])

        # 各通道独立热滞后：不同热电偶位置热质量不同 → 峰值有差（delta_t）。
        # 链速越快，在炉内停留越短，各通道差异被放大 → delta_t 增大（对应立碑风险）。
        speed_factor = 0.7 + (np.clip(speed, 40, 120) - 40) / 80.0 * 0.6   # 0.7~1.3
        base_alpha = 0.18
        out = np.zeros((self.n_channels, self.n_points))
        for c in range(self.n_channels):
            ac = base_alpha * (0.5 + 1.0 * c / max(1, self.n_channels - 1)) * speed_factor
            ch = np.full(self.n_points, target[0])           # 初值=起始目标，避免从0骤升的假陡坡
            for i in range(1, self.n_points):
                ch[i] = ch[i - 1] + ac * (target[i] - ch[i - 1])
            out[c] = ch + np.linspace(-1.5, 1.5, self.n_channels)[c]
        return out


class SurrogateModel:
    """物理基线 + NN 残差。"""

    def __init__(self, n_points=180):
        self.physics = SimpleThermalSolver(n_points=n_points)
        self.residual = MLPRegressor(hidden_layer_sizes=(64, 128, 64),
                                     max_iter=300, random_state=42)
        self.n_points = n_points

    @staticmethod
    def _setting_to_x(setting):
        return np.array(list(setting["zones"]) + [setting["chain_speed"]], dtype=float)

    def fit(self, X_settings, Y_curves):
        """X_settings: list[dict]; Y_curves: (M, 6, 180) 实测曲线。"""
        Xb = np.array([self._setting_to_x(s) for s in X_settings])     # (M, 11)
        base = np.array([self.physics.predict(s) for s in X_settings])
        resid = Y_curves - base                                      # (M, 6, 180)
        R_flat = resid.reshape(resid.shape[0], -1)                  # (M, 1080)
        self.residual.fit(Xb, R_flat)

    def predict(self, setting):
        base = self.physics.predict(setting)                         # (6, 180)
        x = self._setting_to_x(setting).reshape(1, -1)             # (1, 11)
        resid = self.residual.predict(x).reshape(base.shape[0], base.shape[1])
        return base + resid
