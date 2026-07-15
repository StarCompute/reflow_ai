"""特征工程：曲线 → 8 维曲线特征；拼接 16 维模型输入。

对应 V2 §5 与清单 1.2 / 2.3。
输入：6 通道热电偶原始采样（演示用物理基线生成；真实环境从 InfluxDB profile_signal 读 tc1~6）。
"""
import numpy as np
from config import CONFIG

# 16 维输入的特征名（顺序与质量模型/推荐引擎一致）
FEATURE_NAMES = [
    "peak_temp", "tal", "ramp_up", "ramp_down", "delta_t", "soak_temp",
    "time_above_183", "curve_duration",                       # 1-8 曲线特征
    "thickness_mm", "copper_area_pct", "bga_count", "max_bga_size_mm",
    "solder_code", "component_density", "env_temp", "env_humidity",  # 9-16 元信息
]


def extract_curve_features(times, tc_signals):
    """times: (N,) 秒；tc_signals: (6, N) 温度℃ → 8 维曲线特征 dict。

    梯度按时间（°C/s）计算，与 V2 §7 工艺窗口单位一致。
    """
    avg = tc_signals.mean(axis=0)
    dt = np.gradient(times)                                  # 每采样点秒数
    dT = np.gradient(avg, times)                             # °C/s
    step = float(dt.mean()) if len(dt) else 0.0
    feats = {
        "peak_temp": float(avg.max()),
        "tal": float(np.sum(avg >= 217) * step),                 # ≥217℃ 液相线累计(秒)
        "ramp_up": float(np.percentile(dT[:len(dT) // 3], 90)),
        "ramp_down": float(np.percentile(dT[len(dT) * 2 // 3:], 10)),
        "delta_t": float(np.ptp(tc_signals.max(axis=1))),       # 各通道峰值最大差
        "soak_temp": float(avg[len(avg) // 4:len(avg) // 2].mean()),
        "time_above_183": float(np.sum(avg >= 183) * step),
        "curve_duration": float(times[-1] - times[0]),
    }
    return feats


def build_model_input(feats, bom, solder_paste, env=None):
    """拼接 16 维模型输入向量。

    feats: 8 维曲线特征 dict
    bom:   dict（thickness_mm / copper_area_pct / bga_count / max_bga_size_mm / component_density）
    solder_paste: 锡膏型号字符串（SAC305 等）
    env:   dict（env_temp / env_humidity），缺省给车间典型值
    """
    env = env or {}
    vec = [
        feats["peak_temp"], feats["tal"], feats["ramp_up"], feats["ramp_down"],
        feats["delta_t"], feats["soak_temp"], feats["time_above_183"], feats["curve_duration"],
        bom.get("thickness_mm", 1.6),
        bom.get("copper_area_pct", 30.0),
        bom.get("bga_count", 0),
        bom.get("max_bga_size_mm", 0.0),
        CONFIG.solder_map.get(solder_paste, 0),                # 锡膏编码
        bom.get("component_density", 10.0),
        env.get("env_temp", 25.0),
        env.get("env_humidity", 50.0),
    ]
    return np.array(vec, dtype=float)
