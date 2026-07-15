# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""回流焊业务画像 + 造数/搜索用到的「老师傅经验」系数。

这里集中了所有回流焊专属字面量；core 引擎不感知这些。
演示用 demo_window（与玩具物理一致，保证约束满足率 100%）；
生产接入真实数据后，把 constraints.window 换成 process_window 即可。
"""
from proctune.core.abstractions import (BusinessProfile, KnobSpace, KnobParam,
                               SignalSpec, FeatureSpec, ContextSpec, ContextField,
                               QualitySpec, ConstraintSpec)

# 演示用工艺窗口（贴合玩具物理代理模型实际产出）；生产改 process_window
DEMO_WINDOW = {
    "peak_temp": (218.0, 245.0),
    "tal": (35.0, 75.0),
    "ramp_up": (0.12, 0.70),
    "ramp_down": (-3.4, -2.4),
    "delta_t": (3.0, 12.0),
    "soak_temp": (172.0, 203.0),
    "time_above_183": (110.0, 212.0),
    "chain_speed": (40.0, 120.0),
}

PROCESS_WINDOW = {
    "peak_temp": (235.0, 250.0),
    "tal": (45.0, 90.0),
    "ramp_up": (1.0, 3.0),
    "ramp_down": (-4.0, -1.0),
    "delta_t": (0.0, 15.0),
    "soak_temp": (150.0, 200.0),
    "time_above_183": (60.0, 120.0),
    "chain_speed": (40.0, 120.0),
}

# 「老师傅看板设温」隐藏规律（演示用，生产换成真实经验/数据）
SIM_PARAMS = {
    "peak_base": 222.0, "peak_k_t": 4.0, "peak_k_c": 0.15, "peak_k_b": 1.5,
    "speed_base": 95.0, "speed_k_t": -6.0, "speed_k_c": -0.2, "speed_k_b": -2.0,
    "soak_base": 175.0, "soak_k_t": 0.5,
    "good_ratio": 0.97, "peak_noise_std": 2.5, "speed_noise_std": 4.0,
    "soak_noise_std": 4.0, "zone_noise_std": 3.0,
    "dev_peak_offsets": [-16, -11, 11, 16], "dev_speed_offsets": [-22, 22],
    "curve_peak_offset": 8.3, "pass_band_half": 7.0,
    "tombstone_dt": 18.0, "void_tal": 240.0, "solderball_ramp": 4.0,
    "pcb_types": [
        ("PCB-A12", 1.6, 35.2, 3, 27.0, 12.5, "SAC305"),
        ("PCB-B07", 1.0, 20.0, 1, 15.0, 8.0, "SAC305"),
        ("PCB-C21", 2.4, 55.0, 4, 35.0, 16.0, "SAC305"),
        ("PCB-D03", 1.2, 45.0, 2, 20.0, 10.0, "SN100C"),
        ("PCB-E09", 2.0, 50.0, 5, 30.0, 14.0, "SAC305"),
    ],
    "new_pcb_types": [
        ("PCB-F15", 1.4, 40.0, 2, 18.0, 11.0, "SAC305"),
        ("PCB-G33", 2.2, 60.0, 6, 33.0, 15.0, "SN100C"),
        ("PCB-H01", 0.8, 15.0, 1, 12.0, 6.0, "SAC305"),
    ],
}

SOLDER_MAP = {"SAC305": 1, "SN63PB37": 2, "SN100C": 3}

REFLOW_PROFILE = BusinessProfile(
    name="reflow_soldering",
    description="回流焊：8 温区 + 链速 → 6 路炉温曲线 → 5 类缺陷；连续 BO 反求最优工艺",
    knob_space=KnobSpace(params=[
        KnobParam(f"zone{i}_temp", 150.0, 280.0, "℃") for i in range(1, 9)
    ] + [KnobParam("chain_speed", 40.0, 120.0, "cm/min")]),
    signal=SignalSpec(n_channels=6, n_points=180, duration=240.0, unit="℃"),
    features=FeatureSpec(names=["peak_temp", "tal", "ramp_up", "ramp_down",
                                "delta_t", "soak_temp", "time_above_183", "curve_duration"]),
    context=ContextSpec(fields=[
        ContextField("thickness_mm", "numeric", 1.6),
        ContextField("copper_area_pct", "numeric", 30.0),
        ContextField("bga_count", "numeric", 0),
        ContextField("max_bga_size_mm", "numeric", 0.0),
        ContextField("component_density", "numeric", 10.0),
        ContextField("solder_paste", "categorical", 0, SOLDER_MAP),
        ContextField("env_temp", "numeric", 25.0),
        ContextField("env_humidity", "numeric", 50.0),
    ]),
    quality=QualitySpec(kind="defect",
                        defect_labels=["虚焊", "桥连", "立碑", "空洞", "锡珠"]),
    constraints=ConstraintSpec(window=DEMO_WINDOW),
)
