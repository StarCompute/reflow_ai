# Copyright (c) 2026 蒲俊杰（Pu Junjie）. All rights reserved.
# 许可见 https://github.com/StarCompute/reflow_ai/blob/main/proctune/LICENSE.md
# 个人使用（含个人商业）免费；企业/组织商业使用需获得授权。

"""全局配置：工艺窗口硬约束、缺陷标签、路径。

对应 V2 §7 安全约束与清单 1.3 质量红线。
生产环境建议从 YAML / 环境变量读取，这里用 dataclass 便于演示。
"""
from dataclasses import dataclass, field
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    # 关系库连接串；演示用相对 SQLite（在 reflow_ai 目录下运行）
    # 生产改 MySQL: "mysql+pymysql://user:pass@host:3306/reflow_ai"
    db_url: str = "sqlite:///reflow_ai.db"

    zone_num: int = 8

    # 工艺窗口：安全硬约束（越界 100% 拦截，见 V2 §7 / 清单 1.3）
    # 这是生产环境的真实规范（来自 V2 文档），API 安全网关 /dispatch 用它校验。
    process_window: dict = field(default_factory=lambda: {
        "peak_temp": (235.0, 250.0),
        "tal": (45.0, 90.0),
        "ramp_up": (1.0, 3.0),
        "ramp_down": (-4.0, -1.0),
        "delta_t": (0.0, 15.0),
        "soak_temp": (150.0, 200.0),
        "time_above_183": (60.0, 120.0),
        "chain_speed": (40.0, 120.0),
    })

    # 演示窗口：贴合本仓库『玩具物理代理模型』（8 温区）实际产出的特征分布。
    # 已按 8 段温区 + 实测曲线峰值偏移(~8.3℃) 重新标定，使各板『经验最优解』
    # 都能落在窗口内（推荐引擎约束满足率=100%），同时仍能拦截 ±16℃ 的偏离板。
    # 与 process_window 刻意分离——生产接入真实数据/真实物理后，
    # 训练与推荐统一改用 process_window（真实规范）。
    demo_window: dict = field(default_factory=lambda: {
        "peak_temp": (218.0, 245.0),
        "tal": (35.0, 75.0),
        "ramp_up": (0.12, 0.70),
        "ramp_down": (-3.4, -2.4),
        "delta_t": (3.0, 12.0),
        "soak_temp": (172.0, 203.0),
        "time_above_183": (110.0, 212.0),
        "chain_speed": (40.0, 120.0),
    })

    # 模型关注的缺陷类型（顺序即标签索引）
    defect_labels: list = field(default_factory=lambda: ["虚焊", "桥连", "立碑", "空洞", "锡珠"])

    # 锡膏型号 → 编码（对应 V2 §5.2 维度13）
    solder_map: dict = field(default_factory=lambda: {
        "SAC305": 1, "SN63PB37": 2, "SN100C": 3,
    })

    # 模型落盘目录
    model_dir: str = os.path.join(PROJECT_ROOT, "models", "artifacts")


CONFIG = Config()


# ---------------------------------------------------------------------------
# 模拟数据生成参数（仅演示用；生产接入真实数据后整段弃用）
# 集中放置，便于按产线实况调参。默认值已尽量贴近无铅回流焊（SAC305）实况：
#   - 峰值 235~250℃、链速 40~120 cm/min、均热 150~200℃
#   - 良品率取较真实的 ~82%（即约 18% 误设/偏离），而非夸张的 75%
# 真实产线请把下方数值替换为你们的实测分布与工艺窗口。
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    # 历史样本量（演示默认 2000 条）
    n_history: int = 2000

    # 训练用 PCB 板型（BOM 特征）+ 验证用新板（刻意不在训练集）
    # 字段: (product_id, thickness_mm, copper_area_pct, bga_count,
    #        max_bga_size_mm, component_density, solder_paste)
    pcb_types: list = field(default_factory=lambda: [
        ("PCB-A12", 1.6, 35.2, 3, 27.0, 12.5, "SAC305"),   # 标准板
        ("PCB-B07", 1.0, 20.0, 1, 15.0,  8.0, "SAC305"),   # 薄板 / 低铜 / 单 BGA
        ("PCB-C21", 2.4, 55.0, 4, 35.0, 16.0, "SAC305"),   # 厚板 / 高铜 / 多 BGA
        ("PCB-D03", 1.2, 45.0, 2, 20.0, 10.0, "SN100C"),   # 中薄板 / 中铜 / 不同锡膏
        ("PCB-E09", 2.0, 50.0, 5, 30.0, 14.0, "SAC305"),   # 中厚板 / 多 BGA
    ])
    new_pcb_types: list = field(default_factory=lambda: [
        ("PCB-F15", 1.4, 40.0, 2, 18.0, 11.0, "SAC305"),   # 中薄板 / 中铜
        ("PCB-G33", 2.2, 60.0, 6, 33.0, 15.0, "SN100C"),   # 厚板 / 极高铜 / 多 BGA
        ("PCB-H01", 0.8, 15.0, 1, 12.0,  6.0, "SAC305"),   # 极薄板 / 低铜
    ])

    # 老师傅经验最优工艺（隐藏真值）：随热质量线性变化
    #   peak = base + k_t·板厚 + k_c·铜% + k_b·BGA数
    #   speed = base + k_t·板厚 + k_c·铜% + k_b·BGA数（链速随热质量变慢）
    #   soak = base + k_t·板厚
    peak_base: float = 222.0
    peak_k_t: float = 4.0
    peak_k_c: float = 0.15
    peak_k_b: float = 1.5
    speed_base: float = 95.0
    speed_k_t: float = -6.0
    speed_k_c: float = -0.2
    speed_k_b: float = -2.0
    soak_base: float = 175.0
    soak_k_t: float = 0.5

    # 造数分布
    # 良率水平对齐行业实况：普遍 95%~98%，较高 99%~99.5%+。
    # 好板比例即良率近似（偏离板几乎必出缺陷），故默认取 0.97（普遍水平，
    # 实测数据集良率约 95.5%）；想演示『较高水平』可上调到 0.99。
    good_ratio: float = 0.97          # 好板比例（其余为误设/偏离，产生缺陷）
    peak_noise_std: float = 2.5       # 好板峰值噪声（℃）
    speed_noise_std: float = 4.0      # 好板链速噪声（cm/min）
    soak_noise_std: float = 4.0       # 均热温度噪声
    zone_noise_std: float = 3.0       # 各温区设定噪声
    dev_peak_offsets: list = field(default_factory=lambda: [-16, -11, 11, 16])
    dev_speed_offsets: list = field(default_factory=lambda: [-22, 22])

    # 曲线物理偏差：8 温区代理模型产出的曲线峰值比设定低约这么多（热损失）。
    # 实测均值 ≈ -8.3℃（std 2.6）；旧版 10 温区为 ~-5℃。
    # 必须与实际物理一致，否则合格带错位、好板被误判虚焊、良率虚低。
    curve_peak_offset: float = 8.3
    # 合格带半宽：围绕"该板 ideal_curve_peak"的 ±（留出干净高良率平台）
    pass_band_half: float = 7.0

    # 次级缺陷阈值（安全网，常规曲线区间之外才触发）
    tombstone_dt: float = 18.0        # delta_t 超此 → 立碑
    void_tal: float = 240.0           # time_above_183 超此 → 空洞
    solderball_ramp: float = 4.0      # ramp_up 超此 → 锡珠


SIM = SimConfig()
