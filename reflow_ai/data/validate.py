# Copyright (c) 2026 蒲俊杰（Pu Junjie）. All rights reserved.
# 许可见 https://github.com/StarCompute/reflow_ai/blob/main/proctune/LICENSE.md
# 个人使用（含个人商业）免费；企业/组织商业使用需获得授权。

"""数据质量红线校验（对应清单 1.3 / V2 §4 质量约束）。

红线定义与《数据采集清单 & AI 算法处理逻辑清单》1.3 完全一致。
返回结构化报告，供采集 / 算法团队现场核对；不通过则阻断训练（见 demo / CI）。

真实环境：
- R2（曲线点数）应查 InfluxDB `profile_signal` 的 tc1~6 采样计数；
- R3（实测-设定偏差）用 `profile_feature.peak_temp` 与设定峰值比对。
演示环境无时序库，由调用方传入 curve_point_provider 近似。
"""
from data.db import SessionLocal, ReflowRun, QualityResult, ProfileFeature
from config import CONFIG
from utils.logger import get_logger

logger = get_logger("validate")

# 质量红线（id 对应清单 1.3）
RED_LINES = {
    "R1": "run_id 唯一且非空",
    "R2": "每通道曲线采样点 >= 60",
    "R3": "实测峰值 - 设定峰值偏差 <= 30℃（否则报警）",
    "R4": "追溯率 100%：每个 run 均有 quality_result",
    "R5": "时间合法：end_time > start_time",
    "R6": "lot_id 非空（板级追溯）",
    "R7": "defect_type 在枚举内",
    "R8": "必填非空：oven_id / product_id / chain_speed",
}

_VALID_DEFECTS = {"无", "虚焊", "桥连", "立碑", "空洞", "锡珠", "其他"}


def _set_peak(zones):
    return float(max(zones)) if zones else None


def validate_dataset(session=None, curve_point_provider=None):
    """校验全库数据，返回报告 dict。

    curve_point_provider(run) -> int|None：返回该 run 曲线采样点数。
        传 None 则跳过 R2（演示环境用 ProfileFeature 是否存在近似）。
    """
    own = session is None
    session = session or SessionLocal()
    report = {
        "total_runs": 0,
        "issues": [],
        "red_line_hits": {k: 0 for k in RED_LINES},
        "pass": True,
    }
    try:
        runs = session.query(ReflowRun).all()
        report["total_runs"] = len(runs)
        qr_lots = set(q.lot_id for q in session.query(QualityResult.lot_id).all())

        for r in runs:
            rid = r.run_id or "<empty>"
            if not r.run_id:
                report["issues"].append(("R1", f"run_id 为空"))
                report["red_line_hits"]["R1"] += 1
            if not r.lot_id:
                report["issues"].append(("R6", f"run={rid} lot_id 空"))
                report["red_line_hits"]["R6"] += 1
            if not r.oven_id or not r.product_id or r.chain_speed is None:
                report["issues"].append(("R8", f"run={rid} 必填缺失"))
                report["red_line_hits"]["R8"] += 1
            if r.end_time and r.start_time and r.end_time <= r.start_time:
                report["issues"].append(("R5", f"run={rid} 时间非法"))
                report["red_line_hits"]["R5"] += 1
            if r.lot_id and r.lot_id not in qr_lots:
                report["issues"].append(("R4", f"run={rid} lot={r.lot_id} 无质量结果"))
                report["red_line_hits"]["R4"] += 1

            # R2 曲线采样点（真实查 InfluxDB；演示可传 None 跳过）
            if curve_point_provider is not None:
                n = curve_point_provider(r)
                if n is not None and n < 60:
                    report["issues"].append(("R2", f"run={rid} 曲线点={n}<60"))
                    report["red_line_hits"]["R2"] += 1

            # R3 实测-设定偏差
            pf = session.query(ProfileFeature).filter_by(run_id=r.run_id).first()
            if pf is not None and pf.peak_temp is not None:
                sp = _set_peak([getattr(r, f"zone{i}_temp")
                                 for i in range(1, CONFIG.zone_num + 1)])
                if sp is not None and abs(pf.peak_temp - sp) > 30:
                    dev = pf.peak_temp - sp
                    report["issues"].append(
                        ("R3", f"run={rid} 实测-设定偏差={dev:.1f}℃>30"))
                    report["red_line_hits"]["R3"] += 1

        # R7 缺陷枚举
        for q in session.query(QualityResult).all():
            if q.defect_type not in _VALID_DEFECTS:
                report["issues"].append(
                    ("R7", f"lot={q.lot_id} defect_type={q.defect_type} 非法"))
                report["red_line_hits"]["R7"] += 1

        report["pass"] = len(report["issues"]) == 0
        return report
    finally:
        if own:
            session.close()


def print_report(report):
    logger.info(f"质量红线校验：共 {report['total_runs']} 炉次，"
                f"问题 {len(report['issues'])} 项，"
                f"结论={'通过' if report['pass'] else '不通过'}")
    for rid, desc in RED_LINES.items():
        hits = report["red_line_hits"][rid]
        status = "OK" if hits == 0 else f"FAIL x{hits}"
        logger.info(f"  {rid} {desc} -> {status}")
    for code, msg in report["issues"][:20]:
        logger.warning(f"  [{code}] {msg}")
