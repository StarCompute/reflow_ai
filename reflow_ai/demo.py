# Copyright (c) 2026 蒲俊杰（Pu Junjie）. All rights reserved.
# 许可见 https://github.com/StarCompute/reflow_ai/blob/main/proctune/LICENSE.md
# 个人使用（含个人商业）免费；企业/组织商业使用需获得授权。

"""回流焊 AI 工艺优化 · 演示脚本（纯本地，无 docker）。

两条流程彻底分离，可单独运行：
  python demo.py train      数据 + 训练 + 评估（离线，跑一次即可；默认 2000 条历史）
  python demo.py recommend  加载已训模型，对新 PCB 计算推荐（在线，无需重训）
  python demo.py recommend N  同上，动态生成 N 块新 PCB（默认 3）
  python demo.py samples    展示数据库样本数据（默认 20 条；samples N 可指定条数）
  python demo.py            等价于先 train 再 recommend

说明：演示数据按"老师傅看板设温"的真实工艺规律生成——
      每种 PCB 的热质量不同（板厚/铜面积/BGA 数）→ 经验最优峰值、链速不同，
      缺陷由"该板最优峰值"决定（而非全局窗口）。因此质量模型必须
      从 BOM 特征里学出"板不同、合格线不同"，推荐才能按新板自动设计。
      真实数据接入后指标会更准。
"""
import datetime
import numpy as np
import joblib
import os
import random
import warnings

# 全局忽略无害警告（GP 收敛、数据转换、DB 连接未关闭等），避免刷屏
warnings.simplefilter("ignore")

from config import CONFIG, SIM
from utils.logger import get_logger
from utils.console import console, step, banner, make_table

from data.db import init_db, migrate_db, SessionLocal, ReflowRun, QualityResult, ProfileFeature, Bom
from features.extractor import extract_curve_features
from models.surrogate import SurrogateModel, SimpleThermalSolver
from models.quality import QualityModel
from models.recommender import ReflowRecommender, ProcessWindow
from training.train import train_all, load_training_data
from training.evaluate import cross_validate_quality, evaluate_surrogate, evaluate_recommender
from data.validate import validate_dataset, print_report
from pipeline.feedback import record_feedback
import features.extractor as featurizer

# 板型参数统一从 SimConfig 读取（便于按产线调参）
PCB_TYPES = SIM.pcb_types
NEW_PCB_TYPES = SIM.new_pcb_types
from config import CONFIG
from utils.logger import get_logger

log = get_logger("demo")


def _gen_new_pcb(n=3, seed=None):
    """动态生成『训练集未见过』的新 PCB（验证泛化）。

    在训练板型分布区间内随机采样，但随机 product_id 保证不在训练集，
    验证模型对『未见板』的泛化与『看板设温』能力。
    字段同 PCB_TYPES: (product_id, thickness_mm, copper_area_pct, bga_count,
                       max_bga_size_mm, component_density, solder_paste)
    """
    rng = random.Random(seed)
    solders = list(CONFIG.solder_map.keys())
    out = []
    for _ in range(n):
        pid = f"PCB-NX{rng.randint(100, 999)}"
        thickness = round(rng.uniform(0.8, 2.4), 1)
        copper = round(rng.uniform(15.0, 60.0), 1)
        bga = rng.randint(1, 6)
        max_bga = round(rng.uniform(12.0, 35.0), 1)
        density = round(rng.uniform(6.0, 16.0), 1)
        solder = rng.choice(solders)
        out.append((pid, thickness, copper, bga, max_bga, density, solder))
    return out


# ---------------------------------------------------------------------------
# 1) 造数：基于 PCB 属性驱动（老师傅"看板设温"）
# ---------------------------------------------------------------------------
# 多种 PCB：板厚 / 铜面积 / BGA 数量 / 元件密度 / 锡膏 各不相同。
# 老师傅设定温度的前提正是 PCB——热质量越大（厚板 / 高铜 / 多 BGA），
# 需要更高峰值、更慢链速才能把焊点烤透；反之薄板要低峰值、快链速。
# 板型参数统一在 config.SIM 中定义，便于按产线调参。


def _bom_dict(pcb):
    return {
        "thickness_mm": pcb[1], "copper_area_pct": pcb[2],
        "bga_count": pcb[3], "max_bga_size_mm": pcb[4],
        "component_density": pcb[5],
    }


def _master_optimal(bom):
    """老师傅的『经验最优工艺』：随 PCB 热质量变化（隐藏的真实映射，演示用）。

    热质量 ∝ 板厚 + 铜面积 + BGA 数量 → 峰值需更高、链速需更慢。
    历史缺陷标签由它 + 噪声决定，质量模型必须从数据里把它学出来，
    而不是背一条与 PCB 无关的全局规则。
    系数全部来自 config.SIM，便于按产线实况调参。
    """
    t, c, b = bom["thickness_mm"], bom["copper_area_pct"], bom["bga_count"]
    ideal_peak = SIM.peak_base + SIM.peak_k_t * t + SIM.peak_k_c * c + SIM.peak_k_b * b
    ideal_speed = SIM.speed_base + SIM.speed_k_t * t + SIM.speed_k_c * c + SIM.speed_k_b * b
    ideal_soak = SIM.soak_base + SIM.soak_k_t * t
    return ideal_peak, ideal_speed, ideal_soak


def _true_defect(bom, feat):
    """缺陷标签（模拟老师傅真实经验）：依赖『该 PCB 的最优峰值』。

    与旧版 _label_from_features 的关键区别：
    - 旧版用全局 demo_window 判缺陷 → 规则自证、与 PCB 无关；
    - 新版用『该板的 ideal_peak』判缺陷 → 同一峰值对 A12 合格、
      对 C21 却是虚焊，模型必须从 BOM 特征学出"板不同、合格线不同"。

    次级缺陷（立碑/空洞/锡珠）阈值刻意设到常规曲线区间之外，
    仅作安全网——演示数据主要由"峰值 vs 该板最优"驱动，
    这正是"老师傅看板设温"的核心因果。
    """
    t, c, b = bom["thickness_mm"], bom["copper_area_pct"], bom["bga_count"]
    # 判据以"曲线峰值"为准，而曲线峰值系统性比设定低约 curve_peak_offset℃，
    # 故合格带中心下移该值，对齐"老师傅设 ideal_peak → 曲线呈现 ideal_peak-offset"。
    ideal_curve_peak = (SIM.peak_base + SIM.peak_k_t * t + SIM.peak_k_c * c
                        + SIM.peak_k_b * b - SIM.curve_peak_offset)
    peak = feat["peak_temp"]
    # 合格带对称 ±pass_band_half℃：留出一段"干净的高良率平台"，
    # 避免最优点也紧贴边界、残留缺陷概率把连乘良率压低。
    if peak < ideal_curve_peak - SIM.pass_band_half:
        return "虚焊"
    if peak > ideal_curve_peak + SIM.pass_band_half:
        return "桥连"
    if feat["delta_t"] > SIM.tombstone_dt:
        return "立碑"
    if feat["time_above_183"] > SIM.void_tal:
        return "空洞"
    if feat["ramp_up"] > SIM.solderball_ramp:
        return "锡珠"
    return "无"


def _gen_bom_pool(n_extra=40, seed=20260714):
    """生成覆盖『连续 BOM 空间』的板型池：5 个标准板 + n_extra 个随机板。

    这是解决『新板良率偏低』的关键：系统的目的是给『训练集未见过的新板』
    看板设温，但若训练数据只有 5 个离散板型，质量模型只学到 5 个点，
    新板落在点之间/之外 → 输出残留噪声概率 → 被 ∏(1-p) 连乘压低良率。
    让训练集覆盖与 _gen_new_pcb 相同的参数区间，新板即『分布内』，可正常泛化。
    """
    rng = random.Random(seed)
    solders = list(CONFIG.solder_map.keys())
    pool = list(PCB_TYPES)
    for i in range(n_extra):
        pool.append((
            f"PCB-SYN{i:03d}",
            round(rng.uniform(0.8, 2.4), 1),      # 板厚
            round(rng.uniform(15.0, 60.0), 1),    # 铜面积%
            rng.randint(1, 6),                    # BGA 数
            round(rng.uniform(12.0, 35.0), 1),    # 最大 BGA 尺寸
            round(rng.uniform(6.0, 16.0), 1),     # 元件密度
            rng.choice(solders),                  # 锡膏
        ))
    return pool


def seed_demo_data(n=None):
    n = SIM.n_history if n is None else n
    pool = _gen_bom_pool()          # 覆盖连续 BOM 空间，保证新板可泛化
    sess = SessionLocal()
    try:
        # 是否需要清库重建：
        #   1) 旧版单板数据（缺 solder_paste）；
        #   2) 旧库只有离散标准板、缺『连续 BOM 池』(PCB-SYN*) —— 这会让质量
        #      模型只学到几个离散点，对新板泛化差、良率被 ∏(1-p) 压低。
        #   若已存在连续池则跳过造数（避免每次重训都重造，秒级但无谓）。
        has_data = sess.query(ReflowRun).count() > 0
        has_legacy = sess.query(ReflowRun).filter(ReflowRun.solder_paste.is_(None)).first() is not None
        has_continuum = sess.query(Bom).filter(Bom.product_id.like("PCB-SYN%")).first() is not None
        if has_data and (has_legacy or not has_continuum):
            log.info("检测到旧库（单板/无连续 BOM 池），清空后以『连续 BOM 池』重新造数。")
            sess.query(QualityResult).delete()
            sess.query(ProfileFeature).delete()
            sess.query(ReflowRun).delete()
            sess.query(Bom).delete()
            sess.commit()
        elif has_data:
            log.info("已有『连续 BOM 池』炉次数据，跳过造数。")
            return

        rows = []          # (run_id, lot_id, dt)
        for i in range(n):
            pcb = random.choice(pool)
            pid, solder = pcb[0], pcb[6]
            bom = _bom_dict(pcb)
            ideal_peak, ideal_speed, ideal_soak = _master_optimal(bom)

            # 老师傅设定：多数贴近该板最优（好板），少数偏离（产生缺陷）
            if random.random() < SIM.good_ratio:
                tgt_peak = ideal_peak + np.random.normal(0, SIM.peak_noise_std)
                speed = ideal_speed + np.random.normal(0, SIM.speed_noise_std)
            else:
                # 偏离：模拟经验不足或误设
                tgt_peak = ideal_peak + random.choice(SIM.dev_peak_offsets)
                speed = ideal_speed + random.choice(SIM.dev_speed_offsets)

            # 由目标峰值 + 该板均热构建 8 段温区剖面
            # （前 4 段均热爬升、第 5~6 段过渡、第 7~8 段峰值保持）
            soak = ideal_soak + np.random.normal(0, SIM.soak_noise_std)
            base = [soak - 12, soak, soak + 8, soak + 16, 212.0, 225.0,
                    tgt_peak - 10.0, tgt_peak]
            zones = [round(float(np.clip(v + np.random.normal(0, SIM.zone_noise_std), 150, 280)), 1)
                     for v in base]
            speed = round(float(np.clip(speed, 40, 120)), 1)

            run_id = f"R-260714-{i:04d}"
            lot_id = f"L-{i:04d}"
            r = ReflowRun(
                run_id=run_id, oven_id="OVEN-03", product_id=pid,
                lot_id=lot_id, chain_speed=speed, solder_paste=solder,
                **{f"zone{j}_temp": zones[j - 1] for j in range(1, CONFIG.zone_num + 1)},
                start_time=datetime.datetime.now(),
                end_time=datetime.datetime.now()
                + datetime.timedelta(minutes=random.randint(3, 6)),
            )
            sess.add(r)

            # 曲线 → 8 维特征
            curve = SimpleThermalSolver().predict({"zones": zones, "chain_speed": speed})
            feat = extract_curve_features(np.linspace(0, 240, 180), curve)
            pf = ProfileFeature(run_id=run_id, **{k: feat[k] for k in
                                 ["peak_temp", "tal", "ramp_up", "ramp_down",
                                  "delta_t", "soak_temp", "time_above_183", "curve_duration"]})
            sess.add(pf)

            # 缺陷类型由『该 PCB 经验最优』推导（见 _true_defect）
            dt = _true_defect(bom, feat)
            rows.append((run_id, lot_id, dt))

        for (run_id, lot_id, dt) in rows:
            ok = (dt == "无")
            qr = QualityResult(
                lot_id=lot_id,
                aoi_result="PASS" if ok else "FAIL",
                defect_type=dt,
                repair_flag=not ok,
                inspect_time=datetime.datetime.now(),
            )
            sess.add(qr)

        # 每种 PCB 的 BOM 维表都要落库（质量模型输入依赖它）
        for pcb in pool:
            if sess.query(Bom).filter_by(product_id=pcb[0]).count() == 0:
                sess.add(Bom(product_id=pcb[0], thickness_mm=pcb[1],
                             copper_area_pct=pcb[2], bga_count=pcb[3],
                             max_bga_size_mm=pcb[4], component_density=pcb[5]))
        sess.commit()
        log.info(f"已生成 {n} 条『基于 PCB 驱动』模拟炉次（{len(pool)} 种 PCB，覆盖连续 BOM 空间）。")
    finally:
        sess.close()


# ---------------------------------------------------------------------------
# 2) 流程 A：数据 + 训练 + 评估（离线，跑一次即可）
# ---------------------------------------------------------------------------
def pipeline_train(n_history=None):
    n_history = SIM.n_history if n_history is None else n_history
    banner("回流焊 AI 工艺优化 · 训练流程",
           f"造数 {n_history} 条 → 红线校验 → 训练三模型 → 评估验收")
    step("1. 初始化数据库（建表 + 兼容迁移）")
    init_db()
    migrate_db()

    step(f"2. 准备历史数据（基于 PCB 驱动，{n_history} 条）")
    seed_demo_data(n_history)

    step("3. 质量红线校验（清单 1.3）")
    report = validate_dataset()
    print_report(report)
    if not report["pass"]:
        log.warning("数据未通过质量红线，请先治理数据再训练（演示继续）。")

    step("4. 训练三个模型（代理 / 质量 / 推荐）")
    train_all()

    step("5. 模型评估（对照清单 2.6 验收指标）")
    _evaluate()
    log.info("[green]训练完成，模型已落盘（surrogate.pkl / quality.pkl）。[/]")


# ---------------------------------------------------------------------------
# 3) 流程 B：对新 PCB 计算推荐（在线，加载已训模型，无需重训）
# ---------------------------------------------------------------------------
def _load_models():
    surr_path = os.path.join(CONFIG.model_dir, "surrogate.pkl")
    qual_path = os.path.join(CONFIG.model_dir, "quality.pkl")
    if not (os.path.exists(surr_path) and os.path.exists(qual_path)):
        raise SystemExit("未找到已训模型，请先运行：python demo.py train")
    surr = joblib.load(surr_path)
    qual = joblib.load(qual_path)
    qual.set_inference_threads(1)   # 在线推理单/少样本，关并行最快
    return surr, qual


def pipeline_recommend(n=3):
    log.info("[recommend] 加载已训模型")
    surr, qual = _load_models()
    # 演示用 demo_window（与训练标签一致）；生产接真实数据后改用 process_window
    rec = ReflowRecommender(surr, qual, featurizer, ProcessWindow(CONFIG.demo_window))

    banner("回流焊 AI 工艺优化 · 新板推荐（在线，无需重训）",
           "动态生成『训练集未见过的新 PCB』，自动计算差异化工艺")
    step(f"动态生成 {n} 块新 PCB 并计算推荐（验证泛化）")
    new_pcbs = _gen_new_pcb(n=n)
    table = make_table("新 PCB 推荐结果（动态生成）", [
        ("PCB", "left"), ("板参数", "left"),
        ("推荐 8 段温区(℃)", "left"),
        ("链速", "right"), ("峰值(℃)", "right"), ("预测良率", "right"),
    ])
    for pcb in new_pcbs:
        bom = _bom_dict(pcb)
        results = rec.recommend(bom, pcb[6], top_k=1)
        top = results[0]
        yield_score = top["score"]
        ycol = "[green]" if yield_score >= 0.99 else ("[yellow]" if yield_score >= 0.90 else "[red]")
        # 输入是 8 段温区，输出同样给出完整 8 段温区剖面（而非仅峰值）
        zones_str = " / ".join(f"{z:.0f}" for z in top["setting"]["zones"])
        table.add_row(
            pcb[0],
            f"厚{pcb[1]}mm/铜{pcb[2]}%/BGA{pcb[3]}/{pcb[6]}",
            zones_str,
            f"{top['setting']['chain_speed']}",
            f"{top['features']['peak_temp']:.1f}",
            f"{ycol}{yield_score:.3f}[/]",
        )
    console.print(table)

    step("人确认下发（演示：写入 audit_log）")
    log.info("  -> 工艺员在 Web 点『采纳并下发』，经安全校验后写 PLC"
             "（见 api/main.py /dispatch，用代理模型预测曲线校验工艺窗口）")

    step("质量反馈回灌")
    record_feedback("R-260714-0001", "L-0001", accepted=True,
                   actual_defect_type="无", repair=False, operator_id="OP-028")
    log.info("[green]闭环完成，下次训练将吸收本次反馈（数据飞轮）。[/]")


def pipeline_samples(limit=20):
    """展示数据库中的样本数据（默认 20 条），用于核对造数/训练数据分布。

    用法：
        python demo.py samples           # 展示前 20 条
        python demo.py samples 50       # 展示前 50 条
    """
    init_db()
    migrate_db()
    session = SessionLocal()
    try:
        total = session.query(ReflowRun).count()
        runs = session.query(ReflowRun).limit(limit).all()
        if not runs:
            log.warning(f"数据库共 {total} 条炉次样本，暂无可展示数据，请先运行：python demo.py train")
            return
        log.info(f"数据库共 {total} 条炉次样本，以下展示前 {len(runs)} 条。")
        # 预取关联维表/结果/特征，避免逐行查询
        pids = {r.product_id for r in runs}
        boms = {b.product_id: b for b in session.query(Bom).filter(Bom.product_id.in_(pids)).all()}
        lots = {q.lot_id: q for q in session.query(QualityResult)
                .filter(QualityResult.lot_id.in_([r.lot_id for r in runs])).all()}
        rids = {r.run_id for r in runs}
        feats = {f.run_id: f for f in session.query(ProfileFeature)
                 .filter(ProfileFeature.run_id.in_(rids)).all()}

        table = make_table(f"样本数据（共 {total} 条，展示前 {len(runs)} 条）", [
            ("炉次", "left"), ("PCB", "left"), ("板参数", "left"),
            ("8 段温区(℃)", "left"), ("链速", "right"), ("峰值(℃)", "right"),
            ("缺陷", "left"), ("AOI", "center"),
        ])
        from collections import Counter
        dcount = Counter()
        for r in runs:
            bom = boms.get(r.product_id)
            bpar = (f"厚{bom.thickness_mm}mm/铜{bom.copper_area_pct}%/BGA{bom.bga_count}"
                    if bom else "-")
            zones = [getattr(r, f"zone{j}_temp") for j in range(1, CONFIG.zone_num + 1)]
            zones_str = " / ".join(f"{z:.0f}" for z in zones)
            feat = feats.get(r.run_id)
            peak = feat.peak_temp if feat else 0.0
            qr = lots.get(r.lot_id)
            dt = qr.defect_type if qr else "-"
            aoi = qr.aoi_result if qr else "-"
            dcol = "[green]无[/]" if dt == "无" else f"[red]{dt}[/]"
            acol = ("[green]PASS[/]" if aoi == "PASS" else ("[red]FAIL[/]" if aoi == "FAIL" else aoi))
            dcount[dt] += 1
            table.add_row(r.run_id, r.product_id, bpar, zones_str,
                          f"{r.chain_speed}", f"{peak:.1f}", dcol, acol)
        console.print(table)
        log.info("缺陷分布：" + "，".join(f"{k}={v}" for k, v in dcount.most_common()))
    finally:
        session.close()


def _evaluate():
    """评估三模型并对照验收线打印。"""
    session = SessionLocal()
    try:
        X_set, Y_curve, X_feat, Y_def = load_training_data(session)
    finally:
        session.close()

    surr = joblib.load(os.path.join(CONFIG.model_dir, "surrogate.pkl"))
    qual = joblib.load(os.path.join(CONFIG.model_dir, "quality.pkl"))
    qual.set_inference_threads(1)   # 在线推理单/少样本，关并行最快

    table = make_table("模型评估（对照验收线）", [
        ("模型 / 指标", "left"), ("数值", "right"), ("验收线", "right"), ("结论", "center"),
    ])

    # 代理模型
    s_rep = evaluate_surrogate(surr, X_set, Y_curve)
    rmse = s_rep["mean_rmse_c"]
    table.add_row("代理模型 曲线预测 RMSE", f"{rmse:.2f} ℃", "≤ 5 ℃",
                  "[green]PASS[/]" if rmse <= 5 else "[red]FAIL[/]")

    # 质量模型（内部 CV，不污染主模型）
    q_rep = cross_validate_quality(X_feat, Y_def)
    if q_rep:
        auc = q_rep["macro_auc"]
        concl = "[green]PASS[/]" if (not np.isnan(auc) and auc >= 0.85) else "[red]FAIL[/]"
        table.add_row("质量模型 macro-AUC", f"{auc:.3f}", "≥ 0.85",
                      f"{concl} (n={q_rep['n_test']})")
        for i, a in q_rep["per_label_auc"].items():
            label = CONFIG.defect_labels[i] if i < len(CONFIG.defect_labels) else f"#{i}"
            av = f"{a:.3f}" if not (isinstance(a, float) and np.isnan(a)) else "N/A(单类)"
            table.add_row(f"    └ {label} AUC", av, "-", "")

    # 推荐引擎：约束满足率 + 平均预测良率（覆盖全部 PCB，验证差异化）
    rec = ReflowRecommender(surr, qual, featurizer, ProcessWindow(CONFIG.demo_window))
    test_cases = [(_bom_dict(pcb), pcb[6]) for pcb in PCB_TYPES]
    r_rep = evaluate_recommender(rec, test_cases)
    sat = r_rep["constraint_satisfy_rate"]
    table.add_row("推荐引擎 约束满足率", f"{sat*100:.0f}%", "100%",
                  "[green]PASS[/]" if sat >= 1.0 else "[red]FAIL[/]")
    yld = r_rep["avg_predicted_yield"]
    table.add_row("推荐引擎 平均预测良率", f"{yld:.3f}", "≥ 0.90",
                  "[green]PASS[/]" if yld >= 0.90 else "[red]FAIL[/]")
    console.print(table)


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    banner("回流焊 AI 工艺优化 · 演示",
           "train=造数训练 | recommend [N]=新板推荐 | samples [N]=样本数据 | (默认) 先训再推")
    if mode == "train":
        pipeline_train()
    elif mode == "recommend":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        pipeline_recommend(n)
    elif mode == "samples":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        pipeline_samples(n)
    else:  # "all"
        pipeline_train()
        pipeline_recommend()
