# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""对 proctune.easy.EasyTuner 的第三方独立验证脚本（不修改项目代码）。

验证目标：
  - 用「随机生成的样本数据 + 新工艺数据」验证 EasyTuner 的推荐能力；
  - 三种【格式完全不同】的样本，每种用 3 个不同随机种子各跑一遍（共 9 次）；
  - 每次 1000 条历史样本 + 一组新任务，比较「系统推荐」与「随机基线」的质量差异。

三种格式：
  A) CSV 文件 + 中文表头 + 缺陷文本列（注塑场景，复用 InjectionSynthetic）
  B) list[dict] 直接喂入 + 连续评分 0~100（回流焊场景，复用 ReflowSynthetic）
  C) CSV 文件 + 离散类别旋钮（电流密度档位）+ 缺陷文本列（电镀场景，自造规律）
"""
import os
import sys
import csv
import random

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))  # 让 import proctune 可用

from proctune.easy import EasyTuner
from proctune.adapters.injection_molding.synthetic import InjectionSynthetic
from proctune.adapters.injection_molding.profile import MATERIAL_MAP
from proctune.adapters.reflow.synthetic import ReflowSynthetic

SEEDS = [101, 202, 303]
N_SAMPLE = 1000


# ============================================================
# 格式 A：CSV + 中文表头 + 缺陷文本列（注塑）
# ============================================================
def gen_format_a(seed):
    recs = InjectionSynthetic().generate(N_SAMPLE, seed=seed)
    code2mat = {v: k for k, v in MATERIAL_MAP.items()}
    rows = []
    for r in recs:
        rows.append({
            "材料": code2mat[r.context["material_code"]],
            "壁厚": r.context["wall_thickness"],
            "料筒温度": r.setting["barrel_temp"],
            "注射压力": r.setting["inject_pressure"],
            "保压时间": r.setting["holding_time"],
            "模具温度": r.setting["mold_temp"],
            "质量": "OK" if r.quality_label == "无" else r.quality_label,
        })
    jobs = [
        {"材料": "PP", "壁厚": 1.2},
        {"材料": "PC", "壁厚": 3.5},
        {"材料": "ABS", "壁厚": 2.0},
        {"材料": "PP", "壁厚": 4.0},
        {"材料": "ABS", "壁厚": 0.9},
    ]
    return rows, jobs


# ============================================================
# 格式 B：list[dict] + 连续评分 0~100（回流焊）
#   直接用内存 dict，不落盘；质量列用 0~100 评分（无缺陷=100，越严重越低）
# ============================================================
def gen_format_b(seed):
    recs = ReflowSynthetic().generate(N_SAMPLE, seed=seed)
    sev = {"无": 0.0, "虚焊": 35.0, "桥连": 40.0, "立碑": 45.0, "空洞": 30.0, "锡珠": 25.0}
    rows = []
    for r in recs:
        # 评分 = 100 - 缺陷严重度 + 小噪声
        score = max(0.0, min(100.0, 100.0 - sev.get(r.quality_label, 50.0)
                             + random.Random(hash(r.quality_label + str(seed)) % (2**31)).uniform(-3, 3)))
        item = {"质量评分": round(score, 1)}
        for k, v in r.setting.items():
            item[k] = v
        for k, v in r.context.items():
            item[k] = v
        rows.append(item)
    jobs = []
    for t in (0.8, 1.6, 2.2):
        for c in (20.0, 45.0):
            for b in (1, 4):
                jobs.append({"thickness_mm": t, "copper_area_pct": c,
                             "bga_count": b, "max_bga_size_mm": 20.0,
                             "component_density": 10.0, "solder_paste": 1.0,
                             "env_temp": 25.0, "env_humidity": 50.0})
    knob_cols = [f"zone{i}_temp" for i in range(1, 9)] + ["chain_speed"]
    ctx_cols = ["thickness_mm", "copper_area_pct", "bga_count",
                "max_bga_size_mm", "component_density", "solder_paste",
                "env_temp", "env_humidity"]
    return rows, jobs, knob_cols, ctx_cols


# ============================================================
# 格式 C：CSV + 离散类别旋钮（电流密度档位）+ 缺陷文本列（电镀）
#   自造规律：电流密度档位 + 温度 + 时间 决定镀层缺陷
# ============================================================
def gen_format_c(seed):
    rng = random.Random(seed)
    levels = {"低": 1.0, "中": 2.0, "高": 3.0}
    mats = ["镀锌", "镀镍", "镀铬"]
    rows = []
    for _ in range(N_SAMPLE):
        mat = rng.choice(mats)
        level = rng.choice(list(levels.keys()))
        temp = round(rng.uniform(15, 45), 1)
        dur = round(rng.uniform(5, 40), 1)
        # 理想：中档电流 + 温度~30 + 时间~22
        d_burn = (levels[level] - 2.0) * 10 + (temp - 30) * 0.6 + (dur - 22) * 0.4 + rng.gauss(0, 4)
        d_thin = (2.0 - levels[level]) * 8 + (22 - dur) * 0.7 + rng.gauss(0, 4)
        d_pit = abs(temp - 30) - 8 + rng.gauss(0, 3)
        devs = {"烧焦": d_burn, "镀层薄": d_thin, "针孔": d_pit}
        label = "OK"
        best = max(devs.values())
        if best > 6:
            label = max(devs, key=devs.get)
        rows.append({"镀种": mat, "电流密度档": level, "温度": temp,
                     "时间": dur, "质量": label})
    jobs = [
        {"镀种": "镀锌", "温度": 28.0, "时间": 20.0},
        {"镀种": "镀镍", "温度": 32.0, "时间": 24.0},
        {"镀种": "镀铬", "温度": 30.0, "时间": 22.0},
        {"镀种": "镀锌", "温度": 40.0, "时间": 12.0},
        {"镀种": "镀镍", "温度": 18.0, "时间": 35.0},
    ]
    return rows, jobs


# ============================================================
# 隐藏规律（真值）：用于计算「推荐参数」和「随机参数」的真实良率
#   这些规律与造数器一致，但评估时用它独立打分，不依赖模型预测。
# ============================================================
def true_injection_ok(mat, wall, barrel, pressure, hold, mold):
    MAT_BASE = {"PP": 195.0, "ABS": 215.0, "PC": 212.0}
    MAT_IDX = {"PP": 0, "ABS": 1, "PC": 2}
    import numpy as np
    d_short = (MAT_BASE[mat] + 8 * wall) - barrel
    d_flash = pressure - (60 + 15 * wall)
    d_sink = (12 - hold) + 2 * wall
    d_warp = mold - (52 + 5 * MAT_IDX[mat])
    return max(d_short, d_flash, d_sink, d_warp) <= 6


def true_reflow_score(bom, setting):
    # 简化：用峰值温区(zone8)与理想带比较；理想 peak ≈ 235 + 厚度/铜/球的线性
    t, c, b = bom["thickness_mm"], bom["copper_area_pct"], bom["bga_count"]
    ideal_peak = 235 + 10 * t + 0.15 * c + 1.5 * b
    peak = setting["zone8_temp"]
    dev = abs(peak - ideal_peak)
    # 评分 100 - 偏离惩罚
    return max(0.0, 100.0 - dev * 3.0)


def true_electro_ok(level, temp, dur):
    levels = {"低": 1.0, "中": 2.0, "高": 3.0}
    d_burn = (levels[level] - 2.0) * 10 + (temp - 30) * 0.6 + (dur - 22) * 0.4
    d_thin = (2.0 - levels[level]) * 8 + (22 - dur) * 0.7
    d_pit = abs(temp - 30) - 8
    return max(d_burn, d_thin, d_pit) <= 6


def random_setting(tuner, rng):
    setting = {}
    for (k, kind, info) in tuner._knob_defs:
        if kind == "continuous":
            lo, hi = info
            setting[k] = rng.uniform(lo, hi)
        else:
            setting[k] = rng.choice(info)
    return setting


def evaluate_a(tuner, jobs, rng, n_rand=200):
    sys_ok = 0
    for j in jobs:
        rec = tuner.recommend_one(j, top_k=1)[0]
        if true_injection_ok(j["材料"], j["壁厚"], rec["推荐_料筒温度"],
                             rec["推荐_注射压力"], rec["推荐_保压时间"], rec["推荐_模具温度"]):
            sys_ok += 1
    rand_ok = 0
    for j in jobs:
        for _ in range(n_rand):
            s = random_setting(tuner, rng)
            if true_injection_ok(j["材料"], j["壁厚"], s["料筒温度"],
                                 s["注射压力"], s["保压时间"], s["模具温度"]):
                rand_ok += 1
    return sys_ok / len(jobs), rand_ok / (len(jobs) * n_rand)


def _get(rec_or_setting, base_key):
    """从推荐结果(带'推荐_'前缀)或纯设定dict里取旋钮值。"""
    if base_key in rec_or_setting:
        return rec_or_setting[base_key]
    return rec_or_setting.get("推荐_" + base_key)


def evaluate_b(tuner, jobs, rng, n_rand=200):
    knob = tuner.knob_cols
    sys_scores, rand_scores = [], []
    for j in jobs:
        rec = tuner.recommend_one(j, top_k=1)[0]
        setting = {k: _get(rec, k) for k in knob}
        sys_scores.append(true_reflow_score(j, setting))
    for j in jobs:
        for _ in range(n_rand):
            s = random_setting(tuner, rng)
            rand_scores.append(true_reflow_score(j, s))
    return sum(sys_scores) / len(sys_scores), sum(rand_scores) / len(rand_scores)


def evaluate_c(tuner, jobs, rng, n_rand=200):
    sys_ok = 0
    for j in jobs:
        rec = tuner.recommend_one(j, top_k=1)[0]
        if true_electro_ok(rec["推荐_电流密度档"], rec["推荐_温度"], rec["推荐_时间"]):
            sys_ok += 1
    rand_ok = 0
    for j in jobs:
        for _ in range(n_rand):
            s = random_setting(tuner, rng)
            if true_electro_ok(s["电流密度档"], s["温度"], s["时间"]):
                rand_ok += 1
    return sys_ok / len(jobs), rand_ok / (len(jobs) * n_rand)


# ============================================================
# 主流程
# ============================================================
def run():
    print("=" * 70)
    print(f"proctune.easy.EasyTuner 独立验证  样本量={N_SAMPLE}  种子={SEEDS}")
    print("=" * 70)

    # ---- 格式 A：CSV + 缺陷文本 ----
    print("\n【格式 A】CSV 文件 + 中文表头 + 缺陷文本列（注塑）")
    for seed in SEEDS:
        rows, jobs = gen_format_a(seed)
        csv_path = os.path.join(ROOT, f"_a_hist_{seed}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        tuner = EasyTuner(csv_path,
                          knob_cols=["料筒温度", "注射压力", "保压时间", "模具温度"],
                          context_cols=["材料", "壁厚"], quality_col="质量")
        tuner.train()
        rng = random.Random(seed * 7 + 1)
        sys_m, base_m = evaluate_a(tuner, jobs, rng)
        print(f"  seed={seed:>3}  推荐真实良率={sys_m:.3f}  随机基线={base_m:.3f}  "
              f"提升={(sys_m-base_m)*100:+.1f}pp  [{tuner.summary()}]")
        os.remove(csv_path)

    # ---- 格式 B：list[dict] + 连续评分 ----
    print("\n【格式 B】list[dict] 直接喂入 + 连续评分0~100（回流焊）")
    for seed in SEEDS:
        rows, jobs, knob_cols, ctx_cols = gen_format_b(seed)
        tuner = EasyTuner(rows, knob_cols=knob_cols, context_cols=ctx_cols,
                          quality_col="质量评分", quality_kind="score")
        tuner.train()
        rng = random.Random(seed * 7 + 1)
        sys_m, base_m = evaluate_b(tuner, jobs, rng)
        print(f"  seed={seed:>3}  推荐真实评分={sys_m:.2f}  随机基线={base_m:.2f}  "
              f"提升={sys_m-base_m:+.2f}  [{tuner.summary()}]")

    # ---- 格式 C：CSV + 离散类别旋钮 + 缺陷文本 ----
    print("\n【格式 C】CSV 文件 + 离散类别旋钮(电流密度档) + 缺陷文本列（电镀）")
    for seed in SEEDS:
        rows, jobs = gen_format_c(seed)
        csv_path = os.path.join(ROOT, f"_c_hist_{seed}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        tuner = EasyTuner(csv_path,
                          knob_cols=["电流密度档", "温度", "时间"],
                          context_cols=["镀种"], quality_col="质量")
        tuner.train()
        rng = random.Random(seed * 7 + 1)
        sys_m, base_m = evaluate_c(tuner, jobs, rng)
        print(f"  seed={seed:>3}  推荐真实良率={sys_m:.3f}  随机基线={base_m:.3f}  "
              f"提升={(sys_m-base_m)*100:+.1f}pp  [{tuner.summary()}]")
        os.remove(csv_path)

    print("\n验证完成。")


if __name__ == "__main__":
    run()
