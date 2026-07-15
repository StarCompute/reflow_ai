# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""极简示例：不懂 AI 的现场工程师如何只用两张表使用本系统。

本示例会：
  1) 生成「历史样本.csv」（过去的生产记录：调了哪些参数 + 产品属性 + 质量）
  2) 生成「新任务.csv」（这次要加工的产品属性）
  3) 三行代码训练 + 推荐，输出「推荐结果.csv」

你完全可以用自己工厂导出的真实 Excel/CSV 替换这两个文件，无需改任何代码逻辑。
"""
import os
import csv

from proctune.adapters.injection_molding.synthetic import InjectionSynthetic
from proctune.adapters.injection_molding.profile import MATERIAL_MAP
from proctune.easy import EasyTuner

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

HISTORY_CSV = os.path.join(DATA_DIR, "历史样本.csv")
NEWJOB_CSV = os.path.join(DATA_DIR, "新任务.csv")
OUT_CSV = os.path.join(DATA_DIR, "推荐结果.csv")
NEWBATCH_CSV = os.path.join(DATA_DIR, "本周新数据.csv")


def _make_history_csv(n=800):
    """用注塑规律造一批「过去的生产记录」，写成普通人看的表。"""
    recs = InjectionSynthetic().generate(n, seed=42)
    code2mat = {v: k for k, v in MATERIAL_MAP.items()}
    with open(HISTORY_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["材料", "壁厚", "料筒温度", "注射压力", "保压时间", "模具温度", "质量"])
        for r in recs:
            mat = code2mat[r.context["material_code"]]
            w.writerow([mat, r.context["wall_thickness"],
                        r.setting["barrel_temp"], r.setting["inject_pressure"],
                        r.setting["holding_time"], r.setting["mold_temp"],
                        r.quality_label])
    print(f"已生成历史样本：{HISTORY_CSV}（{n} 行）")


def _make_newjob_csv():
    jobs = [
        {"材料": "PP", "壁厚": 1.2},
        {"材料": "PC", "壁厚": 3.5},
        {"材料": "ABS", "壁厚": 2.0},
        {"材料": "PP", "壁厚": 4.0},
    ]
    with open(NEWJOB_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["材料", "壁厚"])
        w.writeheader()
        w.writerows(jobs)
    print(f"已生成新任务：{NEWJOB_CSV}（{len(jobs)} 行）")


def _make_newbatch_csv(n=200):
    """模拟「本周新生产的一批记录」（含质量结果），用于演示追加与重训。"""
    recs = InjectionSynthetic().generate(n, seed=7)
    code2mat = {v: k for k, v in MATERIAL_MAP.items()}
    with open(NEWBATCH_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["材料", "壁厚", "料筒温度", "注射压力", "保压时间", "模具温度", "质量"])
        for r in recs:
            mat = code2mat[r.context["material_code"]]
            w.writerow([mat, r.context["wall_thickness"],
                        r.setting["barrel_temp"], r.setting["inject_pressure"],
                        r.setting["holding_time"], r.setting["mold_temp"],
                        r.quality_label])
    print(f"已生成本周新数据：{NEWBATCH_CSV}（{n} 行）")


def main():
    _make_history_csv()
    _make_newjob_csv()

    # ===== 普通人只用这三行 =====
    tuner = EasyTuner(
        data=HISTORY_CSV,
        knob_cols=["料筒温度", "注射压力", "保压时间", "模具温度"],  # 系统要反推的参数
        context_cols=["材料", "壁厚"],                              # 产品属性（已知）
        quality_col="质量",                                        # 过去的质量结果
    )
    tuner.train()
    tuner.recommend_to_csv(NEWJOB_CSV, OUT_CSV, top_k=1)
    # ============================

    print("\n" + tuner.summary())
    print(f"\n推荐结果已写出：{OUT_CSV}\n")
    with open(OUT_CSV, encoding="utf-8-sig") as f:
        for line in f:
            print(line.rstrip())

    # 训练一次，下次直接复用（无需重新训练）
    tuner.save(os.path.join(DATA_DIR, "tuner_model.pkl"))
    print("\n模型已保存，下次可用 EasyTuner.load('data/tuner_model.pkl') 直接推荐。")

    # ===== 演示：有了新一批生产记录，怎么加进去并重训 =====
    _make_newbatch_csv()                       # 造一批"本周新数据"（含质量结果）
    merge_csv(NEWBATCH_CSV, HISTORY_CSV)       # 1) 并入历史样本表
    tuner = EasyTuner.load(os.path.join(DATA_DIR, "tuner_model.pkl"))
    tuner.add_data(NEWBATCH_CSV)               # 2) 把新样本加进模型（先攒着）
    print(f"\n并入新数据后样本数：{tuner.n_samples}")
    tuner.retrain()                            # 3) 攒够一批，重训一次
    tuner.recommend_to_csv(NEWJOB_CSV, os.path.join(DATA_DIR, "推荐结果_重训后.csv"), top_k=1)
    tuner.save(os.path.join(DATA_DIR, "tuner_model.pkl"))
    print("已用新数据重训，并写出 推荐结果_重训后.csv")


if __name__ == "__main__":
    main()
