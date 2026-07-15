# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""示例 1 · 回流焊：信号介导 + 连续 BO + 缺陷分类。

展示「如何把现有回流焊业务接入通用框架」——只需组装 4 个适配器 + 1 份画像。
运行：python -m proctune.examples.run_reflow
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from proctune.core.pipeline import ProcessTuningEngine
from proctune.core.evaluate import evaluate
from proctune.adapters.reflow.profile import REFLOW_PROFILE
from proctune.adapters.reflow.thermal_solver import ThermalSolver
from proctune.adapters.reflow.curve_features import CurveFeatureExtractor
from proctune.adapters.reflow.synthetic import ReflowSynthetic
from proctune.adapters.reflow.search import ReflowSearchStrategy


def main():
    # 1) 组装引擎：画像 + 适配器（业务差异全在这几行）
    engine = ProcessTuningEngine(
        profile=REFLOW_PROFILE,
        simulator=ThermalSolver(),          # 有信号：物理基线
        extractor=CurveFeatureExtractor(),  # 信号→特征→质量输入
        synth=ReflowSynthetic(),            # 造数（生产换真实数据适配器）
        search=ReflowSearchStrategy(),      # BO 搜索策略
        model_dir="./models_reflow",
    )

    # 2) 训练（造 2000 条 → 训代理 + 质量 + 推荐）
    n = engine.train(2000)
    print(f"[reflow] 训练完成，样本 {n} 条")

    # 3) 对新板推荐（业务侧只需传「这块板的 BOM 上下文」）
    new_boards = [
        {"thickness_mm": 1.6, "copper_area_pct": 35.0, "bga_count": 4,
         "max_bga_size_mm": 25.0, "component_density": 12.0, "solder_paste": "SAC305"},
        {"thickness_mm": 2.4, "copper_area_pct": 55.0, "bga_count": 4,
         "max_bga_size_mm": 35.0, "component_density": 16.0, "solder_paste": "SAC305"},
        {"thickness_mm": 0.8, "copper_area_pct": 15.0, "bga_count": 1,
         "max_bga_size_mm": 12.0, "component_density": 6.0, "solder_paste": "SN100C"},
    ]
    for board in new_boards:
        results = engine.recommend(board, top_k=1)
        top = results[0]
        zones = " / ".join(f"{top['setting'][f'zone{i}_temp']:.0f}" for i in range(1, 9))
        print(f"  板(厚{board['thickness_mm']}mm/铜{board['copper_area_pct']}%) "
              f"→ 推荐温区 {zones} | 链速 {top['setting']['chain_speed']:.0f} "
              f"| 预测良率 {top['score']:.3f}")

    # 4) 评估（约束满足率应为 100%，平均良率 ≥ 0.90）
    test_contexts = new_boards
    rep = evaluate(engine, test_contexts)
    print(f"[reflow] 评估: {rep}")

    # 5) 安全网关：下发前校验（越界 100% 拦截）
    ok, pen, feats = engine.dispatch_check(top["setting"], board)
    print(f"[reflow] 下发校验: {'通过' if ok else '拦截'} (罚分 {pen:.2f})")


if __name__ == "__main__":
    main()
