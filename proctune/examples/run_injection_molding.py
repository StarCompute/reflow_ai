# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""示例 3 · 注塑：无信号 + 连续 BO + 缺陷分类（旋钮直连质量）。

展示「没有中间信号、质量直连旋钮」的业务也能复用同一引擎，
框架自动走「连续贝叶斯优化」分支。
运行：python -m proctune.examples.run_injection_molding
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from proctune.core.pipeline import ProcessTuningEngine
from proctune.core.evaluate import evaluate
from proctune.adapters.injection_molding.profile import INJECTION_PROFILE
from proctune.adapters.injection_molding.features import MoldingFeatureExtractor
from proctune.adapters.injection_molding.synthetic import InjectionSynthetic
from proctune.adapters.injection_molding.search import InjectionSearchStrategy


def main():
    engine = ProcessTuningEngine(
        profile=INJECTION_PROFILE,
        simulator=None,                       # 无过程信号
        extractor=MoldingFeatureExtractor(),  # 旋钮+上下文 → 质量输入
        synth=InjectionSynthetic(),
        search=InjectionSearchStrategy(),     # 连续 BO
        model_dir="./models_im",
    )
    n = engine.train(2000)
    print(f"[injection_molding] 训练完成，样本 {n} 条")

    parts = [
        {"material_code": "PP", "wall_thickness": 1.5},
        {"material_code": "ABS", "wall_thickness": 2.5},
        {"material_code": "PC", "wall_thickness": 3.5},
    ]
    for p in parts:
        results = engine.recommend(p, top_k=1)
        top = results[0]
        s = top["setting"]
        print(f"  制件({p['material_code']}/壁厚{p['wall_thickness']}mm) → "
              f"料温{s['barrel_temp']:.0f}℃ 注射{s['inject_pressure']:.0f}MPa "
              f"保压{s['holding_time']:.0f}s 模温{s['mold_temp']:.0f}℃ | "
              f"预测良率 {top['score']:.3f} | 缺陷概率 {top['quality']}")

    rep = evaluate(engine, parts)
    print(f"[injection_molding] 评估: {rep}")


if __name__ == "__main__":
    main()
