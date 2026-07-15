# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""示例 2 · 黄铜选矿高炉：无信号 + 离散枚举择优 + 评分回归。

展示「类似但不同」的业务如何零改引擎接入：旋钮是离散的（选哪座炉），
质量是直接评分（出料品质 %），框架自动走「枚举择优」分支。
运行：python -m proctune.examples.run_blast_furnace
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from proctune.core.pipeline import ProcessTuningEngine
from proctune.core.evaluate import evaluate
from proctune.adapters.blast_furnace.profile import BLAST_FURNACE_PROFILE
from proctune.adapters.blast_furnace.furnace import FurnaceFeatureExtractor
from proctune.adapters.blast_furnace.synthetic import BlastFurnaceSynthetic


def main():
    # 无 simulator（没有过程信号）；无 search（离散旋钮不需要 BO）
    engine = ProcessTuningEngine(
        profile=BLAST_FURNACE_PROFILE,
        simulator=None,
        extractor=FurnaceFeatureExtractor(),
        synth=BlastFurnaceSynthetic(seed=42),
        search=None,
        model_dir="./models_bf",
    )
    n = engine.train(2000)
    print(f"[blast_furnace] 训练完成，样本 {n} 条")

    # 新批次：给定进料含量，推荐选哪座炉
    batches = [{"content": 30.0}, {"content": 50.0}, {"content": 70.0}]
    for b in batches:
        results = engine.recommend(b, top_k=2)   # 返回两座炉的排序
        ranking = " > ".join(f"{r['setting']['furnace']}(品质{r['quality']['quality_score']:.1f})"
                             for r in results)
        best = results[0]
        print(f"  进料 {b['content']:.0f}% → 推荐 {best['setting']['furnace']} 炉 "
              f"(预测出料品质 {best['quality']['quality_score']:.1f}) | 排序: {ranking}")

    rep = evaluate(engine, batches)
    print(f"[blast_furnace] 评估: {rep}")


if __name__ == "__main__":
    main()
