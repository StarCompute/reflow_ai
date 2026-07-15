# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""通用评估：对照验收线输出指标。

- 约束满足率：top-1 推荐经安全网关后是否仍落在工艺窗口内（验收线 100%）。
- 平均预测 goodness：top-1 推荐的质量评分（回流焊=良率，高炉=出料品质…）。
- 代理模型 RMSE（有信号时）：曲线预测误差（℃）。
"""
import numpy as np

from .models.recommender import Constraint


def evaluate(engine, test_contexts, top_k: int = 1):
    satisfy, goods = 0, []
    n = len(test_contexts)
    for ctx in test_contexts:
        res = engine.recommend(ctx, top_k=top_k)[0]
        ok, _, _ = engine.dispatch_check(res["setting"], ctx)
        if ok:
            satisfy += 1
        goods.append(res["score"])
    # 代理模型 RMSE（有信号时）
    rmse = None
    if engine.sim is not None and engine.surrogate is not None:
        errs = []
        recs = engine.synth.generate(50, seed=7)
        for r in recs:
            if r.signal is None:
                continue
            pred = engine.surrogate.predict(r.setting)
            errs.append(np.sqrt(np.mean((pred - np.asarray(r.signal)) ** 2)))
        if errs:
            rmse = float(np.mean(errs))
    return {
        "constraint_satisfy_rate": satisfy / n if n else 0.0,
        "avg_goodness": float(np.mean(goods)) if goods else 0.0,
        "surrogate_rmse_c": rmse,
        "n_cases": n,
    }
