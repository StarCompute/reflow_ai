# Copyright (c) 2026 蒲俊杰（Pu Junjie）. All rights reserved.
# 许可见 https://github.com/StarCompute/reflow_ai/blob/main/proctune/LICENSE.md
# 个人使用（含个人商业）免费；企业/组织商业使用需获得授权。

"""模型评估与验收指标（对应清单 2.6 / V2 §12 验收）。

在 demo / CI 中调用，输出与验收线对照的量化指标：
- 质量模型：各缺陷 ROC-AUC、macro-AUC（验收线 AUC>=0.85）
- 代理模型：曲线预测 RMSE（℃）
- 推荐引擎：工艺窗口约束满足率（验收线 100% 拦截越界）、平均预测良率
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from models.quality import QualityModel
from utils.logger import get_logger

logger = get_logger("evaluate")


def cross_validate_quality(X, Y, test_size=0.25, seed=42):
    """内部 train/test 拆分评估质量模型泛化 AUC（不污染主模型）。"""
    if len(X) < 10:
        logger.warning("样本不足，跳过质量评估")
        return None
    Xtr, Xte, Ytr, Yte = train_test_split(
        X, Y, test_size=test_size, random_state=seed)
    m = QualityModel()
    m.fit(Xtr, Ytr)
    proba = m.predict_proba(Xte)
    per, aucs = {}, []
    for i in range(Yte.shape[1]):
        pos = int(Yte[:, i].sum())
        if 0 < pos < len(Yte):
            a = float(roc_auc_score(Yte[:, i], proba[:, i]))
        else:
            a = float("nan")          # 单类缺陷无法算 AUC
        per[i] = a
        if not np.isnan(a):
            aucs.append(a)
    return {
        "per_label_auc": per,
        "macro_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "n_test": len(Xte),
        "defect_pos_counts": [int(Yte[:, i].sum()) for i in range(Yte.shape[1])],
    }


def evaluate_surrogate(surr, X_set, Y_curve):
    """代理模型曲线预测 RMSE（℃）。"""
    errs = []
    for s, y in zip(X_set, Y_curve):
        pred = surr.predict(s)
        errs.append(np.sqrt(np.mean((pred - y) ** 2)))
    return {"mean_rmse_c": float(np.mean(errs))}


def evaluate_recommender(rec, test_cases):
    """test_cases: list[(bom, solder)]。

    返回约束满足率（top-1 设定经代理模型预测后是否仍落在工艺窗口内）
    与平均预测良率 P(良)=∏(1-P缺陷)。
    """
    satisfy, yields = 0, []
    for bom, solder in test_cases:
        res = rec.recommend(bom, solder, top_k=1)
        top = res[0]
        pen = rec.window.penalty(top["features"])
        if pen <= 1e-6:
            satisfy += 1
        yields.append(top["score"])
    n = len(test_cases)
    return {
        "constraint_satisfy_rate": satisfy / n if n else 0.0,
        "avg_predicted_yield": float(np.mean(yields)) if yields else 0.0,
        "n_cases": n,
    }
