# Copyright (c) 2026 蒲俊杰（Pu Junjie）. All rights reserved.
# 许可见 https://github.com/StarCompute/reflow_ai/blob/main/proctune/LICENSE.md
# 个人使用（含个人商业）免费；企业/组织商业使用需获得授权。

"""模型2 · 质量预测模型（Quality）。

曲线特征(8) + 板/锡膏元信息(8) → 5 类缺陷概率 + 良率。
实现：多标签分类（演示用 RandomForest；生产换 XGBoost，AUC>=0.85）。
对应 V2 §6 模型2 / 清单 2.3。
"""
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from features.extractor import FEATURE_NAMES
from config import CONFIG


class QualityModel:
    def __init__(self):
        # 生产替换：XGBClassifier(scale_pos_weight=15, ...) 包 MultiOutputClassifier
        base = RandomForestClassifier(n_estimators=200, max_depth=8,
                                     n_jobs=-1, random_state=42,
                                     class_weight="balanced")
        self.model = MultiOutputClassifier(base)

    def fit(self, X, Y):
        """X: (N,16); Y: (N,5) 0/1 多标签。"""
        self.model.fit(X, Y)

    def set_inference_threads(self, n_jobs=1):
        """在线单/少样本推理时关掉并行（避免线程启动开销远大于计算量）。

        训练仍用 n_jobs=-1（大数据并行加速）；加载模型准备推理前调用本方法，
        把每个基学习器的 n_jobs 设为 1，使单样本 predict_proba 从 ~140ms 降到数 ms。
        """
        for est in self.model.estimators_:
            est.n_jobs = n_jobs

    def predict_proba(self, X):
        """返回 (N,5) 正类(缺陷=1)概率。

        稳健处理：若某缺陷在训练集只有单一类别，sklearn 的
        predict_proba 仅返回 1 列，需按 classes_ 推断正类概率。
        """
        probas = self.model.predict_proba(X)
        out = []
        for est, p in zip(self.model.estimators_, probas):
            cls = est.classes_
            if p.shape[1] == 2:
                out.append(p[:, 1])
            elif cls[0] == 1:
                out.append(p[:, 0])          # 唯一类即正类
            else:
                out.append(1.0 - p[:, 0])   # 唯一类为负类 → P(1)=1-P(0)
        return np.stack(out, axis=1)           # (N,5)

    def explain(self, X_row):
        """top-3 特征重要性（树模型近似 SHAP，见 V2 §6 可解释）。"""
        est = self.model.estimators_[0]
        imp = est.feature_importances_
        order = np.argsort(imp)[::-1][:3]
        return [(FEATURE_NAMES[i], float(imp[i])) for i in order]
