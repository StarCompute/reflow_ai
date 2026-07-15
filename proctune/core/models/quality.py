# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""模型2 · 通用质量模型。

支持两种业务：
  - DefectQualityModel：多标签缺陷分类（回流焊/注塑），goodness = 联合良率 ∏(1-p)。
  - ScoreQualityModel  ：连续评分回归（黄铜高炉），goodness = 评分/scale 裁剪到 [0,1]。

两者对外暴露统一接口：fit / goodness(X) -> (N,) / report(X_row) -> (dict, score)，
推荐引擎与评估只依赖 goodness，因此业务间可互换。
"""
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class DefectQualityModel:
    """多标签缺陷分类：输入向量 -> 各缺陷概率；goodness = 整件良率。"""

    def __init__(self, n_labels: int, n_estimators: int = 200):
        base = RandomForestClassifier(n_estimators=n_estimators, max_depth=8,
                                      n_jobs=-1, random_state=42,
                                      class_weight="balanced")
        self.model = MultiOutputClassifier(base)
        self.n_labels = n_labels

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def set_inference_threads(self, n_jobs: int = 1):
        for est in self.model.estimators_:
            est.n_jobs = n_jobs

    def predict_proba(self, X) -> np.ndarray:
        probas = self.model.predict_proba(X)
        out = []
        for est, p in zip(self.model.estimators_, probas):
            cls = est.classes_
            if p.shape[1] == 2:
                out.append(p[:, 1])
            elif cls[0] == 1:
                out.append(p[:, 0])
            else:
                out.append(1.0 - p[:, 0])
        return np.stack(out, axis=1)

    def goodness(self, X) -> np.ndarray:
        p = self.predict_proba(X)
        return np.prod(1.0 - np.clip(p, 0.0, 1.0), axis=1)

    def report(self, X_row) -> (dict, float):
        X_row = np.asarray(X_row, dtype=float).reshape(1, -1)
        p = self.predict_proba(X_row)[0]
        labels = {f"defect_{i}_p": float(p[i]) for i in range(self.n_labels)}
        return labels, float(self.goodness(X_row)[0])


class ScoreQualityModel:
    """连续评分回归：输入向量 -> 质量评分；goodness = 评分/scale。"""

    def __init__(self, scale: float = 100.0, n_estimators: int = 200):
        self.model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=42)
        self.scale = scale

    def fit(self, X, Y):
        self.model.fit(X, np.asarray(Y, dtype=float).ravel())

    def set_inference_threads(self, n_jobs: int = 1):
        self.model.n_jobs = n_jobs

    def predict_score(self, X) -> np.ndarray:
        return self.model.predict(X)

    def goodness(self, X) -> np.ndarray:
        return np.clip(self.predict_score(X) / self.scale, 0.0, 1.0)

    def report(self, X_row) -> (dict, float):
        X_row = np.asarray(X_row, dtype=float).reshape(1, -1)
        s = float(self.predict_score(X_row)[0])
        return {"quality_score": s}, float(np.clip(s / self.scale, 0.0, 1.0))
