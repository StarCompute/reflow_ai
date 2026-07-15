# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""通用流程编排：训练 / 推荐 / 安全网关 / 反馈。

与业务无关：所有业务差异都通过 BusinessProfile + 适配器注入。
生产环境可把 SyntheticGenerator 换成真实数据适配器（实现同样的 generate 接口）。
"""
import os
import joblib
import numpy as np

from .abstractions import BusinessProfile, Record
from .interfaces import SignalSimulator, FeatureExtractor, SyntheticGenerator, SearchStrategy
from .models.surrogate import SurrogateModel
from .models.quality import DefectQualityModel, ScoreQualityModel
from .models.recommender import Recommender, Constraint


class ProcessTuningEngine:
    def __init__(self, profile: BusinessProfile,
                 simulator: SignalSimulator,
                 extractor: FeatureExtractor,
                 synth: SyntheticGenerator,
                 search: SearchStrategy = None,
                 model_dir: str = None):
        self.profile = profile
        self.sim = simulator
        self.ext = extractor
        self.synth = synth
        self.search = search
        self.model_dir = model_dir or os.path.join(os.getcwd(), "models_artifacts")
        self.surrogate = None
        self.quality = None
        self.recommender = None

    # --------------------------- 训练 ---------------------------
    def train(self, n: int = 2000, seed: int = None):
        records = self.synth.generate(n, seed=seed)
        # 1) 代理模型（有信号才训）
        if self.sim is not None:
            signals = [r.signal for r in records if r.signal is not None]
            settings = [r.setting for r in records if r.signal is not None]
            self.surrogate = SurrogateModel(self.sim,
                                            n_points=self.profile.signal.n_points or 180)
            self.surrogate.fit(settings, np.array(signals))
        # 2) 质量模型
        self.quality = self._build_quality()
        X, Y = [], []
        for r in records:
            feats = (self.ext.extract_features(r.signal)
                     if (self.sim is not None and r.signal is not None) else {})
            X.append(self.ext.build_quality_input(r.setting, r.context, feats))
            if self.profile.quality.kind == "defect":
                Y.append([1.0 if r.quality_label == lab else 0.0
                          for lab in self.profile.quality.defect_labels])
            else:
                Y.append([r.quality_score])
        self.quality.fit(np.array(X), np.array(Y))
        self.quality.set_inference_threads(1)
        # 3) 推荐引擎
        self.recommender = Recommender(
            self.profile, self.sim, self.ext, self.quality,
            Constraint(self.profile.constraints.window), self.search,
            n_points=self.profile.signal.n_points or 180)
        self._save()
        return len(records)

    def _build_quality(self):
        q = self.profile.quality
        if q.kind == "defect":
            return DefectQualityModel(len(q.defect_labels))
        return ScoreQualityModel(q.scale)

    def _save(self):
        os.makedirs(self.model_dir, exist_ok=True)
        if self.surrogate is not None:
            joblib.dump(self.surrogate, os.path.join(self.model_dir, "surrogate.pkl"))
        joblib.dump(self.quality, os.path.join(self.model_dir, "quality.pkl"))

    def load(self):
        surr_path = os.path.join(self.model_dir, "surrogate.pkl")
        qual_path = os.path.join(self.model_dir, "quality.pkl")
        if not os.path.exists(qual_path):
            raise RuntimeError("未找到已训模型，请先 engine.train()")
        self.quality = joblib.load(qual_path)
        self.quality.set_inference_threads(1)
        if self.sim is not None and os.path.exists(surr_path):
            self.surrogate = joblib.load(surr_path)
        self.recommender = Recommender(
            self.profile, self.sim, self.ext, self.quality,
            Constraint(self.profile.constraints.window), self.search,
            n_points=self.profile.signal.n_points or 180)
        return self

    # --------------------------- 推荐 ---------------------------
    def recommend(self, context: dict, top_k: int = 3, env: dict = None) -> list:
        if self.recommender is None:
            self.load()
        ctx = self.profile.context.encode(context)
        return self.recommender.recommend(ctx, top_k=top_k, env=env)

    # --------------------------- 安全网关 ---------------------------
    def dispatch_check(self, setting: dict, context: dict, env: dict = None):
        """下发前校验：预测信号 → 特征 → 工艺窗口罚分；越界即拦截。"""
        ctx = self.profile.context.encode(context)
        if self.sim is not None:
            feats = self.ext.extract_features(self.sim.predict(setting))
        else:
            feats = {}
        pen = Constraint(self.profile.constraints.window).penalty(feats)
        return pen <= 1e-6, pen, feats
