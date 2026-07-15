# Copyright (c) 2026 蒲俊杰（Pu Junjie）. All rights reserved.
# 许可见 https://github.com/StarCompute/reflow_ai/blob/main/proctune/LICENSE.md
# 个人使用（含个人商业）免费；企业/组织商业使用需获得授权。

"""基础冒烟测试：不依赖外部服务 / 数据库。

运行：  pip install pytest && pytest tests/ -q
"""
import numpy as np
from config import CONFIG
from features.extractor import extract_curve_features, build_model_input
from models.surrogate import SurrogateModel, SimpleThermalSolver
from models.quality import QualityModel
from models.recommender import ReflowRecommender, ProcessWindow
from data.validate import RED_LINES
from training.evaluate import cross_validate_quality, evaluate_surrogate


def test_extractor_shape():
    t = np.linspace(0, 240, 180)
    curve = SimpleThermalSolver().predict({"zones": [200] * 10, "chain_speed": 80})
    f = extract_curve_features(t, curve)
    assert len(f) == 8
    vec = build_model_input(
        f, {"thickness_mm": 1.6, "copper_area_pct": 30, "bga_count": 0,
             "max_bga_size_mm": 0, "component_density": 10}, "SAC305")
    assert vec.shape == (16,)


def test_red_lines_defined():
    assert set(RED_LINES) == {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"}


def test_window_penalty():
    w = ProcessWindow(CONFIG.process_window)
    assert w.penalty({"peak_temp": 243}) == 0.0
    assert w.penalty({"peak_temp": 300}) > 0.0
    assert w.penalty({"chain_speed": 10}) > 0.0


def test_surrogate_fit_predict():
    s = SurrogateModel()
    X = [{"zones": [200] * 10, "chain_speed": 80} for _ in range(5)]
    Y = np.stack([SimpleThermalSolver().predict(x) for x in X])
    s.fit(X, Y)
    out = s.predict(X[0])
    assert out.shape == (6, 180)


def test_quality_fit_predict_proba():
    q = QualityModel()
    rng = np.random.RandomState(0)
    X = rng.rand(60, 16)
    Y = np.zeros((60, 5), int)
    Y[:30, 0] = 1
    Y[30:, 1] = 1
    q.fit(X, Y)
    p = q.predict_proba(X[:3])
    assert p.shape == (3, 5)
    assert np.all((p >= 0) & (p <= 1))


def test_recommender_returns_topk():
    surr = SurrogateModel()
    X = [{"zones": [200] * 10, "chain_speed": 80} for _ in range(8)]
    Y = np.stack([SimpleThermalSolver().predict(x) for x in X])
    surr.fit(X, Y)
    q = QualityModel()
    q.fit(np.random.RandomState(1).rand(40, 16),
           np.zeros((40, 5), int))
    rec = ReflowRecommender(surr, q, __import__("features.extractor", fromlist=["extractor"]),
                            ProcessWindow(CONFIG.process_window))
    res = rec.recommend({"thickness_mm": 1.6, "copper_area_pct": 30, "bga_count": 0,
                         "max_bga_size_mm": 0, "component_density": 10}, "SAC305", top_k=3)
    assert len(res) == 3
    assert all(0.0 <= r["score"] <= 1.0 for r in res)


def test_evaluate_surrogate_runs():
    s = SurrogateModel()
    X = [{"zones": [200] * 10, "chain_speed": 80} for _ in range(6)]
    Y = np.stack([SimpleThermalSolver().predict(x) for x in X])
    s.fit(X, Y)
    rep = evaluate_surrogate(s, X, Y)
    assert "mean_rmse_c" in rep and rep["mean_rmse_c"] >= 0


def test_cross_validate_quality_runs():
    rng = np.random.RandomState(2)
    X = rng.rand(80, 16)
    Y = (rng.rand(80, 5) > 0.7).astype(int)
    rep = cross_validate_quality(X, Y)
    assert rep is not None
    assert "macro_auc" in rep
