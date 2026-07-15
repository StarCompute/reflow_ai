"""训练 pipeline：读已有数据 → 特征 → 训三模型 → 落盘。

对应 V2 §13 CI/CD 与清单 2.5。
生产环境：本脚本由 CI 触发，模型经 MLflow Registry 版本化后灰度发布。
"""
import numpy as np
import joblib
import os
from data.db import SessionLocal, ReflowRun, QualityResult, ProfileFeature, Bom
from features.extractor import extract_curve_features, build_model_input
from models.surrogate import SurrogateModel, SimpleThermalSolver
from models.quality import QualityModel
from config import CONFIG

FEAT_KEYS = ["peak_temp", "tal", "ramp_up", "ramp_down",
              "delta_t", "soak_temp", "time_above_183", "curve_duration"]


def load_training_data(session):
    """从已有数据读取训练样本。

    曲线：真实环境从 InfluxDB profile_signal 读 tc1~6；
          演示无 InfluxDB 时用物理基线生成近似曲线。
    """
    runs = session.query(ReflowRun).all()
    X_set, Y_curve, X_feat, Y_def = [], [], [], []
    for r in runs:
        zones = [getattr(r, f"zone{i}_temp")
                 for i in range(1, CONFIG.zone_num + 1)]
        setting = {"zones": zones, "chain_speed": r.chain_speed}
        X_set.append(setting)
        Y_curve.append(SimpleThermalSolver().predict(setting))     # (6,180)，秒级时间轴

        bom = session.query(Bom).filter_by(product_id=r.product_id).first()
        bom_d = {
            "thickness_mm": bom.thickness_mm if bom else 1.6,
            "copper_area_pct": bom.copper_area_pct if bom else 30.0,
            "bga_count": bom.bga_count if bom else 0,
            "max_bga_size_mm": bom.max_bga_size_mm if bom else 0.0,
            "component_density": bom.component_density if bom else 10.0,
        }
        pf = session.query(ProfileFeature).filter_by(run_id=r.run_id).first()
        if pf is None:
            feats = extract_curve_features(
                np.linspace(0, 240, 180), SimpleThermalSolver().predict(setting))
        else:
            feats = {c: getattr(pf, c) for c in FEAT_KEYS}
        X_feat.append(build_model_input(feats, bom_d, r.solder_paste or "SAC305", {}))

        qr = session.query(QualityResult).filter_by(lot_id=r.lot_id).first()
        y = [0, 0, 0, 0, 0]
        if qr and qr.defect_type in CONFIG.defect_labels:
            y[CONFIG.defect_labels.index(qr.defect_type)] = 1
        Y_def.append(y)
    return X_set, np.array(Y_curve), np.array(X_feat), np.array(Y_def)


def train_all():
    session = SessionLocal()
    try:
        X_set, Y_curve, X_feat, Y_def = load_training_data(session)
        if len(X_set) == 0:
            print("⚠ 无训练数据，请先准备数据（见 demo.py seed_demo_data）。")
            return
        surr = SurrogateModel()
        surr.fit(X_set, Y_curve)
        qual = QualityModel()
        qual.fit(X_feat, Y_def)
        os.makedirs(CONFIG.model_dir, exist_ok=True)
        joblib.dump(surr, os.path.join(CONFIG.model_dir, "surrogate.pkl"))
        joblib.dump(qual, os.path.join(CONFIG.model_dir, "quality.pkl"))
        print(f"[OK] 训练完成：{len(X_set)} 条样本，模型已保存至 {CONFIG.model_dir}")
    finally:
        session.close()


if __name__ == "__main__":
    train_all()
