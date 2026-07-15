# Copyright (c) 2026 蒲俊杰（Pu Junjie）. All rights reserved.
# 许可见 https://github.com/StarCompute/reflow_ai/blob/main/proctune/LICENSE.md
# 个人使用（含个人商业）免费；企业/组织商业使用需获得授权。

"""应用层 API（FastAPI）。

对应 V2 §9 API 设计。演示实现 recommend / dispatch / feedback / health。
正式部署加 Bearer Token 鉴权（见 V2 §9 / §11 安全）。

安全网关（/dispatch）实现 V2 §7 硬约束：下发前用代理模型预测设定将产生的
炉温曲线，反算曲线特征并校验工艺窗口，越界 100% 拦截（写 audit REJECT）。
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os
import numpy as np
from data.db import SessionLocal, AuditLog
from models.surrogate import SurrogateModel
from models.quality import QualityModel
from models.recommender import ReflowRecommender, ProcessWindow
import features.extractor as featurizer
from config import CONFIG
from utils.logger import get_logger

log = get_logger("api")
app = FastAPI(title="Reflow AI API", version="2.0")

# 启动加载模型（生产经 MLflow Registry 拉取指定版本）
surr = joblib.load(os.path.join(CONFIG.model_dir, "surrogate.pkl"))
qual = joblib.load(os.path.join(CONFIG.model_dir, "quality.pkl"))


class RecommendReq(BaseModel):
    """推荐请求体。"""
    request_id: str = Field("rec-demo", description="请求追踪 ID，原样返回")
    product_id: str = Field(..., description="产品/板型 ID（关联 bom 维表）")
    solder_paste: str = Field(..., description="本次过炉使用的锡膏型号")
    bom: dict = Field(..., description="BOM 特征：{thickness_mm, copper_area_pct, "
                                       "bga_count, max_bga_size_mm, component_density}")
    constraints: dict = Field(default_factory=dict,
                             description="可选：覆盖工艺窗口，如 {'peak_temp':[210,235]}")


class DispatchReq(BaseModel):
    """下发请求体（经安全网关校验）。"""
    request_id: str = Field(..., description="请求追踪 ID")
    oven_id: str = Field(..., description="目标回流焊炉 ID，如 OVEN-03")
    confirmed_by: str = Field(..., description="确认下发的工艺员工号")
    comment: str = Field("", description="备注")
    new_params: dict = Field(default_factory=dict,
                             description="下发参数：{zones:[8 个温度(℃)], chain_speed:数值(cm/min)}")


class FeedbackReq(BaseModel):
    """质量反馈请求体（回灌训练样本池）。"""
    run_id: str = Field(..., description="关联炉次 ID，如 R-260714-0001")
    lot_id: str = Field(..., description="批次号，关联 quality_result")
    accepted: bool = Field(True, description="AI 推荐是否被工艺员采纳")
    defect_type: str = Field("无", description="实际质检缺陷类型：无/虚焊/桥连/立碑/空洞/锡珠/其他")
    repair: bool = Field(False, description="是否返修")
    operator_id: str = Field(None, description="操作人工号")


def _build_window(overrides: dict):
    """合并默认工艺窗口与请求级覆盖。"""
    w = dict(CONFIG.process_window)
    for k, v in (overrides or {}).items():
        if k in w and isinstance(v, (list, tuple)) and len(v) == 2:
            w[k] = (float(v[0]), float(v[1]))
    return ProcessWindow(w)


@app.post("/api/v1/recommend")
def api_recommend(req: RecommendReq):
    """根据 BOM 推荐回流焊工艺（8 段温区 + 链速）。

    响应 recommendations 每项含：
      score(预测良率) / setting{zones[8 个温度], chain_speed} /
      features{peak_temp, tal, ramp_up, ...}
    示例：
      POST /api/v1/recommend
      {"product_id":"PCB-NX381","solder_paste":"SAC305",
       "bom":{"thickness_mm":1.6,"copper_area_pct":35,"bga_count":4,
               "max_bga_size_mm":25,"component_density":12}}
    """
    win = _build_window(req.constraints)
    rec = ReflowRecommender(surr, qual, featurizer, win)
    results = rec.recommend(req.bom, req.solder_paste, top_k=3)
    log.info(f"recommend product={req.product_id} solder={req.solder_paste} "
             f"-> {len(results)} candidates")
    return {"request_id": req.request_id, "recommendations": results}


@app.post("/api/v1/dispatch")
def api_dispatch(req: DispatchReq):
    """下发前安全网关：用代理模型预测曲线并校验工艺窗口（V2 §7 硬约束）。
    越界则写 audit REJECT 并返回 400 拦截；通过则写 audit DISPATCH 返回成功。
    """
    zones = req.new_params.get("zones")
    speed = req.new_params.get("chain_speed")
    if not isinstance(zones, list) or len(zones) != CONFIG.zone_num or speed is None:
        raise HTTPException(400, f"new_params 需含 zones(list,{CONFIG.zone_num}) 与 chain_speed")

    setting = {"zones": [float(z) for z in zones], "chain_speed": float(speed)}

    # —— 安全网关：代理模型预测曲线 → 校验工艺窗口（V2 §7 硬约束）——
    curve = surr.predict(setting)
    feats = featurizer.extract_curve_features(np.linspace(0, 240, 180), curve)
    pen = ProcessWindow(CONFIG.process_window).penalty(feats)

    sess = SessionLocal()
    try:
        if pen > 1e-6:
            log.warning(f"DISPATCH 拦截：设定将产生越界曲线 罚分={pen:.1f} "
                        f"oven={req.oven_id} by={req.confirmed_by}")
            sess.add(AuditLog(action="REJECT", request_id=req.request_id,
                              oven_id=req.oven_id, operator_id=req.confirmed_by,
                              new_params=req.new_params, result="BLOCKED"))
            sess.commit()
            raise HTTPException(
                400, f"设定将产生越界工艺曲线（罚分 {pen:.1f}），已 100% 拦截")
        sess.add(AuditLog(action="DISPATCH", request_id=req.request_id,
                          oven_id=req.oven_id, operator_id=req.confirmed_by,
                          new_params=req.new_params, result="SUCCESS"))
        sess.commit()
    finally:
        sess.close()
    log.info(f"DISPATCH 成功 oven={req.oven_id} by={req.confirmed_by}")
    return {"status": "dispatched", "oven_id": req.oven_id,
            "note": "经安全网关校验通过；生产写 PLC"}


@app.post("/api/v1/feedback")
def api_feedback(req: FeedbackReq):
    """记录质量反馈并回灌训练样本池（供 /retrain 吸收，形成数据飞轮）。"""
    from pipeline.feedback import record_feedback
    record_feedback(req.run_id, req.lot_id, req.accepted,
                    req.defect_type, req.repair, req.operator_id)
    return {"status": "feedback_recorded"}


@app.post("/api/v1/retrain")
def api_retrain():
    """数据飞轮闭环：用当前 DB 中（含反馈回灌的）样本重训三模型，
    并热加载新模型，使后续 recommend/dispatch 立即生效。"""
    from training.train import train_all
    train_all()
    global surr, qual
    surr = joblib.load(os.path.join(CONFIG.model_dir, "surrogate.pkl"))
    qual = joblib.load(os.path.join(CONFIG.model_dir, "quality.pkl"))
    qual.set_inference_threads(1)
    return {"status": "retrained"}


@app.get("/api/v1/health")
def api_health():
    """健康检查，返回 {"status":"ok"}。"""
    return {"status": "ok"}


@app.get("/")
def api_index():
    """API 概览：列出所有端点与说明（详见 /docs Swagger UI）。"""
    return {
        "service": "Reflow AI API",
        "version": "2.0",
        "endpoints": {
            "POST /api/v1/recommend": "BOM → 推荐 8 段温区工艺",
            "POST /api/v1/dispatch": "下发前安全网关校验（越界 100% 拦截）",
            "POST /api/v1/feedback": "质量反馈回灌训练池",
            "POST /api/v1/retrain": "用当前样本重训并热加载模型",
            "GET  /api/v1/health": "健康检查",
        },
        "docs": "/docs (Swagger UI)",
    }
