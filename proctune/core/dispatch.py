# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""安全网关（演示用独立函数，等价于 engine.dispatch_check）。

生产部署：在真正写 PLC / 下发设备前调用，越界 100% 拦截并写审计日志。
"""
from .models.recommender import Constraint
from .interfaces import SignalSimulator, FeatureExtractor
from .abstractions import BusinessProfile


def check_dispatch(profile: BusinessProfile, simulator: SignalSimulator,
                   extractor: FeatureExtractor, setting: dict, context: dict,
                   env: dict = None):
    ctx = profile.context.encode(context)
    if simulator is not None:
        feats = extractor.extract_features(simulator.predict(setting))
    else:
        feats = {}
    pen = Constraint(profile.constraints.window).penalty(feats)
    return pen <= 1e-6, pen, feats
