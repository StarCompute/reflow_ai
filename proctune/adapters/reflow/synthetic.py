# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""回流焊演示造数器：基于 PCB 属性驱动（老师傅「看板设温」）。

实现 core.interfaces.SyntheticGenerator：generate(n) -> List[Record]。
生产环境把本类换成「读真实库」的适配器（同样返回 Record 列表）即可。
"""
import random
import datetime
import numpy as np
from proctune.core.abstractions import Record
from proctune.core.interfaces import SyntheticGenerator
from .profile import SIM_PARAMS, SOLDER_MAP
from .thermal_solver import ThermalSolver
from .curve_features import CurveFeatureExtractor


class ReflowSynthetic(SyntheticGenerator):
    def __init__(self):
        self.solver = ThermalSolver()
        self.feat = CurveFeatureExtractor()

    @staticmethod
    def _bom_dict(pcb):
        return {"thickness_mm": pcb[1], "copper_area_pct": pcb[2],
                "bga_count": pcb[3], "max_bga_size_mm": pcb[4],
                "component_density": pcb[5]}

    @staticmethod
    def _master_optimal(bom):
        t, c, b = bom["thickness_mm"], bom["copper_area_pct"], bom["bga_count"]
        P = SIM_PARAMS
        ideal_peak = P["peak_base"] + P["peak_k_t"] * t + P["peak_k_c"] * c + P["peak_k_b"] * b
        ideal_speed = P["speed_base"] + P["speed_k_t"] * t + P["speed_k_c"] * c + P["speed_k_b"] * b
        ideal_soak = P["soak_base"] + P["soak_k_t"] * t
        return ideal_peak, ideal_speed, ideal_soak

    @staticmethod
    def _true_defect(bom, feat):
        t, c, b = bom["thickness_mm"], bom["copper_area_pct"], bom["bga_count"]
        P = SIM_PARAMS
        ideal_curve_peak = (P["peak_base"] + P["peak_k_t"] * t + P["peak_k_c"] * c
                            + P["peak_k_b"] * b - P["curve_peak_offset"])
        peak = feat["peak_temp"]
        if peak < ideal_curve_peak - P["pass_band_half"]:
            return "虚焊"
        if peak > ideal_curve_peak + P["pass_band_half"]:
            return "桥连"
        if feat["delta_t"] > P["tombstone_dt"]:
            return "立碑"
        if feat["time_above_183"] > P["void_tal"]:
            return "空洞"
        if feat["ramp_up"] > P["solderball_ramp"]:
            return "锡珠"
        return "无"

    def _gen_bom_pool(self, n_extra=40, seed=20260714):
        rng = random.Random(seed)
        solders = list(SOLDER_MAP.keys())
        pool = list(SIM_PARAMS["pcb_types"])
        for i in range(n_extra):
            pool.append((
                f"PCB-SYN{i:03d}",
                round(rng.uniform(0.8, 2.4), 1),
                round(rng.uniform(15.0, 60.0), 1),
                rng.randint(1, 6),
                round(rng.uniform(12.0, 35.0), 1),
                round(rng.uniform(6.0, 16.0), 1),
                rng.choice(solders),
            ))
        return pool

    def generate(self, n: int, seed: int = None) -> list:
        rng = random.Random(seed)
        np.random.seed(seed if seed is not None else 42)
        pool = self._gen_bom_pool()
        P = SIM_PARAMS
        records = []
        for i in range(n):
            pcb = rng.choice(pool)
            pid, solder = pcb[0], pcb[6]
            bom = self._bom_dict(pcb)
            ideal_peak, ideal_speed, ideal_soak = self._master_optimal(bom)
            if rng.random() < P["good_ratio"]:
                tgt_peak = ideal_peak + np.random.normal(0, P["peak_noise_std"])
                speed = ideal_speed + np.random.normal(0, P["speed_noise_std"])
            else:
                tgt_peak = ideal_peak + rng.choice(P["dev_peak_offsets"])
                speed = ideal_speed + rng.choice(P["dev_speed_offsets"])
            soak = ideal_soak + np.random.normal(0, P["soak_noise_std"])
            base = [soak - 12, soak, soak + 8, soak + 16, 212.0, 225.0,
                    tgt_peak - 10.0, tgt_peak]
            zones = [round(float(np.clip(v + np.random.normal(0, P["zone_noise_std"]), 150, 280)), 1)
                     for v in base]
            speed = round(float(np.clip(speed, 40, 120)), 1)
            setting = {f"zone{i}_temp": zones[i - 1] for i in range(1, 9)}
            setting["chain_speed"] = speed

            signal = self.solver.predict(setting) + np.random.normal(0, 0.5, (6, 180))
            feat = self.feat.extract_features(signal)
            label = self._true_defect(bom, feat)
            context = {
                "thickness_mm": bom["thickness_mm"],
                "copper_area_pct": bom["copper_area_pct"],
                "bga_count": bom["bga_count"],
                "max_bga_size_mm": bom["max_bga_size_mm"],
                "component_density": bom["component_density"],
                "solder_paste": float(SOLDER_MAP.get(solder, 0)),
                "env_temp": 25.0, "env_humidity": 50.0,
            }
            records.append(Record(setting=setting, context=context,
                                  signal=signal, quality_label=label))
        return records
