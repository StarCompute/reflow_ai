# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""适配器接口：业务侧实现这些接口，通用引擎只依赖接口，不依赖具体业务。

一个业务要接入，至少实现：
  - SignalSimulator   （有信号业务；无信号可传 None）
  - FeatureExtractor  （把 信号+上下文 拼成质量模型输入向量）
  - SyntheticGenerator（演示/冷启动用的造数器；生产可换成真实数据适配器）
  - SearchStrategy    （连续旋钮的 BO 搜索策略；离散业务不需要）
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np

from .abstractions import Record


class SignalSimulator(ABC):
    """物理基线：旋钮设定 → 过程信号数组 (n_channels, n_points)。"""

    @abstractmethod
    def predict(self, setting: Dict[str, float]) -> np.ndarray:
        ...


class FeatureExtractor(ABC):
    """信号 → 特征字典；以及 (旋钮, 上下文, 特征) → 质量模型输入向量。"""

    @abstractmethod
    def extract_features(self, signal: np.ndarray) -> Dict[str, float]:
        ...

    @abstractmethod
    def build_quality_input(self, setting: Dict[str, float],
                            context: Dict[str, float],
                            feats: Dict[str, float],
                            env: Dict[str, float] = None) -> np.ndarray:
        ...


class SyntheticGenerator(ABC):
    """生成训练样本（演示用；生产换成读真实库的实现即可）。"""

    @abstractmethod
    def generate(self, n: int, seed: int = None) -> List[Record]:
        ...


class SearchStrategy(ABC):
    """连续旋钮的贝叶斯优化搜索策略（离散业务用不到）。"""

    @abstractmethod
    def bounds(self) -> Dict[str, Tuple[float, float]]:
        """连续旋钮的 (lo, hi)。"""

    @abstractmethod
    def initial_candidates(self, context: Dict[str, float]) -> List[Dict[str, float]]:
        """返回若干「已知合法/先验」旋钮设定，作为 BO 初始候选，加速收敛。"""
