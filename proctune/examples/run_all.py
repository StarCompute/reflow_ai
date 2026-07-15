# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""一键跑三个业务示例，验证通用框架对「类似但不同」业务的覆盖。

运行：python -m proctune.examples.run_all
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .run_reflow import main as reflow_main
from .run_blast_furnace import main as bf_main
from .run_injection_molding import main as im_main


def main():
    print("=" * 70)
    print("  proctune · 通用工艺参数推荐框架 — 多业务示例")
    print("=" * 70)
    print("\n--- 业务 1 / 3：回流焊（信号介导 · 连续BO · 缺陷分类）---")
    reflow_main()
    print("\n--- 业务 2 / 3：黄铜高炉（无信号 · 离散枚举 · 评分回归）---")
    bf_main()
    print("\n--- 业务 3 / 3：注塑（无信号 · 连续BO · 缺陷分类）---")
    im_main()
    print("\n" + "=" * 70)
    print("  三个业务共用同一套 core 引擎，仅适配器不同。")
    print("=" * 70)


if __name__ == "__main__":
    main()
