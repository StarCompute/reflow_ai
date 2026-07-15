# Copyright (c) 2026 蒲俊杰（Pu Junjie）. All rights reserved.
# 许可见 https://github.com/StarCompute/reflow_ai/blob/main/proctune/LICENSE.md
# 个人使用（含个人商业）免费；企业/组织商业使用需获得授权。

"""统一日志：演示输出到控制台（rich 彩色）；生产可接文件 / ELK / Loki。

对应 V2 §11 可观测性。用法：
    from utils.logger import get_logger
    log = get_logger("train")
    log.info("训练完成")
"""
import logging
from rich.logging import RichHandler


def get_logger(name: str = "reflow_ai"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_path=False,
            log_time_format="[%X]",
        )
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
