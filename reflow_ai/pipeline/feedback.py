# Copyright (c) 2026 蒲俊杰（Pu Junjie）. All rights reserved.
# 许可见 https://github.com/StarCompute/reflow_ai/blob/main/proctune/LICENSE.md
# 个人使用（含个人商业）免费；企业/组织商业使用需获得授权。

"""反馈回灌：质量结果回流 → 入重训样本池 → 触发重训。

对应 V2 §10 数据飞轮 / 09-feedback.html / 清单 2.5。
"""
import datetime
from data.db import SessionLocal, QualityResult


def record_feedback(run_id, lot_id, accepted=True,
                    actual_defect_type="无", repair=False, operator_id=None):
    sess = SessionLocal()
    try:
        qr = QualityResult(
            lot_id=lot_id,
            aoi_result="PASS" if not repair else "FAIL",
            defect_type=actual_defect_type,
            repair_flag=repair,
            inspect_time=datetime.datetime.now(),
            feedback_ai_valid=accepted,
        )
        sess.add(qr)
        sess.commit()
    finally:
        sess.close()
    # 触发重训练（生产：入 MQ 队列；演示：打印）
    print(f"[OK] 反馈已记录 run={run_id} accepted={accepted} "
          f"defect={actual_defect_type}，已加入重训练样本池。")
