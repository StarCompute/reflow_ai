# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""Web 桥接脚本：供 demo.php（或任何外部程序）调用，完成「训练 + 推荐」。

数据源可在 CSV 与 MySQL 之间切换：
  - 历史样本（训练用）：--history-source csv|mysql
  - 新任务（待推荐）：  --input-source  csv|mysql

示例（CSV 模式，demo.php 默认走这条）：
  python -m proctune.easy.web_bridge \
      --history-source csv --history-csv history_datas.csv \
      --input-source csv --input-csv input.csv \
      --knob-cols 料筒温度,注射压力,保压时间,模具温度 \
      --context-cols 材料,壁厚 --quality-col 质量 \
      --output recommend_result.csv --top-k 1

示例（MySQL 模式）：
  python -m proctune.easy.web_bridge \
      --history-source mysql --history-table t_history \
      --input-source mysql --input-table t_input \
      --mysql-host 127.0.0.1 --mysql-user root --mysql-pass 123456 --mysql-db factory \
      --knob-cols barrel_temp,inject_pressure,holding_time,mold_temp \
      --context-cols material,wall --quality-col quality \
      --output recommend_result.csv

标准输出会打印一行 JSON（含 summary / rows / output），便于调用方解析。
"""
import argparse
import csv
import json
import sys

from proctune.easy import EasyTuner


def _split(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def _load_rows(args):
    """按 source 返回 (history_rows_or_path, input_rows_or_path)。"""
    # 历史样本
    if args.history_source == "mysql":
        from proctune.easy.db import read_mysql_table
        conn = dict(host=args.mysql_host, user=args.mysql_user,
                    password=args.mysql_pass, database=args.mysql_db,
                    port=args.mysql_port)
        history = read_mysql_table(table=args.history_table, **conn)
    else:
        history = args.history_csv

    # 新任务
    if args.input_source == "mysql":
        from proctune.easy.db import read_mysql_table
        conn = dict(host=args.mysql_host, user=args.mysql_user,
                    password=args.mysql_pass, database=args.mysql_db,
                    port=args.mysql_port)
        newdata = read_mysql_table(table=args.input_table, **conn)
    else:
        newdata = args.input_csv

    return history, newdata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-source", choices=["csv", "mysql"], default="csv")
    ap.add_argument("--history-csv", default="history_datas.csv")
    ap.add_argument("--history-table", default="")
    ap.add_argument("--input-source", choices=["csv", "mysql"], default="csv")
    ap.add_argument("--input-csv", default="input.csv")
    ap.add_argument("--input-table", default="")
    ap.add_argument("--mysql-host", default="127.0.0.1")
    ap.add_argument("--mysql-user", default="root")
    ap.add_argument("--mysql-pass", default="")
    ap.add_argument("--mysql-db", default="")
    ap.add_argument("--mysql-port", type=int, default=3306)
    ap.add_argument("--knob-cols", required=True)
    ap.add_argument("--context-cols", required=True)
    ap.add_argument("--quality-col", required=True)
    ap.add_argument("--output", default="recommend_result.csv")
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--n-trees", type=int, default=200)
    args = ap.parse_args()

    history, newdata = _load_rows(args)
    if not history:
        print(json.dumps({"ok": False, "error": "历史样本为空"}))
        return 1

    tuner = EasyTuner(history, _split(args.knob_cols), _split(args.context_cols),
                      args.quality_col, n_trees=args.n_trees)
    tuner.train()
    out = tuner.recommend_to_csv(newdata, args.output, top_k=args.top_k)

    # 读回结果行数
    n = 0
    with open(out, encoding="utf-8-sig") as f:
        n = sum(1 for _ in csv.reader(f)) - 1

    result = {
        "ok": True,
        "summary": tuner.summary(),
        "rows": n,
        "output": out,
        "columns": _split(args.knob_cols) + _split(args.context_cols),
    }
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
