# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""MySQL 读取辅助：把一张表读成 list[dict]，供 EasyTuner 使用。

依赖（任选其一即可，无需两者都装）：
    pip install pymysql
    pip install mysql-connector-python
"""
from typing import List, Dict, Optional


def _get_mysql_module():
    try:
        import pymysql
        return pymysql, "pymysql"
    except ImportError:
        pass
    try:
        import mysql.connector
        return mysql.connector, "mysql.connector"
    except ImportError:
        raise ImportError(
            "未安装 MySQL 驱动，请任选其一安装：\n"
            "  pip install pymysql\n"
            "  pip install mysql-connector-python"
        )


def read_mysql_table(host: str, user: str, password: str, database: str,
                     table: str, port: int = 3306,
                     charset: str = "utf8mb4") -> List[Dict]:
    """读取整张表为 list[dict]（每行一个 dict，键为列名）。

    表里的列名应与你传给 EasyTuner 的 knob_cols / context_cols / quality_col 对应。
    """
    mod, name = _get_mysql_module()
    if name == "pymysql":
        import pymysql.cursors
        conn = mod.connect(host=host, user=user, password=password,
                           database=database, port=port, charset=charset,
                           cursorclass=pymysql.cursors.DictCursor)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM `%s`" % table)
                return list(cur.fetchall())
        finally:
            conn.close()
    else:
        conn = mod.connect(host=host, user=user, password=password,
                           database=database, port=port)
        try:
            with conn.cursor(dictionary=True) as cur:
                cur.execute("SELECT * FROM `%s`" % table)
                return list(cur.fetchall())
        finally:
            conn.close()
