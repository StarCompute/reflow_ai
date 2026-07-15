"""数据库层：SQLAlchemy ORM 定义 + 建表 + 会话。

对应 V2 §4 SQL DDL 与 14 数据字典。字段、类型、约束保持一致。
演示默认 SQLite；生产改 config.db_url 为 MySQL（需 pymysql）。
"""
from sqlalchemy import (create_engine, Column, String, Float, Integer, Boolean,
                        DateTime, Enum as SAEnum, JSON, ForeignKey, inspect, text)
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime
from config import CONFIG

Base = declarative_base()


class ReflowRun(Base):
    """炉次主表：每一次过炉的记录。"""
    __tablename__ = "reflow_run"
    run_id = Column(String(64), primary_key=True)           # R-YYMMDD-NNNN
    oven_id = Column(String(32), nullable=False)            # OVEN-xx
    product_id = Column(String(64), nullable=False)         # 关联 BOM
    lot_id = Column(String(64), nullable=False)            # 批次号（追溯关键）
    chain_speed = Column(Float, nullable=False)            # 链速 cm/min
    zone1_temp = Column(Float, default=0.0)
    zone2_temp = Column(Float, default=0.0)
    zone3_temp = Column(Float, default=0.0)
    zone4_temp = Column(Float, default=0.0)
    zone5_temp = Column(Float, default=0.0)
    zone6_temp = Column(Float, default=0.0)
    zone7_temp = Column(Float, default=0.0)
    zone8_temp = Column(Float, default=0.0)
    air_volume = Column(Float)                             # 风量 %
    oxygen_ppm = Column(Integer)                           # 氮气氛氧含量
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    operator_id = Column(String(32))                      # 操作人工号
    solder_paste = Column(String(64))                     # 本次过炉实际使用的锡膏型号
    ai_recommended = Column(Boolean, default=False)       # 是否 AI 推荐
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class QualityResult(Base):
    """质量结果表：板级追溯 + 缺陷标签。"""
    __tablename__ = "quality_result"
    id = Column(Integer, primary_key=True, autoincrement=True)
    lot_id = Column(String(64), nullable=False)
    sn = Column(String(128))
    aoi_result = Column(SAEnum("PASS", "FAIL"), nullable=False)
    xray_void_rate = Column(Float)                        # BGA 空洞率 %
    ict_result = Column(SAEnum("PASS", "FAIL"))
    defect_type = Column(SAEnum("无", "虚焊", "桥连", "立碑", "空洞", "锡珠", "其他"), nullable=False)
    defect_count = Column(Integer, default=0)
    repair_flag = Column(Boolean, nullable=False, default=False)   # 核心标签：是否返修
    inspect_time = Column(DateTime, nullable=False)
    feedback_ai_valid = Column(Boolean)                  # 人工反馈：AI 推荐对/错


class ProfileFeature(Base):
    """曲线特征表：模型输入（每次过炉计算一次）。"""
    __tablename__ = "profile_feature"
    run_id = Column(String(64), ForeignKey("reflow_run.run_id"), primary_key=True)
    peak_temp = Column(Float)
    tal = Column(Float)
    ramp_up = Column(Float)
    ramp_down = Column(Float)
    delta_t = Column(Float)
    soak_temp = Column(Float)
    time_above_183 = Column(Float)
    curve_duration = Column(Float)


class Bom(Base):
    """产品 BOM 维表：质量模型输入 9~14 维。"""
    __tablename__ = "bom"
    product_id = Column(String(64), primary_key=True)
    thickness_mm = Column(Float)
    copper_area_pct = Column(Float)
    bga_count = Column(Integer)
    max_bga_size_mm = Column(Float)
    component_density = Column(Float)


class SolderPaste(Base):
    """锡膏维表。"""
    __tablename__ = "solder_paste"
    paste_model = Column(String(64), primary_key=True)
    paste_lot = Column(String(64))
    thaw_min = Column(Float)
    expiry = Column(DateTime)


class EnvLog(Base):
    """环境日志：特征 15~16 维。"""
    __tablename__ = "env_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    env_temp = Column(Float)
    env_humidity = Column(Float)
    ts = Column(DateTime)


class AuditLog(Base):
    """审计日志：下发/拦截全程可追溯（对应 V2 §11）。"""
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String(32))          # DISPATCH / REJECT / OVERRIDE
    request_id = Column(String(64))
    oven_id = Column(String(32))
    old_params = Column(JSON)
    new_params = Column(JSON)
    operator_id = Column(String(32))
    result = Column(String(16))           # SUCCESS / BLOCKED
    ip_address = Column(String(45))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


engine = create_engine(CONFIG.db_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, future=True)


def migrate_db():
    """兼容迁移：补加 solder_paste 列；温区数漂移时重建炉次相关表。"""
    insp = inspect(engine)
    if "reflow_run" not in insp.get_table_names():
        return  # init_db 尚未建表，跳过迁移
    cols = [c["name"] for c in insp.get_columns("reflow_run")]
    if "solder_paste" not in cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE reflow_run ADD COLUMN solder_paste VARCHAR(64)"))
        cols.append("solder_paste")

    # 温区数不一致（如 10→8）→ 重建炉次相关表，避免 schema 漂移导致读取越界
    n_zone = sum(1 for i in range(1, 21) if f"zone{i}_temp" in cols)
    if n_zone != CONFIG.zone_num:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS profile_feature"))
            conn.execute(text("DROP TABLE IF EXISTS quality_result"))
            conn.execute(text("DROP TABLE IF EXISTS reflow_run"))
        Base.metadata.create_all(engine)
        print(f"[migrate] 温区数 {n_zone}→{CONFIG.zone_num}，已重建炉次相关表")


def init_db():
    """建表（生产对应 V2 §4 DDL）。"""
    Base.metadata.create_all(engine)


def get_session():
    return SessionLocal()
