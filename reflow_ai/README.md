# 回流焊 AI 工艺优化系统 · 代码实现（V2 配套）

> 对应文档：`docs/tech-design-v2.html`（技术方案 V2）、`docs/data-algo-checklist.html`（采集&算法清单）。
> 前提：**假设数据已经存在**（炉次设定、实测曲线、质量结果、BOM 等已落入数据库）。
> 本代码给出从「已有数据」出发的**端到端可运行实现**：质量校验 → 特征工程 → 三模型训练 → 评估验收 → 推荐 → 人确认下发 → 反馈回灌。

---

## 一、目录结构（结构清楚）

```
reflow_ai/
├── README.md                 # 本文件：结构 / 流程 / 运行说明
├── requirements.txt          # 依赖（轻量，可纯 sklearn 跑通演示）
├── config.py                 # 全局配置：工艺窗口硬约束、缺陷标签、路径
├── data/
│   ├── db.py                # SQLAlchemy 建表 + 会话（对应 V2 §4 DDL）
│   └── validate.py          # 数据质量红线校验（对应清单 1.3）
├── features/
│   └── extractor.py         # 曲线→8 维特征；拼接 16 维模型输入
├── models/
│   ├── surrogate.py          # 模型1 工艺代理模型（物理基线 + NN 残差 / PINN 思路）
│   ├── quality.py            # 模型2 质量预测（多标签，生产换 XGBoost）
│   └── recommender.py       # 模型3 参数推荐引擎（贝叶斯优化 + 约束注入）
├── training/
│   ├── train.py             # 训练 pipeline：读库→特征→训三模型→落盘
│   └── evaluate.py          # 验收指标评估（对应清单 2.6 / V2 §12）
├── api/
│   └── main.py              # FastAPI：recommend / dispatch(安全网关) / feedback / health
├── rag/
│   └── kb.py                # 知识库问答（本地 RAG，大模型可接 Qwen-7B）
├── pipeline/
│   └── feedback.py          # 反馈回灌：质量结果入样本池 → 触发重训
├── utils/
│   └── logger.py            # 统一日志（对应 V2 §11 可观测性）
├── tests/
│   └── test_basic.py       # 基础冒烟测试（pytest）
└── demo.py                  # 端到端演示（无数据则造『工艺相关』模拟数据）
```

---

## 二、端到端流程（流程清楚）

```
[已有数据]  reflow_run / profile_signal(Influx) / quality_result / bom / solder_paste
    │
    ▼  ① 质量红线校验  data/validate.py  （清单 1.3，不通过则阻断训练）
    │
    ▼  ② 特征工程  features/extractor.py
    └─ 6 通道曲线 → 8 维曲线特征 → 拼接 BOM/锡膏/环境 = 16 维向量
    │
    ▼  ③ 训练三模型  training/train.py
    ├─ 代理模型：  (设定值) ──► 预测 6 通道曲线
    ├─ 质量模型：  (16 维)  ──► 5 类缺陷概率 + 良率
    └─ 推荐引擎：  (BOM+锡膏+窗口) ──► 贝叶斯优化搜最优设定
    │
    ▼  ④ 模型评估  training/evaluate.py  （对照清单 2.6 验收线）
    │
    ▼  ⑤ 生成推荐  api/main.py  POST /api/v1/recommend
    └─ 返回 top-k 设定 + 预测良率 + 5 类缺陷概率
    │
    ▼  ⑥ 人确认下发  api/main.py  POST /api/v1/dispatch（安全网关）
    └─ 代理模型预测曲线 → 校验工艺窗口 → 越界 100% 拦截 → 写 audit_log → 写 PLC
    │
    ▼  ⑦ 质量反馈   pipeline/feedback.py  POST /api/v1/feedback
    └─ AOI/X-Ray/ICT 结果回流 quality_result → 入重训样本池
    │
    ▼  ⑧ 数据飞轮   下一轮 training 吸收新样本，模型更准（见 docs/10-improve.html）
```

---

## 三、如何运行（说明清楚）

```bash
cd reflow_ai
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

# 1) 建表（MySQL；演示可用 SQLite 改 config.db_url）
python -c "from data.db import init_db; init_db()"

# 2) 端到端演示（自动造『工艺相关』数据 → 校验 → 训练 → 评估 → 推荐 → 反馈）
python demo.py

# 3) 单元测试
pip install pytest
pytest tests/ -q

# 4) 启动 API 服务（另开终端）
uvicorn api.main:app --reload --port 8000
#   交互式文档： http://127.0.0.1:8000/docs
```

> 演示用轻量依赖即可跑通（numpy / scikit-learn / sqlalchemy / fastapi）。
> 生产级替换见下表，接口不变，仅换实现：

| 模块 | 演示实现 | 生产替换（见 V2） |
|------|---------|------------------|
| 代理模型残差 | `MLPRegressor` | PyTorch **PINN**（物理约束网络）|
| 质量模型 | `RandomForest` 多标签 | **XGBoost** `MultiOutputClassifier`（AUC≥0.85）|
| 推荐搜索 | 自实现 GP+UCB 贝叶斯优化 | `bayesian-optimization` 库 / 同思路 |
| 知识库大模型 | 规则拼接占位 | 本地 **Qwen-7B** + **BGE-M3** 向量 |
| 曲线存储 | 演示用物理基线生成 | 真实 **InfluxDB** `profile_signal` |
| 关系库 | 可 SQLite | **MySQL**（DDL 见 V2 §4）|

---

## 三·补 `demo_window` 与"自洽演示"设计

- `config.py` 里**两套工艺窗口是刻意分开的**：
  - `process_window`：来自 V2 §7 的**真实生产规范**（peak∈[235,250] 等），API 安全网关 `/dispatch` 用它做 100% 越界拦截。
  - `demo_window`：贴合本仓库**玩具物理代理模型**实际能产出的特征分布。演示造数（缺陷标签由"是否违反 `demo_window`"推导）+ 推荐引擎约束都用它。
- 这样设计的妙处（自洽）：质量模型学到的就是"曲线特征→是否越界"的边界；推荐引擎只要把缺陷概率压到最低，设定就自然落在窗口内 → 约束满足率 100%、AUC 可算。生产接入真实数据/真实物理后，训练与推荐统一改用 `process_window`。

## 四、质量红线校验（清单 1.3）

`data/validate.py` 对全库数据执行 8 条红线，返回结构化报告；不通过则 `demo` 警告并建议先治理数据。

| 红线 | 含义 |
|------|------|
| R1 | run_id 唯一且非空 |
| R2 | 每通道曲线采样点 ≥ 60 |
| R3 | 实测峰值 - 设定峰值偏差 ≤ 30℃（否则报警）|
| R4 | 追溯率 100%：每个 run 均有 quality_result |
| R5 | 时间合法：end_time > start_time |
| R6 | lot_id 非空（板级追溯）|
| R7 | defect_type 在枚举内 |
| R8 | 必填非空：oven_id / product_id / chain_speed |

用法：`from data.validate import validate_dataset, print_report`

---

## 五、模型评估与验收指标（清单 2.6）

`training/evaluate.py` 在训练后输出量化指标，对照验收线：

- **质量模型**：各缺陷 ROC-AUC、macro-AUC（验收线 AUC ≥ 0.85），内部 train/test 拆分不污染主模型；
- **代理模型**：曲线预测 RMSE（℃）；
- **推荐引擎**：工艺窗口约束满足率（验收线 100% 拦截越界）、平均预测良率。

`demo.py` 末尾会打印上述指标，便于现场核对是否达标。

---

## 六、与文档对应关系

- `config.py` 的 `process_window` ↔ V2 §7 硬约束（peak∈[235,250] 等）
- `data/db.py` ↔ V2 §4 SQL DDL / `14-datadict.html`
- `data/validate.py` ↔ 清单 1.3 质量红线
- `features/extractor.py` ↔ V2 §5 特征工程 / 清单 1.2
- `models/*` ↔ V2 §6~§7 / 清单 2.2~2.4
- `training/evaluate.py` ↔ 清单 2.6 验收指标 / V2 §12
- `api/main.py` ↔ V2 §9 API Schema / §11 安全网关
- `pipeline/feedback.py` ↔ V2 §10 数据飞轮 / `09-feedback.html`
