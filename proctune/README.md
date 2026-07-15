<!--
版权 (c) 2026 蒲俊杰（Pu Junjie）. 保留所有权利。
许可见 https://github.com/StarCompute/reflow_ai/blob/main/proctune/LICENSE.md
个人使用（含个人商业）免费；企业/组织商业使用需获得授权。
-->

# proctune · 通用「受控工艺参数推荐」框架

> **版权与许可**：版权归 **蒲俊杰（Pu Junjie）** 所有。
> 个人使用（含个人商业用途，如接单、个体户）**免费**；企业/组织用于商业目的需**付费授权**。
> 详见 [LICENSE.md](LICENSE.md)。

把"回流焊工艺优化"这类问题抽象成一套**与业务无关**的引擎。新业务只需提供
**一份业务画像 `BusinessProfile` + 几个适配器**，即可复用训练 / 推荐 / 安全网关 / 评估全流程。

> 对现场工程师 / 不懂 AI 的人：这就是一个把厂里过去的生产记录喂进去、对新活儿自动给机器参数的工具，详见 [docs/EASY.md](docs/EASY.md)。

```
可调工艺参数(旋钮) ─► 物理过程 ─► 可测信号 ─► 提取特征 ─► 质量结果(缺陷/良率)
   Knobs              Process      Signal        Features        Quality
        ▲                                                          │
        └──────────── 推荐引擎：给定「对象上下文 + 安全约束」反求最优 Knobs ──┘
```

## 目录
```
proctune/
├── core/                  # 通用引擎（不含任何业务字面量）
│   ├── abstractions.py    # BusinessProfile 等 5 个抽象
│   ├── interfaces.py      # 适配器接口（SignalSimulator / FeatureExtractor / SyntheticGenerator / SearchStrategy）
│   ├── models/            # 代理模型 / 质量模型 / 推荐引擎（BO + 离散枚举）
│   ├── pipeline.py        # 训练 / 推荐 / 安全网关 编排
│   ├── evaluate.py        # 通用评估
│   └── dispatch.py        # 安全网关（越界 100% 拦截）
├── adapters/              # 各业务实现（领域逻辑只在这里）
│   ├── reflow/            # 回流焊：信号介导 + 连续BO + 缺陷分类
│   ├── blast_furnace/     # 黄铜高炉：无信号 + 离散枚举 + 评分回归
│   └── injection_molding/ # 注塑：无信号 + 连续BO + 缺陷分类
└── examples/              # 三个业务的调用示例（可直接抄去改）
    ├── run_reflow.py
    ├── run_blast_furnace.py
    ├── run_injection_molding.py
    └── run_all.py
```

## 快速开始
```bash
cd d:/AI_REFLOW
python -m proctune.examples.run_all      # 跑三个业务，验证框架可用
```

## 两种用法

| 你是谁 | 怎么用 | 入口 |
|--------|--------|------|
| **现场工程师 / 不懂 AI** | 准备两张表（历史样本 + 新任务），三行代码拿推荐 | [`proctune.easy.EasyTuner`](docs/EASY.md) |
| **算法 / 开发工程师** | 写一份 `BusinessProfile` + 几个适配器，复用全部引擎 | [docs/USAGE.md](docs/USAGE.md) |

## 文档
- [proctune/demo/demo.php](demo/demo.php) —— **网页演示**：浏览器里选两张表、点按钮拿推荐（支持 CSV / MySQL）
- [docs/EASY.md](docs/EASY.md) —— **普通人入口**：只靠两张表（历史样本 + 新任务）让系统推荐参数，无需懂 AI
- [docs/USAGE.md](docs/USAGE.md) —— 架构说明 + **如何把系统接入你自己的业务**（含完整最小示例、四步接入法）
