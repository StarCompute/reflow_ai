<!--
版权 (c) 2026 蒲俊杰（Pu Junjie）. 保留所有权利。
许可见 https://github.com/StarCompute/reflow_ai/blob/main/proctune/LICENSE.md
个人使用（含个人商业）免费；企业/组织商业使用需获得授权。
-->

# proctune 使用文档：如何把系统接入你自己的业务

> **版权与许可**：版权归 **蒲俊杰（Pu Junjie）** 所有。
> 个人使用（含个人商业用途，如接单、个体户）**免费**；企业/组织用于商业目的需**付费授权**。
> 详见 [LICENSE.md](LICENSE.md)。

> 本框架解决的是一类**通用问题**，不只是回流焊：
> 只要你的业务满足「**可调工艺参数 → 过程 → 可测信号/对象属性 → 质量结果**」，
> 就能用同一套引擎做「给定对象，反求最优参数」的推荐，并带安全约束拦截。
>
> 本文档重点回答：**我有一个类似但不同的业务，怎么调用这套代码？**

---

## 一、先判断你的业务是否适配

下面三种形态都能直接用，框架自动选对应推荐模式：

| 业务形态 | 例子 | 信号 | 旋钮 | 质量 | 推荐模式 |
|---------|------|------|------|------|---------|
| A. 信号介导 | 回流焊、波峰焊、热处理 | 有（炉温曲线等） | 连续 | 缺陷分类 | **贝叶斯优化 BO** |
| B. 无信号·直连 | 注塑、涂装、3D 打印 | 无 | 连续 | 缺陷分类 | **贝叶斯优化 BO** |
| C. 无信号·离散 | 黄铜高炉（选哪座炉） | 无 | 离散 | 评分回归 | **离散枚举择优** |

> 形态 B 与 A 的差别只是「没有中间信号」：质量模型输入直接由旋钮+上下文拼成，
> 跳过代理模型。引擎对两者透明。

---

## 二、接入只需要 4 步（核心：写一份画像 + 几个适配器）

以「新增一个业务」为例，你**不需要改 `core/` 任何代码**。

### 第 1 步：写业务画像 `BusinessProfile`
用声明式数据类描述 5 个抽象（旋钮 / 信号 / 特征 / 上下文 / 质量 / 约束）：

```python
from proctune.core.abstractions import (BusinessProfile, KnobSpace, KnobParam,
                                        SignalSpec, FeatureSpec, ContextSpec, ContextField,
                                        QualitySpec, ConstraintSpec)

MY_PROFILE = BusinessProfile(
    name="my_business",
    description="一句话说明你的业务",
    # 1) 旋钮（可调工艺参数）
    knob_space=KnobSpace(params=[
        KnobParam("temp_a", 180.0, 260.0, "℃"),
        KnobParam("pressure", 40.0, 140.0, "MPa"),
        # 离散旋钮这样写：
        # KnobParam("furnace", 0, 1, "", kind="categorical", categories=["new","old"]),
    ]),
    # 2) 信号（没有就填 n_channels=0）
    signal=SignalSpec(n_channels=6, n_points=180, duration=240.0, unit="℃"),
    # 3) 信号特征名（用于约束校验；无信号填空）
    features=FeatureSpec(names=["peak_temp", "tal", "ramp_up"]),
    # 4) 对象上下文（被加工对象的属性；categorical 需给编码表）
    context=ContextSpec(fields=[
        ContextField("thickness_mm", "numeric", 1.6),
        ContextField("material", "categorical", 0, {"PP": 1.0, "ABS": 2.0}),
    ]),
    # 5) 质量（defect=多标签分类 / score=连续评分回归）
    quality=QualitySpec(kind="defect", defect_labels=["缺陷A", "缺陷B"]),
    # 6) 安全硬约束（键必须是上面的特征名；越界 100% 拦截）
    constraints=ConstraintSpec(window={"peak_temp": (235.0, 250.0)}),
)
```

### 第 2 步：实现适配器（实现 `core/interfaces` 里的接口）

**有信号业务（形态 A）需要实现 3 个**：`SignalSimulator`、`FeatureExtractor`、`SyntheticGenerator`，
再加一个 `SearchStrategy`（给 BO 提供边界与先验种子）。

```python
from proctune.core.interfaces import SignalSimulator, FeatureExtractor, SyntheticGenerator, SearchStrategy
import numpy as np

class MySimulator(SignalSimulator):
    """物理基线：旋钮 → 过程信号。生产可换成读真实时序库。"""
    def predict(self, setting):
        # setting 是 {旋钮名: 值} 的字典
        return np.zeros((6, 180))   # 返回 (n_channels, n_points)

class MyExtractor(FeatureExtractor):
    def extract_features(self, signal):
        return {"peak_temp": float(signal.max()), "tal": 0.0, "ramp_up": 0.0}
    def build_quality_input(self, setting, context, feats, env=None):
        # 把「信号特征 + 上下文」拼成质量模型输入向量（顺序自定，但训练/推理一致）
        return np.array([feats["peak_temp"], context["thickness_mm"],
                         context["material"]], dtype=float)

class MySynthetic(SyntheticGenerator):
    """演示造数器；生产换成「读真实库」的适配器（同样返回 Record 列表）。"""
    def generate(self, n, seed=None):
        from proctune.core.abstractions import Record
        rng = np.random.RandomState(seed)
        recs = []
        for _ in range(n):
            setting = {"temp_a": rng.uniform(180,260), "pressure": rng.uniform(40,140)}
            signal = MySimulator().predict(setting)
            feats = MyExtractor().extract_features(signal)
            # 用你的「老师傅经验/真实规律」算标签
            label = "缺陷A" if feats["peak_temp"] < 240 else "无"
            recs.append(Record(setting=setting,
                               context={"thickness_mm": 1.6, "material": 1.0},
                               signal=signal, quality_label=label))
        return recs

class MySearch(SearchStrategy):
    def bounds(self):
        return {"temp_a": (180.0, 260.0), "pressure": (40.0, 140.0)}
    def initial_candidates(self, context):
        # 返回若干「已知合法/先验」设定，加速 BO 收敛
        return [{"temp_a": 240.0, "pressure": 90.0}]
```

**无信号业务（形态 B/C）更简单**：`SignalSimulator` 传 `None`，
`FeatureExtractor.extract_features` 返回 `{}`，`build_quality_input` 直接用旋钮+上下文拼向量即可。
离散业务（形态 C）连 `SearchStrategy` 都不用给。

### 第 3 步：组装引擎并训练

```python
from proctune.core.pipeline import ProcessTuningEngine

engine = ProcessTuningEngine(
    profile=MY_PROFILE,
    simulator=MySimulator(),        # 无信号业务传 None
    extractor=MyExtractor(),
    synth=MySynthetic(),
    search=MySearch(),              # 离散业务传 None
    model_dir="./models_my",
)
engine.train(2000)                  # 造数 → 训代理 + 质量 + 推荐，并落盘
```

### 第 4 步：推荐 + 安全网关（线上调用）

```python
# 给定一块新对象的上下文，反求最优参数
result = engine.recommend({"thickness_mm": 2.0, "material": "PP"}, top_k=1)
top = result[0]
print(top["setting"], top["score"], top["quality"])   # 设定 / 预测良率 / 缺陷概率

# 下发前安全校验：越界 100% 拦截
ok, penalty, feats = engine.dispatch_check(top["setting"], {"thickness_mm": 2.0, "material": "PP"})
if not ok:
    print("被安全网关拦截，罚分", penalty)
```

**就这四步。引擎、三模型、评估、安全网关全部复用，零改动。**

---

## 三、三个内置业务示例（可直接抄去改）

仓库已带三个完整可跑的业务，覆盖全部三种形态，运行：

```bash
python -m proctune.examples.run_all
```

| 示例 | 文件 | 形态 | 看点 |
|------|------|------|------|
| 回流焊 | `examples/run_reflow.py` | A 信号介导·连续BO·缺陷分类 | 8 温区+链速 → 炉温曲线 → 5 缺陷；个性化推荐，约束满足率 100% |
| 黄铜高炉 | `examples/run_blast_furnace.py` | C 无信号·离散枚举·评分回归 | 进料含量 → 选哪座炉；推荐随含量变化（低含量选新炉、高含量选旧炉） |
| 注塑 | `examples/run_injection_molding.py` | B 无信号·连续BO·缺陷分类 | 料温/压力/保压/模温 → 4 缺陷；质量直连旋钮，无中间信号 |

每个示例都是「组装 4 个适配器 + `engine.train` + `engine.recommend` + `evaluate`」的固定套路，
你照着把适配器换成自己的实现即可。

---

## 四、内置示例的关键结果（验证框架有效）

- **回流焊**：约束满足率 `1.0`、平均预测良率 `1.0`、代理模型 RMSE `≈0.5℃`；
  不同板型返回不同温区剖面（薄板峰值更低），体现「看板设温」。
- **黄铜高炉**：进料 30%→推荐新炉(72.3>59.6)，70%→推荐旧炉(87.6>69.5)，
  说明离散择优随对象上下文动态变化。
- **注塑**：三类制件预测良率均 `≈0.88`，缺陷概率低。

---

## 五、生产化替换指引（接口不变，只换实现）

| 模块 | 演示实现 | 生产替换 |
|------|---------|---------|
| 代理模型残差 | `MLPRegressor` | PyTorch PINN（物理约束网络） |
| 质量模型 | `RandomForest` 多标签 / 回归 | XGBoost（`MultiOutputClassifier`） |
| 推荐搜索 | 自实现 GP+UCB | `bayesian-optimization` 库 |
| 造数器 | `SyntheticGenerator`（随机） | 读真实库的同接口适配器 |
| 信号来源 | `SignalSimulator`（物理基线） | 真实时序库（InfluxDB 等）读取 |

所有替换都**只改 `adapters/` 里的实现，不动 `core/`**，保证引擎稳定。

---

## 六、常见问题

**Q：我的业务旋钮既有连续又有离散怎么办？**
A：框架会走「离散枚举」分支——枚举所有离散组合，连续旋钮取中点（或你在
`SearchStrategy.initial_candidates` 里给的先验）。纯连续才走 BO。

**Q：质量用连续评分（不是缺陷分类）？**
A：画像里 `quality=QualitySpec(kind="score", scale=100.0)`，引擎自动用
`ScoreQualityModel`（回归），goodness = 评分/scale。

**Q：没有过程信号（如注塑）能跑吗？**
A：能。`simulator=None`，质量模型输入由 `FeatureExtractor.build_quality_input`
直接用旋钮+上下文拼成，跳过代理模型。

**Q：约束窗口的键必须是信号特征吗？**
A：是的。约束用于「下发前用信号特征校验」（安全网关）。若你的业务无信号，
约束窗口留空 `{}`，越界拦截逻辑自动跳过（仍建议在生产里加对象级硬约束）。
