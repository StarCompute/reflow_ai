#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
黄铜选矿高炉智能推荐系统（无预设规律版）
-------------------------------------------
- 历史数据随机生成，两个高炉无固定优劣关系
- 使用线性回归（或多项式回归）建模出料品质
- 新批次：分别预测两个高炉品质，选择预测值更高的
- 对比“平均分配”与“优化分配”总产出
"""

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from typing import Tuple, List, Dict

# ---------- 解决控制台中文乱码 ----------
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ---------- 设置 Matplotlib 中文字体 ----------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 数据模拟（无预设规律） ----------
def generate_synthetic_data(n_samples: int = 120, seed: int = 42, degree: int = 1) -> Dict[str, np.ndarray]:
    """
    生成两个高炉的历史数据，使用随机多项式关系 + 噪声。
    两个高炉的系数随机生成，无固定优劣模式。
    """
    random.seed(seed)
    np.random.seed(seed)

    X_new = np.random.uniform(20, 80, n_samples)
    X_old = np.random.uniform(20, 80, n_samples)

    # 随机生成多项式系数（最高 degree 次）
    # 确保系数较小，使 y 在 0~100 范围内
    coef_new = np.random.uniform(-0.2, 0.3, degree+1)  # 常数项 ~ [-0.2,0.3] 但我们会调整
    coef_new[0] = np.random.uniform(5, 20)   # 截距
    coef_new[1:] = np.random.uniform(-0.5, 0.8, degree)  # 一次项及以上

    coef_old = np.random.uniform(-0.2, 0.3, degree+1)
    coef_old[0] = np.random.uniform(5, 20)
    coef_old[1:] = np.random.uniform(-0.5, 0.8, degree)

    # 构造多项式值
    def poly_val(x, coef):
        val = 0
        for i, c in enumerate(coef):
            val += c * (x ** i)
        return val

    y_new = np.array([poly_val(x, coef_new) + np.random.normal(0, 3) for x in X_new])
    y_old = np.array([poly_val(x, coef_old) + np.random.normal(0, 3) for x in X_old])

    # 截断到 [0, 100]
    y_new = np.clip(y_new, 0, 100)
    y_old = np.clip(y_old, 0, 100)

    return {"new": {"X": X_new, "y": y_new}, "old": {"X": X_old, "y": y_old}}

# ---------- 回归模型（支持线性/二次） ----------
def build_regressor(degree: int = 1):
    """返回一个多项式回归管道"""
    if degree == 1:
        return LinearRegression()
    else:
        return Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('linear', LinearRegression())
        ])

def train_model(X: np.ndarray, y: np.ndarray, degree: int = 1):
    model = build_regressor(degree)
    model.fit(X.reshape(-1, 1), y)
    return model

def predict_model(model, X: float) -> float:
    return model.predict([[X]])[0]

# ---------- 绘图 ----------
def plot_models(data: Dict, models: Dict, degree: int):
    plt.figure(figsize=(10, 6))
    colors = {"new": "blue", "old": "red"}
    labels = {"new": "新高炉", "old": "旧高炉"}
    for key in ["new", "old"]:
        X = data[key]["X"]
        y = data[key]["y"]
        model = models[key]
        plt.scatter(X, y, alpha=0.6, color=colors[key], label=f"{labels[key]} 历史数据")
        # 绘制拟合曲线
        x_line = np.linspace(min(X), max(X), 100)
        y_line = [predict_model(model, x) for x in x_line]
        plt.plot(x_line, y_line, color=colors[key], linestyle='--', label=f"{labels[key]} 拟合曲线")
    plt.xlabel("进料含量 (%)")
    plt.ylabel("出料品质 (%)")
    plt.title(f"高炉历史数据及拟合模型 (degree={degree})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ---------- 主程序 ----------
def main():
    print("=" * 70)
    print("     黄铜选矿高炉智能推荐系统（无预设规律版）")
    print("=" * 70)

    # 参数选择
    degree = 2  # 可改为1（线性）或更高，请根据数据复杂度调整
    print(f"\n使用多项式回归 (degree={degree}) 建模历史数据。")

    print("\n正在模拟历史数据并训练模型...")
    data = generate_synthetic_data(n_samples=150, seed=42, degree=degree)
    models = {}
    for key in ["new", "old"]:
        X, y = data[key]["X"], data[key]["y"]
        model = train_model(X, y, degree)
        models[key] = model

    hist_min = min(np.min(data["new"]["X"]), np.min(data["old"]["X"]))
    hist_max = max(np.max(data["new"]["X"]), np.max(data["old"]["X"]))
    print(f"  历史进料含量范围: [{hist_min:.1f}, {hist_max:.1f}]")

    # 绘图
    try:
        plot_models(data, models, degree)
    except Exception as e:
        print(f"（绘图失败，忽略: {e}）")

    print("\n训练完成！请输入新一批原材料的各批次进料含量。")
    print("支持多个数值，用空格或逗号分隔（例如: 30 50 70）")
    print("输入 'exit' 或 'quit' 退出程序。\n")

    while True:
        raw = input(">>> 请输入各批次含量: ").strip()
        if raw.lower() in ("exit", "quit", "q"):
            print("感谢使用，再见！")
            break

        parts = raw.replace(',', ' ').split()
        try:
            values = [float(x) for x in parts]
        except ValueError:
            print("  输入包含非数字，请重新输入。")
            continue

        if not values:
            print("  未输入任何数值，请重新输入。")
            continue

        out_of_range = [v for v in values if v < 0 or v > 100]
        if out_of_range:
            print(f"  含量 {out_of_range} 超出 0~100 范围，请修正。")
            continue

        n_batches = len(values)
        print(f"\n共 {n_batches} 批原材料，含量分别为: {values}")

        # 预测
        pred_new = [predict_model(models["new"], x) for x in values]
        pred_old = [predict_model(models["old"], x) for x in values]

        # 明细
        print("\n--- 各批次推荐明细 ---")
        for i, x in enumerate(values):
            pn = pred_new[i]
            po = pred_old[i]
            better = "新高炉" if pn > po else ("旧高炉" if po > pn else "相同")
            print(f"  批次{i+1}: 含量 {x:.1f}% → 新高炉预测 {pn:.2f}%, 旧高炉预测 {po:.2f}%  → 推荐 {better}")

        total_avg = sum(0.5 * (pn + po) for pn, po in zip(pred_new, pred_old))
        total_opt = sum(max(pn, po) for pn, po in zip(pred_new, pred_old))
        total_all_new = sum(pred_new)
        total_all_old = sum(pred_old)

        print("\n--- 总产出对比（假设每批质量相同） ---")
        print(f"  平均分配（各一半）总产出: {total_avg:.2f} %")
        print(f"  优化推荐（每批选优）总产出: {total_opt:.2f} %")
        print(f"  全部给新高炉总产出: {total_all_new:.2f} %")
        print(f"  全部给旧高炉总产出: {total_all_old:.2f} %")

        improvement = total_opt - total_avg
        if improvement > 1e-6:
            print(f"\n✅ 推荐采用优化分配，相比平均分配提升 {improvement:.2f} 个百分点")
            print(f"   相对提升幅度: {improvement / total_avg * 100:.2f} %")
        elif improvement < -1e-6:
            print(f"\n⚠️  优化分配低于平均分配（检查模型或数据）")
        else:
            print("\n⚖️  优化分配与平均分配结果相同。")

        if any(v < hist_min or v > hist_max for v in values):
            print(f"\n⚠️  部分输入含量超出历史数据范围 [{hist_min:.1f}, {hist_max:.1f}]，预测可能不准确。")

        print("\n" + "-" * 70 + "\n")

if __name__ == "__main__":
    main()