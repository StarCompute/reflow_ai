# Copyright (c) 2026 蒲俊杰（Pu Junjie）。保留所有权利。
# 许可见 LICENSE.md：个人使用（含个人商业）免费，企业商业使用需付费授权。

"""EasyTuner 命令行入口：不写代码，一条命令即可训练并出推荐。

用法示例：
    # 【极简·自动模式】什么列都不用填，自动识别可调参数/属性/质量列
    python -m proctune.easy --data 历史样本.csv --new 新任务.csv --out 推荐结果.csv

    # 【专业·显式模式】自己指定各列（数值列里既有可调参数又有属性时更可靠）
    python -m proctune.easy \
        --data 历史样本.csv \
        --knobs 料筒温度,注射压力,保压时间,模具温度 \
        --context 材料,壁厚 \
        --quality 质量 \
        --new 新任务.csv \
        --out 推荐结果.csv

    # 只训练并保存模型（下次免训练）
    python -m proctune.easy --data 历史样本.csv \
        --knobs 料筒温度,注射压力 --context 材料,壁厚 --quality 质量 \
        --save tuner_model.pkl

    # 用已保存的模型直接推荐（不再需要 --knobs/--context/--quality）
    python -m proctune.easy --model tuner_model.pkl --new 新任务.csv --out 推荐结果.csv
"""
import sys
import argparse

from .tuner import EasyTuner, merge_csv


def _split(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m proctune.easy",
        description="EasyTuner 命令行：两张表进、推荐参数出，不用写代码。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # 数据源（二选一）
    p.add_argument("--data", help="历史样本 CSV 路径（训练用）")
    p.add_argument("--model", help="已保存的模型 .pkl 路径（免训练直接推荐）")
    # 列定义（全部可选；不填则自动推断。数值列里既有可调参数又有属性时建议显式指定）
    p.add_argument("--knobs", help="[可选] 可调参数列名，逗号分隔，如 料筒温度,注射压力；不填自动推断")
    p.add_argument("--context", default=None, help="[可选] 产品属性列名，逗号分隔，如 材料,壁厚；不填自动推断")
    p.add_argument("--quality", help="[可选] 质量结果列名，如 质量；不填自动推断")
    p.add_argument("--quality-kind", choices=["defect", "score"], default=None,
                   help="质量类型：defect(缺陷文本)/score(数值评分)；默认自动判断")
    p.add_argument("--n-trees", type=int, default=200, help="随机森林树数（默认 200，越小越快）")
    # 推荐
    p.add_argument("--new", help="新任务 CSV 路径（要出推荐的产品）")
    p.add_argument("--out", help="推荐结果输出 CSV 路径")
    p.add_argument("--top-k", type=int, default=1, help="每个新任务给出的候选数（默认 1）")
    # 追加与保存
    p.add_argument("--add", help="并入一批新数据 CSV 后再训练（列结构需与 --data 一致）")
    p.add_argument("--save", help="训练后把模型保存到该 .pkl 路径")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if not args.data and not args.model:
        print("错误：需要 --data（训练）或 --model（加载已训练模型）二者之一。", file=sys.stderr)
        return 2

    # 1) 得到一个已训练的 tuner
    if args.model:
        tuner = EasyTuner.load(args.model)
        print(f"已加载模型：{args.model}（样本数={tuner.n_samples}）")
    else:
        if args.add:
            merge_csv(args.add, args.data)
            print(f"已把 {args.add} 并入 {args.data}")
        # 列参数全部可选：未提供的传 None，交给 EasyTuner 自动推断
        tuner = EasyTuner(
            data=args.data,
            knob_cols=_split(args.knobs) if args.knobs else None,
            context_cols=_split(args.context) if args.context else None,
            quality_col=args.quality,
            quality_kind=args.quality_kind,
            n_trees=args.n_trees,
        )
        if tuner.auto_inferred:
            print("提示：已启用【自动推断】列定义。若数值列里既有可调参数又有产品属性，"
                  "建议用 --knobs/--context/--quality 显式指定以获得最可靠结果。")
        tuner.train()
        print(tuner.summary())

    # 2) 保存模型（可选）
    if args.save:
        tuner.save(args.save)
        print(f"模型已保存：{args.save}")

    # 3) 推荐（可选）
    if args.new:
        if not args.out:
            print("错误：指定了 --new 就必须同时指定 --out。", file=sys.stderr)
            return 2
        tuner.recommend_to_csv(args.new, args.out, top_k=args.top_k)
        print(f"推荐结果已写出：{args.out}")
        # 顺带把结果打印到屏幕
        with open(args.out, encoding="utf-8-sig") as f:
            for line in f:
                print("  " + line.rstrip())
    elif not args.save:
        print("提示：未提供 --new/--out，仅完成训练。加 --new 新任务.csv --out 推荐结果.csv 可直接出推荐。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
