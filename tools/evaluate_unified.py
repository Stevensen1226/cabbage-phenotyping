#!/usr/bin/env python3
"""
统一评估入口 — 单条命令完成 val/test 划分评估

用法:
  # 在 Val 上调参
  python evaluate.py --input evalaute_test --config configs/default.yaml --split val

  # 参数冻结后在 Test 上评估
  python evaluate.py --input evalaute_test --config configs/default.yaml --split test

  # 全量评估 (不拆分)
  python evaluate.py --input evalaute_test --config configs/default.yaml --split all

  # 兼容原有用法 (无 --split)
  python evaluate.py --input evalaute_test --config configs/default.yaml

  # 一次跑完 val + test
  python tools/evaluate_unified.py --config configs/default.yaml

输出:
  output/{config}_{split}.json      — 详细指标 (per-file + summary)
  output/{config}_compare.json      — val vs test 对比 (仅 unified 模式)
"""
import argparse
import json
import logging
import os
import sys
import time

# 将项目根加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EvalUnified")


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_summary(label: str, summary: dict):
    print(f"\n  [{label}]")
    print(f"    Files:        {summary.get('processed_files', '?')}")
    print(f"    Inst Prec:    {summary.get('avg_inst_prec', 0):.4f}")
    print(f"    Inst Rec:     {summary.get('avg_inst_rec', 0):.4f}")
    print(f"    Inst F1:      {summary.get('avg_inst_f1', 0):.4f}")
    print(f"    Inst mIoU:    {summary.get('avg_inst_miou', 0):.4f}")
    print(f"    Sem mIoU:     {summary.get('avg_sem_miou', 0):.4f}")
    print(f"    Count MAE:    {summary.get('mae_count', 0):.2f}")
    print(f"    Avg Runtime:  {summary.get('avg_runtime', 0):.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description='统一评估 — 自动处理 val/test 划分并对比',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tools/evaluate_unified.py --config configs/default.yaml
  python tools/evaluate_unified.py --config configs/default.yaml --input evalaute_test
  python tools/evaluate_unified.py --config configs/default.yaml --input evalaute_test --skip-val
        """,
    )
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--input', type=str, default='evalaute_test')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--skip-val', action='store_true', help='Skip val, only run test')
    parser.add_argument('--skip-test', action='store_true', help='Skip test, only run val')
    parser.add_argument('--gt-label-col', type=int, default=-2)
    parser.add_argument('--gt-instance-col', type=int, default=-1)
    parser.add_argument('--gt-binarize', action='store_true')

    args_ns = parser.parse_args()
    cfg_name = os.path.splitext(os.path.basename(args_ns.config))[0]
    os.makedirs(args_ns.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 构建 base args
    # ------------------------------------------------------------------
    from types import SimpleNamespace
    base_args = SimpleNamespace(
        input=args_ns.input,
        config=args_ns.config,
        split=None,
        output=None,
        gt_label_col=args_ns.gt_label_col,
        gt_instance_col=args_ns.gt_instance_col,
        gt_binarize=args_ns.gt_binarize,
    )

    results = {}
    t_start = time.time()

    # 在此处导入，确保 --help 在无 numpy 环境下仍可用
    from cabbage_pheno.service.evaluation import run_evaluation

    # ------------------------------------------------------------------
    # Val
    # ------------------------------------------------------------------
    if not args_ns.skip_val:
        print_header("VALIDATION SET (调参用)")
        base_args.split = 'val'
        base_args.output = os.path.join(args_ns.output_dir, f'{cfg_name}_val.json')
        results['val'] = run_evaluation(base_args)
        print_summary('Val', results['val'].get('summary', {}))
    else:
        logger.info("Skipping val (--skip-val)")

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    if not args_ns.skip_test:
        print_header("TEST SET (最终报告)")
        base_args.split = 'test'
        base_args.output = os.path.join(args_ns.output_dir, f'{cfg_name}_test.json')
        results['test'] = run_evaluation(base_args)
        print_summary('Test', results['test'].get('summary', {}))
    else:
        logger.info("Skipping test (--skip-test)")

    total_time = time.time() - t_start

    # ------------------------------------------------------------------
    # Compare & Save
    # ------------------------------------------------------------------
    compare = {
        'config': args_ns.config,
        'input': args_ns.input,
        'total_runtime': round(total_time, 1),
    }
    for split_name in ('val', 'test'):
        if split_name in results:
            compare[f'{split_name}_summary'] = results[split_name].get('summary', {})
            compare[f'{split_name}_files'] = results[split_name].get('processed_files', [])

    compare_path = os.path.join(args_ns.output_dir, f'{cfg_name}_compare.json')
    with open(compare_path, 'w') as f:
        json.dump(compare, f, indent=2)

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    print_header("COMPARISON")
    if 'val' in results and 'test' in results:
        vs = results['val']['summary']
        ts = results['test']['summary']
        print(f"\n  {'Metric':<18} {'Val':>10} {'Test':>10} {'Delta':>10}")
        print(f"  {'-'*48}")
        for key, label in [
            ('avg_inst_prec', 'Inst Prec'),
            ('avg_inst_rec', 'Inst Rec'),
            ('avg_inst_f1', 'Inst F1'),
            ('avg_inst_miou', 'Inst mIoU'),
            ('avg_sem_miou', 'Sem mIoU'),
            ('mae_count', 'Count MAE'),
        ]:
            v = vs.get(key, 0)
            t = ts.get(key, 0)
            delta = t - v
            sign = '+' if delta > 0 else ''
            print(f"  {label:<18} {v:>10.4f} {t:>10.4f} {sign}{delta:>9.4f}")

    print(f"\n  Results saved to: {args_ns.output_dir}/")
    print(f"    {cfg_name}_val.json")
    print(f"    {cfg_name}_test.json")
    print(f"    {cfg_name}_compare.json")
    print(f"\n  Total time: {total_time:.1f}s")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
