import argparse
import csv
import json
import os
from types import SimpleNamespace

from cabbage_pheno.service.evaluation import run_evaluation


def build_args(input_path, config_path, gt_label_col, gt_instance_col, gt_binarize):
    return SimpleNamespace(
        input=input_path,
        config=config_path,
        gt_label_col=gt_label_col,
        gt_instance_col=gt_instance_col,
        gt_binarize=gt_binarize,
    )


def main():
    parser = argparse.ArgumentParser(description='Run multiple cabbage evaluation configs on the same dataset.')
    parser.add_argument('--input', required=True, help='Path to a file, directory, or glob pattern')
    parser.add_argument('--configs', nargs='*', default=[], help='Config files to compare (may be empty if using --precomputed)')
    parser.add_argument('--labels', nargs='*', help='Optional labels for --configs entries')
    parser.add_argument('--precomputed', nargs='*', default=[], help='Pre-computed JSON result files (from bgpseg_bridge etc.)')
    parser.add_argument('--precomputed-labels', nargs='*', help='Optional labels for --precomputed entries')
    parser.add_argument('--gt-label-col', type=int, default=-2)
    parser.add_argument('--gt-instance-col', type=int, default=-1)
    parser.add_argument('--gt-binarize', action='store_true')
    parser.add_argument('--output', default='output/compare_experiments.csv', help='CSV file for the summary table')
    parser.add_argument('--per-file-output', default='output/compare_experiments_per_file.csv', help='CSV file for per-file rows')
    parser.add_argument('--json-output', default='output/compare_experiments.json', help='JSON file for the full summary')
    args = parser.parse_args()

    labels = args.labels or []
    if labels and len(labels) != len(args.configs):
        raise ValueError('When provided, --labels must match --configs in length.')
    pc_labels = args.precomputed_labels or []
    if pc_labels and len(pc_labels) != len(args.precomputed):
        raise ValueError('When provided, --precomputed-labels must match --precomputed in length.')

    rows = []
    per_file_rows = []

    # ---- config-based runs ----
    for idx, config_path in enumerate(args.configs):
        label = labels[idx] if idx < len(labels) else os.path.splitext(os.path.basename(config_path))[0]
        print('\n' + '=' * 100)
        print(f'Running comparison case: {label}')
        print('=' * 100)

        run_args = build_args(args.input, config_path, args.gt_label_col, args.gt_instance_col, args.gt_binarize)
        result = run_evaluation(run_args)
        summary = result.get('summary', {})
        row = {'label': label, 'config': config_path, 'input': args.input}
        row.update(summary)
        rows.append(row)

        for file_row in result.get('per_file', []):
            detailed_row = {'label': label, 'config': config_path, 'input': args.input}
            detailed_row.update(file_row)
            per_file_rows.append(detailed_row)

    # ---- precomputed runs ----
    for idx, pc_path in enumerate(args.precomputed):
        with open(pc_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        label = pc_labels[idx] if idx < len(pc_labels) else result.get('label', os.path.splitext(os.path.basename(pc_path))[0])
        summary = result.get('summary', {})
        row = {'label': label, 'config': pc_path, 'input': args.input, 'source': 'precomputed'}
        row.update(summary)
        rows.append(row)

        for file_row in result.get('per_file', []):
            detailed_row = {'label': label, 'config': pc_path, 'input': args.input}
            detailed_row.update(file_row)
            per_file_rows.append(detailed_row)

    if not rows:
        print('No results to compare. Provide --configs or --precomputed.')
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.per_file_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.per_file_output)), exist_ok=True)
        per_file_fields = []
        for row in per_file_rows:
            for key in row.keys():
                if key not in per_file_fields:
                    per_file_fields.append(key)
        with open(args.per_file_output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=per_file_fields)
            writer.writeheader()
            writer.writerows(per_file_rows)

    if args.json_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_output)), exist_ok=True)
        payload = {
            'runs': rows,
            'per_file': per_file_rows,
        }
        with open(args.json_output, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print('\nComparison summary:')
    for row in rows:
        print(
            f"{row['label']}: "
            f"mIoU={row.get('avg_sem_miou', 0.0):.4f}, "
            f"InstF1={row.get('avg_inst_f1', 0.0):.4f}, "
            f"Runtime={row.get('avg_runtime', 0.0):.4f}s"
        )
    print(f'CSV written to: {args.output}')
    if args.per_file_output:
        print(f'Per-file CSV written to: {args.per_file_output}')
    if args.json_output:
        print(f'JSON written to: {args.json_output}')


if __name__ == '__main__':
    main()
