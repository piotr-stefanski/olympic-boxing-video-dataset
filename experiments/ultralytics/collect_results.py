"""
Collect Ultralytics training results from all run directories into results_ultralytics.csv.

Each run dir must contain results.csv (written by Ultralytics) and args.yaml.
Per-IoU metrics (map50..map95) are read from per_iou_metrics.json when present
(written by train_ultralytics.py after final validation on best.pt).
Best epoch is the one with the highest metrics/mAP50-95(B).

Usage:
    python experiments/ultralytics/collect_results.py
    python experiments/ultralytics/collect_results.py --runs-dir runs --output results_ultralytics.csv
"""
import argparse
import csv
import json
import os
import re
import sys

import yaml


def parse_run_name(dir_name: str) -> tuple[str, str]:
    """Strip trailing _JOBID from dir name. Returns (run_name, job_id)."""
    m = re.match(r'^(.+?)_(\d+)$', dir_name)
    if m:
        return m.group(1), m.group(2)
    return dir_name, ''


def collect_run(run_dir: str) -> dict | None:
    results_csv = os.path.join(run_dir, 'results.csv')
    if not os.path.exists(results_csv):
        return None

    rows = []
    try:
        with open(results_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k.strip(): v.strip() for k, v in row.items()})
    except Exception as e:
        print(f'  WARNING: could not read {results_csv}: {e}', file=sys.stderr)
        return None

    if not rows:
        return None

    sample = rows[0]
    map_col = next((c for c in sample if 'mAP50-95' in c and 'val' not in c.lower()), None)
    map50_col = next((c for c in sample if 'mAP50' in c and '95' not in c and 'val' not in c.lower()), None)
    epoch_col = 'epoch' if 'epoch' in sample else list(sample.keys())[0]

    if map_col is None:
        print(f'  WARNING: no mAP50-95 column in {results_csv}', file=sys.stderr)
        print(f'  Available columns: {list(sample.keys())}', file=sys.stderr)
        return None

    best_row = max(rows, key=lambda r: float(r[map_col]) if r[map_col] else float('-inf'))
    total_epochs = int(rows[-1][epoch_col])  # Ultralytics epochs are 1-indexed
    best_epoch = int(best_row[epoch_col])

    # Load args.yaml for configured max epochs
    max_epochs = total_epochs
    args_yaml = os.path.join(run_dir, 'args.yaml')
    if os.path.exists(args_yaml):
        with open(args_yaml) as f:
            args_data = yaml.safe_load(f)
        max_epochs = args_data.get('epochs', max_epochs)

    early_stopped = total_epochs < max_epochs

    dir_name = os.path.basename(run_dir)
    run_name, job_id = parse_run_name(dir_name)

    parts = run_name.split('_', 2)
    question = parts[0] if len(parts) > 0 else ''
    scenario = parts[1] if len(parts) > 1 else ''
    model = parts[2] if len(parts) > 2 else ''

    # Per-IoU metrics from final validation on best.pt (written by train_ultralytics.py)
    per_iou_path = os.path.join(run_dir, 'per_iou_metrics.json')
    if os.path.exists(per_iou_path):
        with open(per_iou_path) as f:
            per_iou = json.load(f)
        best_map50    = per_iou.get('map50', float('nan'))
        best_map55    = per_iou.get('map55', float('nan'))
        best_map60    = per_iou.get('map60', float('nan'))
        best_map65    = per_iou.get('map65', float('nan'))
        best_map70    = per_iou.get('map70', float('nan'))
        best_map75    = per_iou.get('map75', float('nan'))
        best_map80    = per_iou.get('map80', float('nan'))
        best_map85    = per_iou.get('map85', float('nan'))
        best_map90    = per_iou.get('map90', float('nan'))
        best_map95    = per_iou.get('map95', float('nan'))
        best_map50_95 = per_iou.get('map50_95', float('nan'))
    else:
        # Fallback: read from results.csv (only map50 and map50-95 available)
        nan = float('nan')
        best_map50    = float(best_row[map50_col]) * 100 if map50_col else nan
        best_map55 = best_map60 = best_map65 = best_map70 = nan
        best_map75 = best_map80 = best_map85 = best_map90 = best_map95 = nan
        best_map50_95 = float(best_row[map_col]) * 100

    return {
        'run_name':     run_name,
        'question':     question,
        'scenario':     scenario,
        'model':        model,
        'job_id':       job_id,
        'total_epochs': total_epochs,
        'max_epochs':   max_epochs,
        'early_stopped': early_stopped,
        'best_epoch':   best_epoch,
        'best_map50':   best_map50,
        'best_map55':   best_map55,
        'best_map60':   best_map60,
        'best_map65':   best_map65,
        'best_map70':   best_map70,
        'best_map75':   best_map75,
        'best_map80':   best_map80,
        'best_map85':   best_map85,
        'best_map90':   best_map90,
        'best_map95':   best_map95,
        'best_map50_95': best_map50_95,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', default='runs')
    parser.add_argument('--output', default='results_ultralytics.csv')
    args = parser.parse_args()

    if not os.path.isdir(args.runs_dir):
        print(f'Runs directory not found: {args.runs_dir}')
        sys.exit(1)

    result_rows = []
    for dir_name in sorted(os.listdir(args.runs_dir)):
        run_dir = os.path.join(args.runs_dir, dir_name)
        if not os.path.isdir(run_dir) or dir_name.startswith('_'):
            continue
        if not os.path.exists(os.path.join(run_dir, 'results.csv')):
            continue

        print(f'Processing: {dir_name}')
        row = collect_run(run_dir)
        if row:
            result_rows.append(row)
            per_iou_note = '(per-IoU from json)' if os.path.exists(
                os.path.join(run_dir, 'per_iou_metrics.json')) else '(map50/map50-95 only)'
            print(f'  best_epoch={row["best_epoch"]}, best_map50_95={row["best_map50_95"]:.3f}, '
                  f'total_epochs={row["total_epochs"]}/{row["max_epochs"]}, '
                  f'early_stopped={row["early_stopped"]} {per_iou_note}')

    if not result_rows:
        print('No results found.')
        return

    fieldnames = list(result_rows[0].keys())
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    print(f'\nSaved {len(result_rows)} run(s) to {args.output}')


if __name__ == '__main__':
    main()
