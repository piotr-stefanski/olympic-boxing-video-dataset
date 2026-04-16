"""
Walk runs/ and aggregate per-run metrics into a single CSV.

For each run directory containing a metrics.json (one JSON object per line,
produced by Detectron2's JSONWriter), pick the eval entry with the highest
bbox/map50_95 as "best", and the last eval entry as "final".

Usage:
    python scripts/collect_results.py
    python scripts/collect_results.py --runs-dir runs --out results.csv
"""
import argparse
import csv
import json
import os
from glob import glob

METRIC_KEYS = [
    'bbox/map50', 'bbox/map55', 'bbox/map60', 'bbox/map65', 'bbox/map70',
    'bbox/map75', 'bbox/map80', 'bbox/map85', 'bbox/map90', 'bbox/map95',
    'bbox/map50_95', 'bbox/mar100',
]


def load_eval_entries(metrics_path: str) -> list[dict]:
    entries = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'bbox/map50_95' in obj:
                entries.append(obj)
    return entries


def parse_run_name(run_name: str) -> dict:
    # q1_s2_faster_rcnn_R50_FPN -> question=q1, scenario=s2, model=faster_rcnn_R50_FPN
    parts = run_name.split('_', 2)
    if len(parts) >= 3 and parts[0].startswith('q') and parts[1].startswith('s'):
        return {'question': parts[0], 'scenario': parts[1], 'model': parts[2]}
    return {'question': '', 'scenario': '', 'model': run_name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', default='runs')
    ap.add_argument('--out', default='results.csv')
    args = ap.parse_args()

    rows = []
    for metrics_path in sorted(glob(os.path.join(args.runs_dir, '*', 'metrics.json'))):
        run_dir = os.path.dirname(metrics_path)
        run_name = os.path.basename(run_dir)
        entries = load_eval_entries(metrics_path)
        if not entries:
            print(f'[skip] {run_name}: no eval entries')
            continue

        best = max(entries, key=lambda e: e.get('bbox/map50_95', float('-inf')))
        final = entries[-1]

        row = {'run_name': run_name, **parse_run_name(run_name),
               'num_evals': len(entries),
               'best_iteration': best.get('iteration', ''),
               'final_iteration': final.get('iteration', '')}
        for k in METRIC_KEYS:
            row[f'best_{k.split("/")[-1]}'] = best.get(k, '')
        for k in METRIC_KEYS:
            row[f'final_{k.split("/")[-1]}'] = final.get(k, '')
        rows.append(row)
        print(f'[ok]   {run_name}: best map50_95={best.get("bbox/map50_95", "?"):.3f} '
              f'@iter {best.get("iteration", "?")}')

    if not rows:
        print('No runs found.')
        return

    fieldnames = list(rows[0].keys())
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f'\nWrote {len(rows)} rows -> {args.out}')


if __name__ == '__main__':
    main()
