"""
Walk runs/ and aggregate per-run metrics into a single CSV.

For each run directory containing a metrics.json (one JSON object per line,
produced by Detectron2's JSONWriter), pick the eval entry with the highest
bbox/map50_95 as "best".

Columns written per run:
  run_name, question, scenario, model,
  total_epochs, max_epochs, early_stopped, best_epoch,
  best_map50 .. best_map50_95, best_mar100

Usage:
    python experiments/detectron2/collect_results.py
    python experiments/detectron2/collect_results.py --runs-dir runs --out results.csv
"""
import argparse
import csv
import json
import os
from glob import glob

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

METRIC_KEYS = [
    'bbox/map50', 'bbox/map55', 'bbox/map60', 'bbox/map65', 'bbox/map70',
    'bbox/map75', 'bbox/map80', 'bbox/map85', 'bbox/map90', 'bbox/map95',
    'bbox/map50_95', 'bbox/mar100',
]


def load_eval_entries(metrics_path: str) -> list[dict]:
    """Return deduplicated, sorted eval entries from metrics.json."""
    seen_iters = set()
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
            if 'bbox/map50_95' not in obj:
                continue
            it = obj.get('iteration', -1)
            if it in seen_iters:
                continue
            seen_iters.add(it)
            entries.append(obj)
    entries.sort(key=lambda e: e.get('iteration', 0))
    return entries


def load_run_config(run_dir: str) -> dict:
    config_path = os.path.join(run_dir, 'config.yaml')
    if not HAS_YAML or not os.path.exists(config_path):
        return {}
    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def parse_run_name(run_name: str) -> dict:
    # q1_s2_faster_rcnn_2534440 -> question=q1, scenario=s2, model=faster_rcnn, job_id=2534440
    import re
    m = re.match(r'^(.+?)_(\d+)$', run_name)
    job_id = m.group(2) if m else ''
    base = m.group(1) if m else run_name
    parts = base.split('_', 2)
    if len(parts) >= 3 and parts[0].startswith('q') and parts[1].startswith('s'):
        return {'question': parts[0], 'scenario': parts[1], 'model': parts[2], 'job_id': job_id}
    return {'question': '', 'scenario': '', 'model': base, 'job_id': job_id}


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

        cfg = load_run_config(run_dir)
        eval_period = cfg.get('TEST', {}).get('EVAL_PERIOD', None)
        max_iter = cfg.get('SOLVER', {}).get('MAX_ITER', None)

        # iters_per_epoch: prefer config, fall back to first eval point
        iters_per_epoch = eval_period if eval_period else (entries[0]['iteration'] + 1)

        best = max(entries, key=lambda e: e.get('bbox/map50_95', float('-inf')))
        final = entries[-1]

        best_epoch = (best['iteration'] + 1) // iters_per_epoch
        total_epochs = (final['iteration'] + 1) // iters_per_epoch
        max_epochs = (max_iter // iters_per_epoch) if max_iter else ''
        early_stopped = bool(max_epochs) and (total_epochs < max_epochs)

        row = {
            'run_name': run_name,
            **parse_run_name(run_name),
            'total_epochs': total_epochs,
            'max_epochs': max_epochs,
            'early_stopped': early_stopped,
            'best_epoch': best_epoch,
        }
        for k in METRIC_KEYS:
            row[f'best_{k.split("/")[-1]}'] = best.get(k, '')
        rows.append(row)
        print(f'[ok]   {run_name}: best map50_95={best.get("bbox/map50_95", "?"):.3f} '
              f'@ epoch {best_epoch}/{total_epochs}  '
              f'(max={max_epochs})  early_stopped={early_stopped}')

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
