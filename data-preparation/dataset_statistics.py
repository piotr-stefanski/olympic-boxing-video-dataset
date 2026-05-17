"""
Dataset Statistics for Olympic Boxing Video Dataset (COCO format).

Loads per-fold COCO annotation files and produces:
  1. Total images & annotations
  2. Per-fold breakdown
  3. Per-category breakdown
For each table the script prints a formatted text table *and* a LaTeX table,
and saves a publication-ready figure to the output directory.

Usage:
    uv run dataset_statistics.py
    uv run dataset_statistics.py --annotations-dir /path/to/annotations --output-dir ./stats_output
"""
import json
import os
import argparse
from collections import defaultdict, Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ──────────────────────────── helpers ────────────────────────────

def load_all_folds(annotations_dir: str, fold_numbers: list[int]):
    """Return combined lists of images, annotations, and category map."""
    all_images = []
    all_annotations = []
    categories = None

    for fold_num in fold_numbers:
        path = os.path.join(annotations_dir, f"annotations_fold_{fold_num}.json")
        if not os.path.exists(path):
            print(f"[WARN] {path} not found, skipping.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if categories is None:
            categories = {c["id"]: c["name"] for c in data["categories"]}
        all_images.extend(data["images"])
        all_annotations.extend(data["annotations"])

    return all_images, all_annotations, categories


def category_short_name(name: str) -> str:
    """Shorten long category names for figure labels."""
    replacements = {
        "Punch to the head with the left hand": "Head L",
        "Punch to the head with the right hand": "Head R",
        "Punch to the torso with the left hand": "Torso L",
        "Punch to the torso with the right hand": "Torso R",
        "Block with the left hand": "Block L",
        "Block with the right hand": "Block R",
        "Missed punch with the left hand": "Miss L",
        "Missed punch with the right hand": "Miss R",
    }
    return replacements.get(name, name)


# ──────────────────────────── printing ────────────────────────────

def print_text_table(headers: list[str], rows: list[list], title: str = ""):
    """Print a nicely formatted ASCII table."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    col_widths = [max(len(str(h)), *(len(str(r[i])) for r in rows))
                  for i, h in enumerate(headers)]

    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))

    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row)))
    print()


def print_latex_table(headers: list[str], rows: list[list], caption: str, label: str):
    """Print a LaTeX table ready to paste into a paper."""
    n = len(headers)
    col_spec = "l" + "r" * (n - 1)
    print(f"% ---- LaTeX table: {caption} ----")
    print(r"\begin{table}[ht]")
    print(r"  \centering")
    print(f"  \\caption{{{caption}}}")
    print(f"  \\label{{{label}}}")
    print(f"  \\begin{{tabular}}{{{col_spec}}}")
    print(r"    \toprule")
    print("    " + " & ".join(headers) + r" \\")
    print(r"    \midrule")
    for row in rows:
        print("    " + " & ".join(str(v) for v in row) + r" \\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    print()


# ──────────────────────────── figures ────────────────────────────

# Colour palette (8 muted tones that work in print & on screen)
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]

def _style_ax(ax):
    """Common styling for publication figures."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))


def save_total_summary_figure(total_images: int, total_annotations: int,
                              output_dir: str):
    """Bar chart with total images vs annotations."""
    fig, ax = plt.subplots(figsize=(4, 3.5))
    bars = ax.bar(["Images", "Annotations"],
                  [total_images, total_annotations],
                  color=[PALETTE[0], PALETTE[1]], width=0.5, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{int(bar.get_height()):,}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Dataset Totals", fontsize=13, fontweight="bold")
    _style_ax(ax)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig_dataset_totals.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {path}")


def save_fold_figure(fold_stats: dict, output_dir: str):
    """Grouped bar chart: images & annotations per fold."""
    folds = sorted(fold_stats.keys())
    images_vals = [fold_stats[f]["images"] for f in folds]
    anns_vals = [fold_stats[f]["annotations"] for f in folds]

    x = np.arange(len(folds))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    bars1 = ax.bar(x - w / 2, images_vals, w, label="Images",
                   color=PALETTE[0], edgecolor="white")
    bars2 = ax.bar(x + w / 2, anns_vals, w, label="Annotations",
                   color=PALETTE[1], edgecolor="white")

    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    f"{int(bar.get_height()):,}", ha="center", va="bottom",
                    fontsize=9)

    fold_labels = [f.replace("_", " ").title() for f in folds]
    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels, fontsize=10)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Images & Annotations per Fold", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    _style_ax(ax)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig_per_fold.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {path}")


def save_category_figure(cat_stats: dict, categories: dict, output_dir: str):
    """Horizontal bar chart: images & annotations per category."""
    cat_ids = sorted(cat_stats.keys())
    labels = [category_short_name(categories[cid]) for cid in cat_ids]
    images_vals = [cat_stats[cid]["images"] for cid in cat_ids]
    anns_vals = [cat_stats[cid]["annotations"] for cid in cat_ids]

    y = np.arange(len(cat_ids))
    h = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.barh(y - h / 2, images_vals, h, label="Images",
                    color=PALETTE[0], edgecolor="white")
    bars2 = ax.barh(y + h / 2, anns_vals, h, label="Annotations",
                    color=PALETTE[1], edgecolor="white")

    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height() / 2,
                    f"{int(bar.get_width()):,}", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Count", fontsize=11)
    ax.set_title("Images & Annotations per Category", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.invert_yaxis()
    _style_ax(ax)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig_per_category.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {path}")


def save_category_images_only_figure(cat_stats: dict, categories: dict, output_dir: str):
    """Horizontal bar chart: images per category (no annotations)."""
    cat_ids = sorted(cat_stats.keys())
    labels = [category_short_name(categories[cid]) for cid in cat_ids]
    images_vals = [cat_stats[cid]["images"] for cid in cat_ids]

    y = np.arange(len(cat_ids))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(y, images_vals, height=0.55,
                   color=[PALETTE[i % len(PALETTE)] for i in range(len(cat_ids))],
                   edgecolor="white")

    for bar in bars:
        ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width()):,}", va="center", fontsize=10)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Number of Images", fontsize=11)
    ax.set_title("Class Distribution (Images)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    _style_ax(ax)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig_class_distribution_images.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {path}")


# ──────────────────────────── main ────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute and display dataset statistics from COCO annotation folds"
    )
    parser.add_argument(
        "--annotations-dir",
        default="../datasets/olympic-boxing-video-dataset/annotations",
        help="Directory with annotations_fold_*.json files",
    )
    parser.add_argument(
        "--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5],
        help="Fold numbers to include (default: 1 2 3 4 5)",
    )
    parser.add_argument(
        "--output-dir", default="stats_output",
        help="Directory to save figures (default: stats_output)",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── load data ───────────────────────────────────────────────
    all_images, all_annotations, categories = load_all_folds(
        args.annotations_dir, args.folds
    )
    total_images = len(all_images)
    total_annotations = len(all_annotations)

    # ── 1. Total summary ───────────────────────────────────────
    print_text_table(
        ["Metric", "Count"],
        [["Total images", f"{total_images:,}"],
         ["Total annotations", f"{total_annotations:,}"]],
        title="Overall Dataset Summary",
    )
    print_latex_table(
        ["Metric", "Count"],
        [["Total images", f"{total_images:,}"],
         ["Total annotations", f"{total_annotations:,}"]],
        caption="Overall dataset summary.",
        label="tab:dataset_summary",
    )
    save_total_summary_figure(total_images, total_annotations, args.output_dir)

    # ── 2. Per-fold statistics ─────────────────────────────────
    fold_stats: dict[str, dict] = defaultdict(lambda: {"images": 0, "annotations": 0})
    # count images per fold
    for img in all_images:
        fold_stats[img["fold_number"]]["images"] += 1
    # count annotations per fold (via image_id → fold lookup)
    img_id_to_fold = {img["id"]: img["fold_number"] for img in all_images}
    for ann in all_annotations:
        fold = img_id_to_fold.get(ann["image_id"])
        if fold:
            fold_stats[fold]["annotations"] += 1

    fold_rows = []
    for fold in sorted(fold_stats):
        fold_rows.append([
            fold.replace("_", " ").title(),
            f'{fold_stats[fold]["images"]:,}',
            f'{fold_stats[fold]["annotations"]:,}',
        ])
    fold_rows.append([
        "\\textbf{Total}",
        f"\\textbf{{{total_images:,}}}",
        f"\\textbf{{{total_annotations:,}}}",
    ])

    print_text_table(
        ["Fold", "Images", "Annotations"],
        fold_rows[:-1],   # skip LaTeX-formatted total for text table
        title="Statistics by Fold",
    )
    # Add a plain-text total row for the text table
    print(f"  Total: {total_images:,} images, {total_annotations:,} annotations\n")

    print_latex_table(
        ["Fold", "Images", "Annotations"],
        fold_rows,
        caption="Number of images and annotations per fold.",
        label="tab:per_fold",
    )
    save_fold_figure(fold_stats, args.output_dir)

    # ── 3. Per-category statistics ─────────────────────────────
    cat_stats: dict[int, dict] = defaultdict(lambda: {"images": 0, "annotations": 0})
    # count annotations per category
    for ann in all_annotations:
        cat_stats[ann["category_id"]]["annotations"] += 1

    # count unique images that contain at least one annotation of each category
    cat_image_sets: dict[int, set] = defaultdict(set)
    for ann in all_annotations:
        cat_image_sets[ann["category_id"]].add(ann["image_id"])
    for cid in cat_image_sets:
        cat_stats[cid]["images"] = len(cat_image_sets[cid])

    cat_rows = []
    for cid in sorted(categories.keys()):
        cat_rows.append([
            categories[cid],
            f'{cat_stats[cid]["images"]:,}',
            f'{cat_stats[cid]["annotations"]:,}',
        ])

    # Use the true unique totals (an image can belong to multiple categories)
    cat_rows_with_total = cat_rows + [[
        "Total",
        f"{total_images:,}",
        f"{total_annotations:,}",
    ]]

    print_text_table(
        ["Category", "Images", "Annotations"],
        cat_rows_with_total,
        title="Statistics by Category",
    )
    print("  Note: An image may contain annotations of multiple categories,")
    print("        so per-category image counts do not sum to the total.\n")

    cat_rows_latex = []
    for cid in sorted(categories.keys()):
        cat_rows_latex.append([
            categories[cid],
            f'{cat_stats[cid]["images"]:,}',
            f'{cat_stats[cid]["annotations"]:,}',
        ])
    cat_rows_latex.append([
        "\\textbf{Total}",
        f"\\textbf{{{total_images:,}}}",
        f"\\textbf{{{total_annotations:,}}}",
    ])

    print_latex_table(
        ["Category", "Images", "Annotations"],
        cat_rows_latex,
        caption="Number of images and annotations per category. "
                "An image may contain annotations of multiple categories, "
                "so per-category image counts do not sum to the total.",
        label="tab:per_category",
    )
    save_category_figure(cat_stats, categories, args.output_dir)
    save_category_images_only_figure(cat_stats, categories, args.output_dir)

    print(f"[DONE] All tables printed and figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
