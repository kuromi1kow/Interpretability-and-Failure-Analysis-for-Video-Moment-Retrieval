#!/usr/bin/env python3
"""Generate plots from temporal bias analysis outputs.

Usage (on SCC):
    cd /projectnb/cs585/projects/VMR/vmr_project
    python analysis/temporal_bias/plot_results.py
"""

import json
import csv
import os
import numpy as np

PROJECT_ROOT = "/projectnb/cs585/projects/VMR/vmr_project"
OUT_DIR = os.path.join(PROJECT_ROOT, "analysis/temporal_bias/outputs")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available — skipping plots. Install with: pip install matplotlib")


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_length_buckets():
    """Bar chart: R1@0.5 by moment length, grouped by model."""
    rows = load_csv(os.path.join(OUT_DIR, "length_bucket_metrics.csv"))
    models = []
    seen = set()
    for r in rows:
        if r["model"] not in seen:
            models.append(r["model"])
            seen.add(r["model"])

    buckets = ["short", "medium", "long"]
    x = np.arange(len(buckets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(models):
        vals = []
        for b in buckets:
            row = [r for r in rows if r["model"] == model and r["bucket"] == b]
            vals.append(float(row[0]["R1@0.5"]) if row else 0)
        ax.bar(x + i * width, vals, width, label=model.replace("_", "-").upper())

    ax.set_xlabel("Moment Length Bucket")
    ax.set_ylabel("R1@0.5")
    ax.set_title("Moment Retrieval R1@0.5 by Moment Length")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Short (0-10s)", "Medium (10-30s)", "Long (30-150s)"])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "length_bucket_r1_05.png"), dpi=150)
    plt.close()
    print("  Saved: length_bucket_r1_05.png")


def plot_position_buckets():
    """Bar chart: R1@0.5 by temporal position, grouped by model."""
    rows = load_csv(os.path.join(OUT_DIR, "position_bucket_metrics.csv"))
    models = []
    seen = set()
    for r in rows:
        if r["model"] not in seen:
            models.append(r["model"])
            seen.add(r["model"])

    buckets = ["early", "middle", "late"]
    x = np.arange(len(buckets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(models):
        vals = []
        for b in buckets:
            row = [r for r in rows if r["model"] == model and r["bucket"] == b]
            vals.append(float(row[0]["R1@0.5"]) if row else 0)
        ax.bar(x + i * width, vals, width, label=model.replace("_", "-").upper())

    ax.set_xlabel("Normalized Temporal Position of GT Moment")
    ax.set_ylabel("R1@0.5")
    ax.set_title("Moment Retrieval R1@0.5 by Temporal Position in Video")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Early (0-33%)", "Middle (33-67%)", "Late (67-100%)"])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "position_bucket_r1_05.png"), dpi=150)
    plt.close()
    print("  Saved: position_bucket_r1_05.png")


def plot_pred_vs_gt_center():
    """Scatter plot: predicted center vs GT center (normalized), for each model."""
    models = ["moment_detr", "qd_detr", "cg_detr"]
    available = [m for m in models if os.path.exists(os.path.join(OUT_DIR, f"{m}_per_sample.csv"))]

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5), squeeze=False)
    for i, model in enumerate(available):
        rows = load_csv(os.path.join(OUT_DIR, f"{model}_per_sample.csv"))
        gt_c = [float(r["norm_gt_center"]) for r in rows]
        pred_c = [float(r["norm_pred_center"]) for r in rows]
        ious = [float(r["iou"]) for r in rows]

        ax = axes[0][i]
        sc = ax.scatter(gt_c, pred_c, c=ious, cmap="RdYlGn", s=8, alpha=0.5, vmin=0, vmax=1)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
        ax.set_xlabel("GT center (normalized)")
        ax.set_ylabel("Pred center (normalized)")
        ax.set_title(model.replace("_", "-").upper())
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        plt.colorbar(sc, ax=ax, label="IoU", shrink=0.8)

    plt.suptitle("Predicted vs GT Moment Center (color = IoU)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pred_vs_gt_center.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: pred_vs_gt_center.png")


def plot_pred_center_histogram():
    """Histogram of predicted centers (normalized) vs GT distribution."""
    models = ["moment_detr", "qd_detr", "cg_detr"]
    available = [m for m in models if os.path.exists(os.path.join(OUT_DIR, f"{m}_per_sample.csv"))]

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 21)

    # GT distribution (from first available model, same for all)
    if available:
        rows = load_csv(os.path.join(OUT_DIR, f"{available[0]}_per_sample.csv"))
        gt_c = [float(r["norm_gt_center"]) for r in rows]
        ax.hist(gt_c, bins=bins, alpha=0.3, color="gray", label="GT", density=True)

    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    for i, model in enumerate(available):
        rows = load_csv(os.path.join(OUT_DIR, f"{model}_per_sample.csv"))
        pred_c = [float(r["norm_pred_center"]) for r in rows]
        ax.hist(pred_c, bins=bins, alpha=0.4, color=colors[i % len(colors)],
                label=model.replace("_", "-").upper() + " pred", density=True, histtype="step", linewidth=2)

    ax.set_xlabel("Normalized temporal position")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Predicted vs GT Moment Centers")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "center_distribution_histogram.png"), dpi=150)
    plt.close()
    print("  Saved: center_distribution_histogram.png")


def main():
    if not HAS_MPL:
        return

    print("Generating plots...")
    plot_length_buckets()
    plot_position_buckets()
    plot_pred_vs_gt_center()
    plot_pred_center_histogram()
    print("All plots saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
