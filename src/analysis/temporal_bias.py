#!/usr/bin/env python3
"""
VMR Project: Temporal Bias Analysis
Stage 6 of the project — analyzes where models predict moments
and compares against a bias-only baseline.

This script:
1. Computes the training-set temporal distribution of moments
2. Implements a bias-only baseline (predicts from distribution, ignores query/video)
3. Compares model predictions against the bias baseline
4. Generates histograms and analysis figures

Usage:
    python src/analysis/temporal_bias.py \
        --train_annotations data/qvhighlights/annotations/highlight_train_release.jsonl \
        --val_annotations data/qvhighlights/annotations/highlight_val_release.jsonl \
        --predictions results/predictions/moment_detr_val.jsonl \
        --output_dir results/bias_analysis/
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for SCC
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.eval.evaluate import compute_iou, compute_recall_at_iou


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def extract_moment_positions(annotations):
    """
    Extract normalized moment center and duration from annotations.
    Returns arrays of (center_normalized, duration_normalized, start_normalized, end_normalized).
    """
    centers = []
    durations = []
    starts = []
    ends = []

    for item in annotations:
        vid_dur = item.get("duration", 0)
        if vid_dur <= 0:
            continue
        for window in item.get("relevant_windows", []):
            if len(window) == 2:
                s, e = window
                centers.append(((s + e) / 2) / vid_dur)
                durations.append((e - s) / vid_dur)
                starts.append(s / vid_dur)
                ends.append(e / vid_dur)

    return {
        "centers": np.array(centers),
        "durations": np.array(durations),
        "starts": np.array(starts),
        "ends": np.array(ends),
    }


def bias_only_baseline(train_stats, val_annotations):
    """
    Predict moments using only the training-set temporal distribution.
    For each val query, predict the mean start/end from training.
    """
    mean_start = np.mean(train_stats["starts"])
    mean_end = np.mean(train_stats["ends"])

    predictions = {}
    ground_truths = {}

    for item in val_annotations:
        qid = item.get("qid", item.get("query_id"))
        vid_dur = item.get("duration", 0)
        if vid_dur <= 0:
            continue

        # Bias-only prediction: use mean normalized position * video duration
        pred_start = mean_start * vid_dur
        pred_end = mean_end * vid_dur
        predictions[qid] = [pred_start, pred_end]

        ground_truths[qid] = item.get("relevant_windows", [])

    return predictions, ground_truths


def center_baseline(train_stats, val_annotations):
    """
    Even simpler baseline: always predict the center of the video
    with the mean moment duration from training.
    """
    mean_dur_ratio = np.mean(train_stats["durations"])

    predictions = {}
    ground_truths = {}

    for item in val_annotations:
        qid = item.get("qid", item.get("query_id"))
        vid_dur = item.get("duration", 0)
        if vid_dur <= 0:
            continue

        moment_dur = mean_dur_ratio * vid_dur
        center = vid_dur / 2
        pred_start = max(0, center - moment_dur / 2)
        pred_end = min(vid_dur, center + moment_dur / 2)
        predictions[qid] = [pred_start, pred_end]
        ground_truths[qid] = item.get("relevant_windows", [])

    return predictions, ground_truths


def load_model_predictions(pred_path):
    """Load model predictions and extract top-1 moments."""
    predictions = {}
    data = load_jsonl(pred_path)
    for item in data:
        qid = item.get("qid", item.get("query_id"))
        if "pred_relevant_windows" in item:
            predictions[qid] = item["pred_relevant_windows"][0]
        elif "predicted_times" in item:
            predictions[qid] = item["predicted_times"][0]
    return predictions


def extract_pred_positions(predictions, val_annotations):
    """Get normalized center positions from predictions."""
    vid_durs = {}
    for item in val_annotations:
        qid = item.get("qid", item.get("query_id"))
        vid_durs[qid] = item.get("duration", 0)

    centers = []
    for qid, (s, e) in predictions.items():
        vid_dur = vid_durs.get(qid, 0)
        if vid_dur > 0:
            centers.append(((s + e) / 2) / vid_dur)
    return np.array(centers)


def plot_temporal_distribution(train_stats, output_dir, pred_centers=None, model_name=None):
    """Plot histograms of moment positions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Plot 1: Center position distribution (training)
    axes[0].hist(train_stats["centers"], bins=30, alpha=0.7, color="#2E5A88",
                 edgecolor="white", density=True, label="Train GT")
    if pred_centers is not None:
        axes[0].hist(pred_centers, bins=30, alpha=0.5, color="#E8634A",
                     edgecolor="white", density=True, label=f"{model_name} pred")
    axes[0].set_xlabel("Normalized moment center position", fontsize=11)
    axes[0].set_ylabel("Density", fontsize=11)
    axes[0].set_title("Moment Center Distribution", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(0, 1)

    # Plot 2: Duration distribution (training)
    axes[1].hist(train_stats["durations"], bins=30, alpha=0.7, color="#2E5A88",
                 edgecolor="white", density=True)
    axes[1].set_xlabel("Normalized moment duration", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)
    axes[1].set_title("Moment Duration Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlim(0, 1)

    # Plot 3: Start vs End scatter
    sample_idx = np.random.choice(len(train_stats["starts"]),
                                   size=min(500, len(train_stats["starts"])),
                                   replace=False)
    axes[2].scatter(train_stats["starts"][sample_idx], train_stats["ends"][sample_idx],
                    alpha=0.3, s=10, color="#2E5A88")
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    axes[2].set_xlabel("Normalized start position", fontsize=11)
    axes[2].set_ylabel("Normalized end position", fontsize=11)
    axes[2].set_title("Start vs End (Train GT)", fontsize=12, fontweight="bold")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    out_path = output_dir / "temporal_distribution.pdf"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_bias_comparison(results_dict, output_dir):
    """Bar chart comparing bias baselines vs model."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models = list(results_dict.keys())
    metrics = ["R1@0.3", "R1@0.5", "R1@0.7"]
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    colors = ["#AAAAAA", "#888888", "#2E5A88", "#3A8FD6", "#E8634A"]

    for i, model_name in enumerate(models):
        values = [results_dict[model_name].get(m, 0) for m in metrics]
        ax.bar(x + i * width, values, width, label=model_name,
               color=colors[i % len(colors)], edgecolor="white")

    ax.set_xlabel("Metric", fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Bias Baselines vs Model Performance", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 80)

    plt.tight_layout()
    out_path = output_dir / "bias_comparison.pdf"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Temporal Bias Analysis")
    parser.add_argument("--train_annotations", type=str, required=True)
    parser.add_argument("--val_annotations", type=str, required=True)
    parser.add_argument("--predictions", type=str, default=None,
                        help="Model predictions JSONL (optional, for comparison)")
    parser.add_argument("--model_name", type=str, default="Model")
    parser.add_argument("--output_dir", type=str, default="results/bias_analysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Compute training set temporal distribution
    print("=" * 60)
    print("Step 1: Training set temporal distribution")
    print("=" * 60)
    train_data = load_jsonl(args.train_annotations)
    train_stats = extract_moment_positions(train_data)

    print(f"  Training moments: {len(train_stats['centers'])}")
    print(f"  Mean center position: {np.mean(train_stats['centers']):.3f} "
          f"(± {np.std(train_stats['centers']):.3f})")
    print(f"  Mean duration ratio:  {np.mean(train_stats['durations']):.3f} "
          f"(± {np.std(train_stats['durations']):.3f})")
    print(f"  Mean start position:  {np.mean(train_stats['starts']):.3f}")
    print(f"  Mean end position:    {np.mean(train_stats['ends']):.3f}")

    # Step 2: Bias-only baselines
    print()
    print("=" * 60)
    print("Step 2: Bias-only baselines on validation set")
    print("=" * 60)
    val_data = load_jsonl(args.val_annotations)

    # Baseline 1: Mean position
    mean_preds, gt = bias_only_baseline(train_stats, val_data)
    mean_recall = compute_recall_at_iou(mean_preds, gt)
    print(f"\n  Mean-position baseline:")
    for k, v in mean_recall.items():
        print(f"    {k}: {v:.2f}")

    # Baseline 2: Center position
    center_preds, gt = center_baseline(train_stats, val_data)
    center_recall = compute_recall_at_iou(center_preds, gt)
    print(f"\n  Center-of-video baseline:")
    for k, v in center_recall.items():
        print(f"    {k}: {v:.2f}")

    # Step 3: Compare with model predictions (if provided)
    results_dict = {
        "Center baseline": center_recall,
        "Mean-pos baseline": mean_recall,
    }

    pred_centers = None
    if args.predictions and Path(args.predictions).exists():
        print()
        print("=" * 60)
        print(f"Step 3: Comparing with {args.model_name}")
        print("=" * 60)
        model_preds = load_model_predictions(args.predictions)
        model_recall = compute_recall_at_iou(model_preds, gt)
        print(f"\n  {args.model_name}:")
        for k, v in model_recall.items():
            print(f"    {k}: {v:.2f}")
        results_dict[args.model_name] = model_recall
        pred_centers = extract_pred_positions(model_preds, val_data)

    # Step 4: Generate figures
    print()
    print("=" * 60)
    print("Step 4: Generating figures")
    print("=" * 60)
    plot_temporal_distribution(train_stats, output_dir, pred_centers, args.model_name)
    plot_bias_comparison(results_dict, output_dir)

    # Step 5: Save summary
    summary = {
        "training_stats": {
            "n_moments": int(len(train_stats["centers"])),
            "mean_center": float(np.mean(train_stats["centers"])),
            "std_center": float(np.std(train_stats["centers"])),
            "mean_duration": float(np.mean(train_stats["durations"])),
            "std_duration": float(np.std(train_stats["durations"])),
        },
        "baseline_results": {
            "center_baseline": {k: float(v) for k, v in center_recall.items()},
            "mean_position_baseline": {k: float(v) for k, v in mean_recall.items()},
        }
    }
    if args.predictions and Path(args.predictions).exists():
        summary["model_results"] = {
            args.model_name: {k: float(v) for k, v in model_recall.items()}
        }

    summary_path = output_dir / "bias_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    print()
    print("Bias analysis complete!")


if __name__ == "__main__":
    main()
