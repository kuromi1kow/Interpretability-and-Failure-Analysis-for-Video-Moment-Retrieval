#!/usr/bin/env python3
"""
VMR Project: Result Visualization and Table Generation
Compiles all experiment results into publication-quality figures and LaTeX tables.

Usage:
    python src/utils/generate_figures.py \
        --results_csv results/eval_results.csv \
        --bias_summary results/bias_analysis/bias_analysis_summary.json \
        --linguistic_summary results/linguistic/linguistic_analysis_summary.json \
        --output_dir results/figures/
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def set_plot_style():
    """Set consistent publication-quality plot style."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.figsize": (8, 5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


COLORS = {
    "Moment-DETR": "#2E5A88",
    "QD-DETR": "#3A8FD6",
    "CG-DETR": "#E8634A",
    "Center baseline": "#AAAAAA",
    "Mean-pos baseline": "#888888",
    "Full query": "#2E5A88",
    "Verb-masked": "#E8634A",
    "Noun-masked": "#5DAE5F",
    "Verb-swapped": "#D4A843",
}


def plot_baseline_comparison(results_csv, output_dir):
    """Bar chart comparing all three baseline models."""
    df = pd.read_csv(results_csv)
    if df.empty:
        print("  No results in CSV, skipping baseline comparison plot")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    models = df["model"].tolist()
    metrics = ["R1@0.3", "R1@0.5", "R1@0.7", "mAP", "HIT@1"]
    available_metrics = [m for m in metrics if m in df.columns]

    x = np.arange(len(available_metrics))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        values = [df[df["model"] == model][m].values[0] for m in available_metrics]
        color = COLORS.get(model, f"C{i}")
        ax.bar(x + i * width, values, width, label=model, color=color, edgecolor="white")

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score (%)")
    ax.set_title("Baseline Model Comparison on QVHighlights Val")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(available_metrics)
    ax.legend()
    ax.set_ylim(0, 80)

    plt.tight_layout()
    out_path = output_dir / "baseline_comparison.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def generate_latex_table(results_csv, output_dir):
    """Generate LaTeX table from results CSV."""
    df = pd.read_csv(results_csv)
    if df.empty:
        print("  No results, skipping LaTeX table")
        return

    # Build LaTeX
    metrics = ["R1@0.3", "R1@0.5", "R1@0.7", "mAP", "HIT@1"]
    available_metrics = [m for m in metrics if m in df.columns]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Baseline reproduction results on QVHighlights val set.}")
    lines.append("\\label{tab:baselines}")

    col_spec = "l" + "c" * len(available_metrics)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header
    header = "Model & " + " & ".join(available_metrics) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Rows
    for _, row in df.iterrows():
        values = [f"{row[m]:.1f}" if pd.notna(row.get(m)) else "---" for m in available_metrics]
        line = f"{row['model']} & " + " & ".join(values) + " \\\\"
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex_str = "\n".join(lines)
    out_path = output_dir / "baselines_table.tex"
    with open(out_path, "w") as f:
        f.write(latex_str)
    print(f"  Saved: {out_path}")
    print()
    print(latex_str)


def plot_prediction_timeline(predictions_jsonl, gt_jsonl, output_dir, num_examples=10):
    """
    For a few examples, plot GT vs predicted intervals on a timeline.
    """
    # Load data
    gt_data = {}
    with open(gt_jsonl, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            qid = item.get("qid", item.get("query_id"))
            gt_data[qid] = {
                "query": item.get("query", ""),
                "duration": item.get("duration", 0),
                "windows": item.get("relevant_windows", []),
            }

    pred_data = {}
    with open(predictions_jsonl, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            qid = item.get("qid", item.get("query_id"))
            if "pred_relevant_windows" in item:
                pred_data[qid] = item["pred_relevant_windows"][0]
            elif "predicted_times" in item:
                pred_data[qid] = item["predicted_times"][0]

    # Pick examples where we have both GT and predictions
    common_qids = list(set(gt_data.keys()) & set(pred_data.keys()))[:num_examples]

    if not common_qids:
        print("  No common examples found, skipping timeline plot")
        return

    fig, axes = plt.subplots(len(common_qids), 1,
                              figsize=(10, len(common_qids) * 1.2 + 1))
    if len(common_qids) == 1:
        axes = [axes]

    for i, qid in enumerate(common_qids):
        ax = axes[i]
        gt = gt_data[qid]
        pred = pred_data[qid]

        duration = gt["duration"]
        query = gt["query"]

        # Draw timeline
        ax.set_xlim(0, duration)
        ax.set_ylim(-0.5, 1.5)

        # GT intervals (blue)
        for window in gt["windows"]:
            ax.barh(0, window[1] - window[0], left=window[0], height=0.6,
                    color="#2E5A88", alpha=0.7, edgecolor="white")

        # Predicted interval (red)
        ax.barh(1, pred[1] - pred[0], left=pred[0], height=0.6,
                color="#E8634A", alpha=0.7, edgecolor="white")

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["GT", "Pred"], fontsize=8)
        ax.set_title(f'"{query[:60]}..."' if len(query) > 60 else f'"{query}"',
                     fontsize=9, loc="left")
        ax.tick_params(axis="x", labelsize=8)

    axes[-1].set_xlabel("Time (seconds)", fontsize=10)

    # Legend
    gt_patch = mpatches.Patch(color="#2E5A88", alpha=0.7, label="Ground truth")
    pred_patch = mpatches.Patch(color="#E8634A", alpha=0.7, label="Predicted")
    fig.legend(handles=[gt_patch, pred_patch], loc="upper right", fontsize=9)

    plt.suptitle("Prediction Timelines", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = output_dir / "prediction_timelines.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# Need this import for the timeline function
import matplotlib.patches as mpatches


def main():
    parser = argparse.ArgumentParser(description="Generate Result Figures")
    parser.add_argument("--results_csv", type=str, default="results/eval_results.csv")
    parser.add_argument("--bias_summary", type=str, default=None)
    parser.add_argument("--linguistic_summary", type=str, default=None)
    parser.add_argument("--predictions", type=str, default=None,
                        help="Predictions JSONL for timeline plots")
    parser.add_argument("--gt", type=str, default=None,
                        help="Ground truth JSONL for timeline plots")
    parser.add_argument("--output_dir", type=str, default="results/figures")
    args = parser.parse_args()

    set_plot_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Publication Figures")
    print("=" * 60)

    # Baseline comparison
    if Path(args.results_csv).exists():
        print("\n--- Baseline Comparison ---")
        plot_baseline_comparison(args.results_csv, output_dir)
        generate_latex_table(args.results_csv, output_dir)

    # Timeline visualization
    if args.predictions and args.gt:
        if Path(args.predictions).exists() and Path(args.gt).exists():
            print("\n--- Prediction Timelines ---")
            plot_prediction_timeline(args.predictions, args.gt, output_dir)

    print("\n" + "=" * 60)
    print("Figure generation complete!")
    print(f"All figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
