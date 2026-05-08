#!/usr/bin/env python3
"""Generate presentation-ready figures from saved experiment outputs.

This script avoids GPU/SCC dependencies by reading the analysis CSV/JSON files
already stored in the repo and producing clean PNGs for the slide deck.
"""

from __future__ import annotations

import csv
import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "docs" / "figures"
TEMPORAL_DIR = PROJECT_ROOT / "analysis" / "temporal_bias" / "outputs"
VERB_NOUN_DIR = PROJECT_ROOT / "analysis" / "verb_noun_ablation" / "outputs"
HEAD_DIR = PROJECT_ROOT / "analysis" / "head_ablation" / "outputs"
ANNOTATION_PATH = PROJECT_ROOT / "data" / "qvhighlights" / "annotations" / "highlight_val_release.jsonl"


COLORS = {
    "moment_detr": "#355C7D",
    "qd_detr": "#2A9D8F",
    "cg_detr": "#E76F51",
    "original": "#355C7D",
    "verb_masked": "#E9C46A",
    "noun_masked": "#E76F51",
    "verb": "#C06C2B",
    "noun": "#2A9D8F",
    "gt": "#355C7D",
    "pred": "#E76F51",
}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> dict | list:
    with path.open() as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def prettify_model_name(name: str) -> str:
    return name.replace("_", "-").upper()


def save_figure(fig: plt.Figure, filename: str) -> None:
    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")


def load_annotations_by_qid() -> dict[str, dict]:
    return {str(item["qid"]): item for item in load_jsonl(ANNOTATION_PATH)}


def build_saliency_track(item: dict) -> np.ndarray:
    duration = float(item["duration"])
    n_clips = max(int(math.ceil(duration / 2.0)), max(item["relevant_clip_ids"], default=-1) + 1)
    values = np.zeros(n_clips)
    counts = np.zeros(n_clips)

    for clip_id, scores in zip(item["relevant_clip_ids"], item["saliency_scores"]):
        values[clip_id] += float(np.mean(scores))
        counts[clip_id] += 1.0

    nonzero = counts > 0
    values[nonzero] /= counts[nonzero]
    return values


def plot_vmr_query_to_output_examples() -> None:
    annotations = load_annotations_by_qid()
    rows = {row["qid"]: row for row in load_csv(TEMPORAL_DIR / "qd_detr_per_sample.csv")}
    qid = "601"

    saliency_cmap = mcolors.LinearSegmentedColormap.from_list(
        "saliency", ["#F8F6F1", "#CFE4EE", "#5BA4BF", "#1F5E74"]
    )

    item = annotations[qid]
    row = rows[qid]
    saliency = build_saliency_track(item)
    duration = float(row["duration"])
    gt_start = float(row["gt_start"])
    gt_end = float(row["gt_end"])
    pred_start = float(row["pred_start"])
    pred_end = float(row["pred_end"])
    iou = float(row["iou"])

    fig, ax = plt.subplots(1, 1, figsize=(11.4, 3.8))
    ax.imshow(
        saliency[np.newaxis, :],
        extent=[0, duration, 2.15, 2.48],
        cmap=saliency_cmap,
        aspect="auto",
        vmin=0,
        vmax=4,
    )
    ax.barh(1.35, gt_end - gt_start, left=gt_start, color=COLORS["gt"], height=0.34)
    ax.barh(0.60, pred_end - pred_start, left=pred_start, color=COLORS["pred"], height=0.34)

    ax.set_xlim(0, duration)
    ax.set_ylim(0.1, 2.75)
    ax.set_yticks([0.60, 1.35, 2.31])
    ax.set_yticklabels(["QD-DETR pred", "Ground truth", "Human saliency"], fontsize=11)
    ax.text(
        duration,
        1.80,
        f"IoU = {iou:.2f}   |   GT [{gt_start:.0f}, {gt_end:.0f}]s",
        ha="right",
        va="center",
        fontsize=11,
        color=COLORS["gt"],
    )
    ax.text(
        duration,
        1.00,
        f"Pred [{pred_start:.0f}, {pred_end:.0f}]s",
        ha="right",
        va="center",
        fontsize=11,
        color=COLORS["pred"],
    )
    ax.grid(axis="x", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks(np.arange(0, 151, 30))
    ax.set_xlabel("Time in 150-second video", fontsize=11)
    fig.text(
        0.02,
        0.01,
        "The blue strip shows human highlight saliency over 2-second clips. The model output itself is the predicted [start, end] time window.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    save_figure(fig, "vmr_query_to_output_examples.png")


def plot_temporal_bias_length_buckets() -> None:
    rows = load_csv(TEMPORAL_DIR / "length_bucket_metrics.csv")
    bucket_order = ["short", "medium", "long"]
    bucket_labels = ["Short\n(0-10s)", "Medium\n(10-30s)", "Long\n(30-150s)"]
    models = []
    for row in rows:
        if row["model"] not in models:
            models.append(row["model"])

    x = np.arange(len(bucket_order))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    for idx, model in enumerate(models):
        values = []
        for bucket in bucket_order:
            match = next(r for r in rows if r["model"] == model and r["bucket"] == bucket)
            values.append(float(match["R1@0.5"]))
        offset = (idx - (len(models) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=prettify_model_name(model),
            color=COLORS[model],
            edgecolor="white",
            linewidth=1.0,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.8,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title("Temporal Bias: Short Moments Are the Main Failure Mode", fontsize=15, weight="bold")
    ax.set_ylabel("R1@0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylim(0, 75)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.04))

    short_qd = next(r for r in rows if r["model"] == "qd_detr" and r["bucket"] == "short")
    ax.text(
        0.02,
        0.04,
        (
            "QD-DETR short moments: GT length 4.6s vs predicted 32.0s\n"
            "The models localize the right region less precisely when the event is brief."
        ),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F8F7F4", "edgecolor": "#D9D5CB"},
    )

    fig.tight_layout()
    save_figure(fig, "temporal_bias_length_buckets.png")


def plot_temporal_bias_failure_examples() -> None:
    cases = load_json(TEMPORAL_DIR / "qd_detr_failure_cases.json")
    selected = [cases[1], cases[3], cases[5], cases[9]]

    fig, axes = plt.subplots(len(selected), 1, figsize=(10.5, 7.2), sharex=False)
    fig.suptitle("Temporal Bias: Example QD-DETR Failure Cases", fontsize=15, weight="bold", y=0.99)

    for ax, case in zip(axes, selected):
        duration = case["duration"]
        gt_start, gt_end = case["gt_window"]
        pred_start, pred_end = case["pred_window"]
        query = textwrap.fill(case["query"], width=72)

        ax.barh(1, gt_end - gt_start, left=gt_start, color=COLORS["gt"], height=0.34, label="GT")
        ax.barh(0, pred_end - pred_start, left=pred_start, color=COLORS["pred"], height=0.34, label="Pred")

        ax.set_xlim(0, duration)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Pred", "GT"])
        ax.set_title(query, fontsize=10, loc="left", pad=6)
        ax.grid(axis="x", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.text(duration, 1.17, f"IoU = {case['iou']:.2f}", ha="right", va="bottom", fontsize=9, color="#444444")

    axes[-1].set_xlabel("Time (seconds)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["gt"]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["pred"]),
    ]
    fig.legend(handles, ["Ground Truth", "Prediction"], loc="upper right", frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, "temporal_bias_failure_examples.png")


def plot_verb_noun_metric_drop() -> None:
    comparison = load_json(VERB_NOUN_DIR / "ablation_comparison.json")
    conditions = ["original", "verb_masked", "noun_masked"]
    metrics = ["MR-full-R1@0.5", "MR-full-R1@0.7", "MR-full-mAP"]
    metric_labels = ["R1@0.5", "R1@0.7", "mAP"]

    x = np.arange(len(metric_labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    for idx, condition in enumerate(conditions):
        values = [comparison[condition][metric] for metric in metrics]
        offset = (idx - (len(conditions) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=condition.replace("_", " ").title(),
            color=COLORS[condition],
            edgecolor="white",
            linewidth=1.0,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.5,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    verb_drop = comparison["original"]["MR-full-R1@0.5"] - comparison["verb_masked"]["MR-full-R1@0.5"]
    noun_drop = comparison["original"]["MR-full-R1@0.5"] - comparison["noun_masked"]["MR-full-R1@0.5"]
    ratio = noun_drop / verb_drop if verb_drop else float("inf")

    ax.set_title("Verb vs Noun Sensitivity: Nouns Matter More Than Verbs", fontsize=15, weight="bold")
    ax.set_ylabel("Validation Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(34, 66)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    ax.text(
        0.02,
        0.04,
        f"Noun masking drop at R1@0.5: -{noun_drop:.2f}\nVerb masking drop at R1@0.5: -{verb_drop:.2f}\nNouns are {ratio:.1f}x more important in this probe.",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F8F7F4", "edgecolor": "#D9D5CB"},
    )

    fig.tight_layout()
    save_figure(fig, "verb_noun_metric_drop.png")


def choose_query_examples(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    picked = []
    for row in rows:
        if row["split"] != "val":
            continue
        if int(row["n_verbs"]) < 1 or int(row["n_nouns"]) < 2:
            continue
        if int(row["n_words"]) > 14:
            continue
        picked.append(row)
        if len(picked) == 2:
            break
    return picked


def plot_verb_noun_query_examples() -> None:
    rows = load_csv(VERB_NOUN_DIR / "query_composition.csv")
    selected = choose_query_examples(rows)
    fig, axes = plt.subplots(len(selected), 1, figsize=(11.0, 4.8))
    fig.suptitle("Verb vs Noun Sensitivity: Query Examples", fontsize=15, weight="bold", y=0.99)

    for ax, row in zip(axes, selected):
        ax.axis("off")
        query = textwrap.fill(row["query"], width=72)
        verbs = row["verbs"].replace("|", ", ")
        nouns = row["nouns"].replace("|", ", ")

        ax.text(0.00, 0.86, "Query", fontsize=11, weight="bold", transform=ax.transAxes)
        ax.text(0.00, 0.58, query, fontsize=12, transform=ax.transAxes)
        ax.text(0.00, 0.25, "Verb tokens", fontsize=10, color=COLORS["verb"], weight="bold", transform=ax.transAxes)
        ax.text(0.18, 0.25, verbs, fontsize=10, color=COLORS["verb"], transform=ax.transAxes)
        ax.text(0.00, 0.05, "Noun tokens", fontsize=10, color=COLORS["noun"], weight="bold", transform=ax.transAxes)
        ax.text(0.18, 0.05, nouns, fontsize=10, color=COLORS["noun"], transform=ax.transAxes)
        ax.add_patch(
            plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor="#D9D5CB", linewidth=1.0)
        )

    fig.text(
        0.02,
        0.01,
        "These examples make the ablation easy to explain: the model often has several object/entity cues, and removing noun information hurts more than removing the action words.",
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_figure(fig, "verb_noun_query_examples.png")


def shorten_module_name(name: str) -> str:
    if "t2v_encoder.layers.0" in name:
        return "T2V L0"
    if "t2v_encoder.layers.1" in name:
        return "T2V L1"
    if "encoder.layers.0" in name:
        return "ENC L0"
    if "encoder.layers.1" in name:
        return "ENC L1"
    return name


def plot_head_ablation_heatmap() -> None:
    data = load_json(HEAD_DIR / "head_ablation_results.json")
    modules = [m["name"] for m in data["modules"]]
    module_labels = [shorten_module_name(m) for m in modules]
    heads = list(range(8))
    matrix = np.zeros((len(modules), len(heads)))

    lookup = {(row["module"], row["head"]): row["delta_R1@0.5"] for row in data["ablation_results"]}
    for i, module in enumerate(modules):
        for j, head in enumerate(heads):
            matrix[i, j] = lookup[(module, head)]

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    norm = mcolors.TwoSlopeNorm(vmin=min(-0.8, matrix.min()), vcenter=0.0, vmax=max(1.0, matrix.max()))
    im = ax.imshow(matrix, cmap="RdBu", norm=norm, aspect="auto")

    ax.set_title("Head Ablation: No Single Head Causes a Large Collapse", fontsize=15, weight="bold")
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Module")
    ax.set_xticks(range(len(heads)))
    ax.set_xticklabels([f"H{h}" for h in heads])
    ax.set_yticks(range(len(module_labels)))
    ax.set_yticklabels(module_labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("delta R1@0.5")
    fig.tight_layout()
    save_figure(fig, "head_ablation_heatmap.png")


def plot_head_ablation_top_heads() -> None:
    data = load_json(HEAD_DIR / "head_ablation_results.json")
    baseline = data["baseline"]["MR-full-R1@0.5"]
    worst = sorted(data["ablation_results"], key=lambda row: row["delta_R1@0.5"])[:5]

    labels = [f"{shorten_module_name(row['module'])}.H{row['head']}" for row in worst]
    values = [row["delta_R1@0.5"] for row in worst]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    y = np.arange(len(labels))
    ax.barh(y, values, color="#D95F5F", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("delta R1@0.5 after ablating one head")
    ax.set_title("Most Impactful Heads: Even the Worst Drop Is Small", fontsize=15, weight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(0, color="#555555", linewidth=1.0)

    for yi, value in zip(y, values):
        ax.text(value - 0.02, yi, f"{value:+.2f}", ha="right", va="center", fontsize=10, color="white")

    ax.text(
        0.98,
        0.04,
        f"Baseline QD-DETR R1@0.5 = {baseline:.2f}\nWorst single-head drop = {min(values):+.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F8F7F4", "edgecolor": "#D9D5CB"},
    )

    fig.tight_layout()
    save_figure(fig, "head_ablation_top_heads.png")


def main() -> None:
    ensure_out_dir()
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10.5,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    plot_vmr_query_to_output_examples()
    plot_temporal_bias_length_buckets()
    plot_temporal_bias_failure_examples()
    plot_verb_noun_metric_drop()
    plot_verb_noun_query_examples()
    plot_head_ablation_heatmap()
    plot_head_ablation_top_heads()


if __name__ == "__main__":
    main()
