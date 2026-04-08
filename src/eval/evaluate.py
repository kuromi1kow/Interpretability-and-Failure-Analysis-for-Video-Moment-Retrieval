#!/usr/bin/env python3
"""
VMR Project: Unified Evaluation Script
Computes R1@IoU={0.3, 0.5, 0.7}, mAP, and HIT@1 for QVHighlights.

This script implements the standard QVHighlights evaluation metrics.
It can be used standalone or as a module imported by other scripts.

Usage:
    python src/eval/evaluate.py --pred predictions.jsonl --gt val.jsonl
    python src/eval/evaluate.py --pred_dir checkpoints/moment_detr/ --gt data/qvhighlights/annotations/highlight_val_release.jsonl
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


# =============================================================================
# Core Metric Functions
# =============================================================================

def compute_iou(pred, gt):
    """Compute IoU between two temporal intervals [start, end]."""
    intersection = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return intersection / union if union > 0 else 0.0


def compute_recall_at_iou(predictions, ground_truths, iou_thresholds=[0.3, 0.5, 0.7]):
    """
    Compute Recall@1 at various IoU thresholds.

    Args:
        predictions: dict {qid: [start, end]} — top-1 predicted moment
        ground_truths: dict {qid: [[start, end], ...]} — GT moments
        iou_thresholds: list of IoU thresholds

    Returns:
        dict {threshold: recall_value}
    """
    results = {}
    for thresh in iou_thresholds:
        correct = 0
        total = 0
        for qid, pred_moment in predictions.items():
            if qid not in ground_truths:
                continue
            gt_moments = ground_truths[qid]
            # Check if prediction overlaps with ANY ground-truth moment above threshold
            max_iou = max(compute_iou(pred_moment, gt) for gt in gt_moments)
            if max_iou >= thresh:
                correct += 1
            total += 1
        results[f"R1@{thresh}"] = (correct / total * 100) if total > 0 else 0.0
    return results


def compute_highlight_metrics(pred_saliency, gt_saliency):
    """
    Compute mAP and HIT@1 for highlight detection.

    Args:
        pred_saliency: dict {qid: [score_per_clip]}
        gt_saliency: dict {qid: [score_per_clip]}

    Returns:
        dict with mAP and HIT@1
    """
    aps = []
    hits = []

    for qid in pred_saliency:
        if qid not in gt_saliency:
            continue

        pred_scores = np.array(pred_saliency[qid])
        gt_scores = np.array(gt_saliency[qid])

        # For mAP: threshold GT at "Very Good" (typically score >= 3 on 1-5 scale)
        gt_binary = (gt_scores >= np.percentile(gt_scores, 75)).astype(int)

        if gt_binary.sum() == 0:
            continue

        # Average Precision
        sorted_indices = np.argsort(-pred_scores)
        sorted_gt = gt_binary[sorted_indices]
        tp_cumsum = np.cumsum(sorted_gt)
        precision_at_k = tp_cumsum / np.arange(1, len(sorted_gt) + 1)
        ap = np.sum(precision_at_k * sorted_gt) / gt_binary.sum()
        aps.append(ap)

        # HIT@1: is the highest-predicted clip a true highlight?
        top1_idx = np.argmax(pred_scores)
        hits.append(int(gt_binary[top1_idx]))

    return {
        "mAP": np.mean(aps) * 100 if aps else 0.0,
        "HIT@1": np.mean(hits) * 100 if hits else 0.0,
    }


# =============================================================================
# I/O and Formatting
# =============================================================================

def load_predictions(path):
    """Load model predictions from JSONL file."""
    predictions = {}
    saliency_predictions = {}

    with open(path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            qid = item.get("qid", item.get("query_id"))

            # Top-1 predicted moment
            if "pred_relevant_windows" in item:
                predictions[qid] = item["pred_relevant_windows"][0]  # take top-1
            elif "predicted_times" in item:
                predictions[qid] = item["predicted_times"][0]

            # Predicted saliency scores
            if "pred_saliency_scores" in item:
                saliency_predictions[qid] = item["pred_saliency_scores"]

    return predictions, saliency_predictions


def load_ground_truth(path):
    """Load ground truth from QVHighlights annotation file."""
    ground_truths = {}
    gt_saliency = {}

    with open(path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            qid = item.get("qid", item.get("query_id"))

            # Ground-truth moments
            if "relevant_windows" in item:
                ground_truths[qid] = item["relevant_windows"]

            # Ground-truth saliency
            if "saliency_scores" in item:
                # QVHighlights has per-clip saliency from multiple annotators
                # Average across annotators
                scores = item["saliency_scores"]
                if isinstance(scores[0], list):
                    gt_saliency[qid] = [np.mean(s) for s in scores]
                else:
                    gt_saliency[qid] = scores

    return ground_truths, gt_saliency


def print_results(model_name, recall_results, highlight_results=None):
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<15} {'Value':>8}")
    print(f"  {'-' * 25}")
    for metric, value in recall_results.items():
        print(f"  {metric:<15} {value:>8.2f}")
    if highlight_results:
        for metric, value in highlight_results.items():
            print(f"  {metric:<15} {value:>8.2f}")
    print(f"{'=' * 60}")


def results_to_csv_row(model_name, recall_results, highlight_results=None):
    """Format results as a CSV row."""
    row = [model_name]
    for thresh in [0.3, 0.5, 0.7]:
        key = f"R1@{thresh}"
        row.append(f"{recall_results.get(key, 0):.2f}")
    if highlight_results:
        row.append(f"{highlight_results.get('mAP', 0):.2f}")
        row.append(f"{highlight_results.get('HIT@1', 0):.2f}")
    else:
        row.extend(["—", "—"])
    return ",".join(row)


def results_to_latex_row(model_name, recall_results, highlight_results=None):
    """Format results as a LaTeX table row."""
    values = [model_name]
    for thresh in [0.3, 0.5, 0.7]:
        key = f"R1@{thresh}"
        values.append(f"{recall_results.get(key, 0):.1f}")
    if highlight_results:
        values.append(f"{highlight_results.get('mAP', 0):.1f}")
        values.append(f"{highlight_results.get('HIT@1', 0):.1f}")
    else:
        values.extend(["—", "—"])
    return " & ".join(values) + " \\\\"


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="QVHighlights Evaluation")
    parser.add_argument("--pred", type=str, help="Path to predictions JSONL file")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth JSONL file")
    parser.add_argument("--model_name", type=str, default="model", help="Model name for display")
    parser.add_argument("--output", type=str, default=None, help="Save results to this CSV file")
    args = parser.parse_args()

    # Load data
    ground_truths, gt_saliency = load_ground_truth(args.gt)
    predictions, pred_saliency = load_predictions(args.pred)

    print(f"Loaded {len(predictions)} predictions, {len(ground_truths)} ground truths")

    # Compute metrics
    recall_results = compute_recall_at_iou(predictions, ground_truths)

    highlight_results = None
    if pred_saliency and gt_saliency:
        highlight_results = compute_highlight_metrics(pred_saliency, gt_saliency)

    # Print
    print_results(args.model_name, recall_results, highlight_results)

    # LaTeX row
    print(f"\nLaTeX row:")
    print(f"  {results_to_latex_row(args.model_name, recall_results, highlight_results)}")

    # Save
    if args.output:
        header = "model,R1@0.3,R1@0.5,R1@0.7,mAP,HIT@1\n"
        row = results_to_csv_row(args.model_name, recall_results, highlight_results) + "\n"
        mode = "a" if Path(args.output).exists() else "w"
        with open(args.output, mode) as f:
            if mode == "w":
                f.write(header)
            f.write(row)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
