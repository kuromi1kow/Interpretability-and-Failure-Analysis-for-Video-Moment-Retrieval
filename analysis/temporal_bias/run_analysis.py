#!/usr/bin/env python3
"""Temporal bias and failure analysis for QVHighlights VMR predictions.

Loads ground-truth annotations and per-model prediction files, then produces
bucketed performance tables (by moment length and temporal position) plus
bias-baseline comparisons.

Usage (on SCC):
    cd /projectnb/cs585/projects/VMR/vmr_project
    python analysis/temporal_bias/run_analysis.py
"""

import json
import os
import csv
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path

# ── Absolute SCC paths ──────────────────────────────────────────────────────
PROJECT_ROOT = "/projectnb/cs585/projects/VMR/vmr_project"
ANNO_DIR = os.path.join(PROJECT_ROOT, "data/qvhighlights/annotations")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "lighthouse/results")
OUT_DIR = os.path.join(PROJECT_ROOT, "analysis/temporal_bias/outputs")

GT_VAL = os.path.join(ANNO_DIR, "highlight_val_release.jsonl")
GT_TRAIN = os.path.join(ANNO_DIR, "highlight_train_release.jsonl")

MODELS = {
    "moment_detr": os.path.join(RESULTS_DIR, "moment_detr/qvhighlight/clip_slowfast/best_qvhighlight_val_preds.jsonl"),
    "qd_detr":     os.path.join(RESULTS_DIR, "qd_detr/qvhighlight/clip_slowfast/best_qvhighlight_val_preds.jsonl"),
    "cg_detr":     os.path.join(RESULTS_DIR, "cg_detr/qvhighlight/clip_slowfast/best_qvhighlight_val_preds.jsonl"),
}

METRICS_FILES = {
    "moment_detr": os.path.join(RESULTS_DIR, "moment_detr/qvhighlight/clip_slowfast/best_qvhighlight_val_preds_metrics.json"),
    "qd_detr":     os.path.join(RESULTS_DIR, "qd_detr/qvhighlight/clip_slowfast/best_qvhighlight_val_preds_metrics.json"),
    "cg_detr":     os.path.join(RESULTS_DIR, "cg_detr/qvhighlight/clip_slowfast/best_qvhighlight_val_preds_metrics.json"),
}

# Moment-length buckets (seconds)
LENGTH_BUCKETS = [
    ("short",  0,  10),
    ("medium", 10, 30),
    ("long",   30, 150),
]

# Normalized temporal-position buckets (center of GT window / duration)
POSITION_BUCKETS = [
    ("early",  0.0, 0.33),
    ("middle", 0.33, 0.67),
    ("late",   0.67, 1.01),
]


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def compute_iou(pred_window, gt_window):
    """IoU between two [start, end] intervals."""
    inter_start = max(pred_window[0], gt_window[0])
    inter_end = min(pred_window[1], gt_window[1])
    inter = max(0, inter_end - inter_start)
    union = (pred_window[1] - pred_window[0]) + (gt_window[1] - gt_window[0]) - inter
    return inter / union if union > 0 else 0.0


def best_iou_for_query(pred_windows, gt_windows):
    """For the top-1 predicted window, return its best IoU against any GT window."""
    if not pred_windows or not gt_windows:
        return 0.0
    top1 = pred_windows[0][:2]  # [start, end] (ignore score)
    return max(compute_iou(top1, gw) for gw in gt_windows)


def recall_at_iou(ious, threshold):
    """Fraction of samples with IoU >= threshold."""
    return np.mean([1.0 if iou >= threshold else 0.0 for iou in ious]) * 100


def bucket_label(value, buckets):
    for name, lo, hi in buckets:
        if lo <= value < hi:
            return name
    return buckets[-1][0]


# ── Bias baselines ──────────────────────────────────────────────────────────
def center_baseline_prediction(duration):
    """Predict the center 30% of the video."""
    start = duration * 0.35
    end = duration * 0.65
    return [[start, end]]


def train_distribution_baseline(train_data):
    """Compute mean normalized start/end from training set."""
    norm_starts, norm_ends = [], []
    for item in train_data:
        dur = item["duration"]
        if dur <= 0:
            continue
        for w in item["relevant_windows"]:
            norm_starts.append(w[0] / dur)
            norm_ends.append(w[1] / dur)
    mean_start = np.mean(norm_starts)
    mean_end = np.mean(norm_ends)
    return mean_start, mean_end


# ── Main analysis ───────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load ground truth
    print("Loading ground truth...")
    gt_val = load_jsonl(GT_VAL)
    gt_train = load_jsonl(GT_TRAIN)
    gt_by_qid = {item["qid"]: item for item in gt_val}
    print(f"  Val queries: {len(gt_val)}, Train queries: {len(gt_train)}")

    # ── 1. Overall metrics comparison ───────────────────────────────────
    print("\n" + "="*70)
    print("1. OVERALL METRICS COMPARISON (from saved metric files)")
    print("="*70)
    overall_rows = []
    for model_name, mf in METRICS_FILES.items():
        if not os.path.exists(mf):
            print(f"  WARNING: {mf} not found")
            continue
        with open(mf) as f:
            m = json.load(f)["brief"]
        row = {"model": model_name, **m}
        overall_rows.append(row)
        print(f"  {model_name:15s}  R1@0.5={m['MR-full-R1@0.5']:6.2f}  "
              f"R1@0.7={m['MR-full-R1@0.7']:6.2f}  mAP={m['MR-full-mAP']:6.2f}  "
              f"HL-Fair={m['HL-min-Fair-mAP']:6.2f}  HL-Good={m['HL-min-Good-mAP']:6.2f}  "
              f"HL-VGood={m['HL-min-VeryGood-mAP']:6.2f}")

    # Save overall metrics CSV
    if overall_rows:
        with open(os.path.join(OUT_DIR, "overall_metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=overall_rows[0].keys())
            w.writeheader()
            w.writerows(overall_rows)

    # ── 2. Per-sample IoU analysis ──────────────────────────────────────
    print("\n" + "="*70)
    print("2. PER-SAMPLE IoU AND BUCKETED ANALYSIS")
    print("="*70)

    all_model_results = {}

    for model_name, pred_path in MODELS.items():
        if not os.path.exists(pred_path):
            print(f"  WARNING: {pred_path} not found, skipping {model_name}")
            continue

        preds = load_jsonl(pred_path)
        pred_by_qid = {p["qid"]: p for p in preds}

        samples = []  # list of dicts per query
        for gt in gt_val:
            qid = gt["qid"]
            dur = gt["duration"]
            gt_windows = gt["relevant_windows"]

            pred = pred_by_qid.get(qid)
            if pred is None:
                continue

            pred_windows = pred.get("pred_relevant_windows", [])
            iou = best_iou_for_query(pred_windows, gt_windows)

            # Ground truth properties for bucketing
            # Use the first (primary) GT window for bucketing
            primary_gt = gt_windows[0]
            gt_len = primary_gt[1] - primary_gt[0]
            gt_center = (primary_gt[0] + primary_gt[1]) / 2.0
            norm_center = gt_center / dur if dur > 0 else 0.5

            # Predicted window properties
            if pred_windows:
                top1 = pred_windows[0][:2]
                pred_center = (top1[0] + top1[1]) / 2.0
                norm_pred_center = pred_center / dur if dur > 0 else 0.5
                pred_len = top1[1] - top1[0]
            else:
                pred_center = dur / 2
                norm_pred_center = 0.5
                pred_len = 0

            samples.append({
                "qid": qid,
                "vid": gt["vid"],
                "duration": dur,
                "gt_start": primary_gt[0],
                "gt_end": primary_gt[1],
                "gt_length": gt_len,
                "gt_center": gt_center,
                "norm_gt_center": norm_center,
                "pred_start": pred_windows[0][0] if pred_windows else 0,
                "pred_end": pred_windows[0][1] if pred_windows else 0,
                "pred_length": pred_len,
                "pred_center": pred_center,
                "norm_pred_center": norm_pred_center,
                "iou": iou,
                "length_bucket": bucket_label(gt_len, LENGTH_BUCKETS),
                "position_bucket": bucket_label(norm_center, POSITION_BUCKETS),
            })

        all_model_results[model_name] = samples
        print(f"\n  Model: {model_name} ({len(samples)} samples)")

        # Overall recall
        ious = [s["iou"] for s in samples]
        print(f"    R1@0.3={recall_at_iou(ious, 0.3):.2f}  "
              f"R1@0.5={recall_at_iou(ious, 0.5):.2f}  "
              f"R1@0.7={recall_at_iou(ious, 0.7):.2f}  "
              f"mean_IoU={np.mean(ious):.4f}")

    # ── 3. Length-bucket analysis ───────────────────────────────────────
    print("\n" + "="*70)
    print("3. PERFORMANCE BY MOMENT LENGTH")
    print("="*70)

    length_rows = []
    for model_name, samples in all_model_results.items():
        print(f"\n  {model_name}:")
        for bname, blo, bhi in LENGTH_BUCKETS:
            bucket_samples = [s for s in samples if s["length_bucket"] == bname]
            if not bucket_samples:
                continue
            ious = [s["iou"] for s in bucket_samples]
            r5 = recall_at_iou(ious, 0.5)
            r7 = recall_at_iou(ious, 0.7)
            miou = np.mean(ious)
            mean_pred_len = np.mean([s["pred_length"] for s in bucket_samples])
            mean_gt_len = np.mean([s["gt_length"] for s in bucket_samples])
            print(f"    {bname:8s} (n={len(bucket_samples):4d})  "
                  f"R1@0.5={r5:6.2f}  R1@0.7={r7:6.2f}  "
                  f"mean_IoU={miou:.4f}  "
                  f"mean_gt_len={mean_gt_len:.1f}s  mean_pred_len={mean_pred_len:.1f}s")
            length_rows.append({
                "model": model_name,
                "bucket": bname,
                "range_s": f"{blo}-{bhi}",
                "n_samples": len(bucket_samples),
                "R1@0.5": round(r5, 2),
                "R1@0.7": round(r7, 2),
                "mean_iou": round(miou, 4),
                "mean_gt_length": round(mean_gt_len, 1),
                "mean_pred_length": round(mean_pred_len, 1),
            })

    with open(os.path.join(OUT_DIR, "length_bucket_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=length_rows[0].keys())
        w.writeheader()
        w.writerows(length_rows)

    # ── 4. Temporal-position-bucket analysis ────────────────────────────
    print("\n" + "="*70)
    print("4. PERFORMANCE BY TEMPORAL POSITION (norm gt center / duration)")
    print("="*70)

    position_rows = []
    for model_name, samples in all_model_results.items():
        print(f"\n  {model_name}:")
        for bname, blo, bhi in POSITION_BUCKETS:
            bucket_samples = [s for s in samples if s["position_bucket"] == bname]
            if not bucket_samples:
                continue
            ious = [s["iou"] for s in bucket_samples]
            r5 = recall_at_iou(ious, 0.5)
            r7 = recall_at_iou(ious, 0.7)
            miou = np.mean(ious)
            print(f"    {bname:8s} (n={len(bucket_samples):4d})  "
                  f"R1@0.5={r5:6.2f}  R1@0.7={r7:6.2f}  "
                  f"mean_IoU={miou:.4f}")
            position_rows.append({
                "model": model_name,
                "bucket": bname,
                "range": f"{blo:.2f}-{bhi:.2f}",
                "n_samples": len(bucket_samples),
                "R1@0.5": round(r5, 2),
                "R1@0.7": round(r7, 2),
                "mean_iou": round(miou, 4),
            })

    with open(os.path.join(OUT_DIR, "position_bucket_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=position_rows[0].keys())
        w.writeheader()
        w.writerows(position_rows)

    # ── 5. Prediction center distribution (bias signal) ─────────────────
    print("\n" + "="*70)
    print("5. PREDICTION CENTER DISTRIBUTION (temporal bias signal)")
    print("="*70)

    bias_rows = []
    for model_name, samples in all_model_results.items():
        norm_pred_centers = [s["norm_pred_center"] for s in samples]
        norm_gt_centers = [s["norm_gt_center"] for s in samples]
        pred_mean = np.mean(norm_pred_centers)
        pred_std = np.std(norm_pred_centers)
        gt_mean = np.mean(norm_gt_centers)
        gt_std = np.std(norm_gt_centers)

        # Count predictions in each third
        early_frac = np.mean([1 if c < 0.33 else 0 for c in norm_pred_centers]) * 100
        mid_frac = np.mean([1 if 0.33 <= c < 0.67 else 0 for c in norm_pred_centers]) * 100
        late_frac = np.mean([1 if c >= 0.67 else 0 for c in norm_pred_centers]) * 100

        print(f"  {model_name}:")
        print(f"    pred_center: mean={pred_mean:.3f} std={pred_std:.3f}")
        print(f"    gt_center:   mean={gt_mean:.3f} std={gt_std:.3f}")
        print(f"    pred distribution: early={early_frac:.1f}%  middle={mid_frac:.1f}%  late={late_frac:.1f}%")

        bias_rows.append({
            "model": model_name,
            "pred_center_mean": round(pred_mean, 4),
            "pred_center_std": round(pred_std, 4),
            "gt_center_mean": round(gt_mean, 4),
            "gt_center_std": round(gt_std, 4),
            "pct_early": round(early_frac, 1),
            "pct_middle": round(mid_frac, 1),
            "pct_late": round(late_frac, 1),
        })

    with open(os.path.join(OUT_DIR, "prediction_center_distribution.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bias_rows[0].keys())
        w.writeheader()
        w.writerows(bias_rows)

    # ── 6. Bias baselines ───────────────────────────────────────────────
    print("\n" + "="*70)
    print("6. BIAS BASELINES")
    print("="*70)

    # 6a. Center-of-video baseline
    print("\n  [A] Center-of-video baseline (predict center 30% of video):")
    center_ious = []
    for gt in gt_val:
        dur = gt["duration"]
        gt_windows = gt["relevant_windows"]
        pred = center_baseline_prediction(dur)
        iou = best_iou_for_query([[p[0], p[1], 1.0] for p in pred], gt_windows)
        center_ious.append(iou)
    print(f"      R1@0.3={recall_at_iou(center_ious, 0.3):.2f}  "
          f"R1@0.5={recall_at_iou(center_ious, 0.5):.2f}  "
          f"R1@0.7={recall_at_iou(center_ious, 0.7):.2f}  "
          f"mean_IoU={np.mean(center_ious):.4f}")

    # 6b. Training distribution baseline
    print("\n  [B] Training-distribution baseline (predict mean normalized start/end):")
    mean_start, mean_end = train_distribution_baseline(gt_train)
    print(f"      Train mean normalized window: [{mean_start:.3f}, {mean_end:.3f}]")
    train_dist_ious = []
    for gt in gt_val:
        dur = gt["duration"]
        gt_windows = gt["relevant_windows"]
        pred_start = mean_start * dur
        pred_end = mean_end * dur
        pred = [[pred_start, pred_end, 1.0]]
        iou = best_iou_for_query(pred, gt_windows)
        train_dist_ious.append(iou)
    print(f"      R1@0.3={recall_at_iou(train_dist_ious, 0.3):.2f}  "
          f"R1@0.5={recall_at_iou(train_dist_ious, 0.5):.2f}  "
          f"R1@0.7={recall_at_iou(train_dist_ious, 0.7):.2f}  "
          f"mean_IoU={np.mean(train_dist_ious):.4f}")

    # 6c. Random-uniform baseline
    print("\n  [C] Random-uniform baseline (random start+end in [0, duration], 100 trials):")
    rng = np.random.RandomState(42)
    random_r5_trials = []
    for trial in range(100):
        trial_ious = []
        for gt in gt_val:
            dur = gt["duration"]
            gt_windows = gt["relevant_windows"]
            a, b = sorted(rng.uniform(0, dur, 2))
            pred = [[a, b, 1.0]]
            iou = best_iou_for_query(pred, gt_windows)
            trial_ious.append(iou)
        random_r5_trials.append(recall_at_iou(trial_ious, 0.5))
    print(f"      R1@0.5 mean={np.mean(random_r5_trials):.2f}  std={np.std(random_r5_trials):.2f}")

    # Save bias baseline summary
    baseline_summary = {
        "center_baseline": {
            "description": "predict center 30% of video [0.35*dur, 0.65*dur]",
            "R1@0.3": round(recall_at_iou(center_ious, 0.3), 2),
            "R1@0.5": round(recall_at_iou(center_ious, 0.5), 2),
            "R1@0.7": round(recall_at_iou(center_ious, 0.7), 2),
            "mean_iou": round(float(np.mean(center_ious)), 4),
        },
        "train_distribution_baseline": {
            "description": f"predict mean train window [{mean_start:.3f}, {mean_end:.3f}] * duration",
            "R1@0.3": round(recall_at_iou(train_dist_ious, 0.3), 2),
            "R1@0.5": round(recall_at_iou(train_dist_ious, 0.5), 2),
            "R1@0.7": round(recall_at_iou(train_dist_ious, 0.7), 2),
            "mean_iou": round(float(np.mean(train_dist_ious)), 4),
        },
        "random_baseline": {
            "description": "random [a,b] in [0, duration], 100 trials",
            "R1@0.5_mean": round(float(np.mean(random_r5_trials)), 2),
            "R1@0.5_std": round(float(np.std(random_r5_trials)), 2),
        },
    }
    with open(os.path.join(OUT_DIR, "bias_baselines.json"), "w") as f:
        json.dump(baseline_summary, f, indent=2)

    # ── 7. Per-sample output (for downstream analysis/plotting) ─────────
    for model_name, samples in all_model_results.items():
        outpath = os.path.join(OUT_DIR, f"{model_name}_per_sample.csv")
        with open(outpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=samples[0].keys())
            w.writeheader()
            w.writerows(samples)
        print(f"\n  Saved per-sample CSV: {outpath}")

    # ── 8. Failure case examples (lowest IoU for QD-DETR) ───────────────
    print("\n" + "="*70)
    print("7. FAILURE CASES (QD-DETR, 10 worst by IoU)")
    print("="*70)
    if "qd_detr" in all_model_results:
        qd_samples = sorted(all_model_results["qd_detr"], key=lambda s: s["iou"])
        failure_cases = []
        for s in qd_samples[:10]:
            gt = gt_by_qid.get(s["qid"], {})
            print(f"  qid={s['qid']}  IoU={s['iou']:.3f}  "
                  f"GT=[{s['gt_start']:.0f},{s['gt_end']:.0f}]  "
                  f"Pred=[{s['pred_start']:.0f},{s['pred_end']:.0f}]  "
                  f"dur={s['duration']}  "
                  f"query=\"{gt.get('query','?')[:60]}\"")
            failure_cases.append({
                "qid": s["qid"],
                "iou": round(s["iou"], 4),
                "gt_window": [s["gt_start"], s["gt_end"]],
                "pred_window": [s["pred_start"], s["pred_end"]],
                "duration": s["duration"],
                "query": gt.get("query", ""),
            })
        with open(os.path.join(OUT_DIR, "qd_detr_failure_cases.json"), "w") as f:
            json.dump(failure_cases, f, indent=2)

    # ── 9. Summary JSON ────────────────────────────────────────────────
    summary = {
        "models_analyzed": list(all_model_results.keys()),
        "val_queries": len(gt_val),
        "overall_metrics": overall_rows,
        "length_buckets": length_rows,
        "position_buckets": position_rows,
        "bias_signal": bias_rows,
        "bias_baselines": baseline_summary,
    }
    with open(os.path.join(OUT_DIR, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Full summary saved to: {os.path.join(OUT_DIR, 'analysis_summary.json')}")

    print("\n" + "="*70)
    print("DONE. All outputs in:", OUT_DIR)
    print("="*70)


if __name__ == "__main__":
    main()
