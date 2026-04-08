# Temporal Bias and Failure Analysis — QVHighlights

**Post-Milestone-2 analysis step for CS585 VMR project.**

## What This Does

1. Compares overall MR metrics across Moment-DETR, QD-DETR, CG-DETR (shared baseline predictions)
2. Buckets performance by **moment length** (short / medium / long)
3. Buckets performance by **normalized temporal position** (early / middle / late)
4. Measures where prediction centers tend to fall — detects temporal bias
5. Computes **bias baselines**: center-of-video, training-distribution mean, random
6. Identifies QD-DETR failure cases (worst IoU samples)
7. Generates plots (if matplotlib available)

## SCC Commands

```bash
# 1. Run the main analysis (no GPU needed, ~10s)
cd /projectnb/cs585/projects/VMR/vmr_project
python analysis/temporal_bias/run_analysis.py

# 2. Generate plots (needs matplotlib)
python analysis/temporal_bias/plot_results.py

# 3. If matplotlib not installed:
pip install matplotlib
python analysis/temporal_bias/plot_results.py
```

## Input Files Used (read-only, not modified)

- `/projectnb/cs585/projects/VMR/vmr_project/data/qvhighlights/annotations/highlight_val_release.jsonl`
- `/projectnb/cs585/projects/VMR/vmr_project/data/qvhighlights/annotations/highlight_train_release.jsonl`
- `/projectnb/cs585/projects/VMR/vmr_project/lighthouse/results/{moment_detr,qd_detr,cg_detr}/qvhighlight/clip_slowfast/best_qvhighlight_val_preds.jsonl`
- `/projectnb/cs585/projects/VMR/vmr_project/lighthouse/results/{moment_detr,qd_detr,cg_detr}/qvhighlight/clip_slowfast/best_qvhighlight_val_preds_metrics.json`

## Output Files

All outputs go to `analysis/temporal_bias/outputs/`:

| File | Format | Description |
|------|--------|-------------|
| `overall_metrics.csv` | CSV | Overall MR and HL metrics per model |
| `length_bucket_metrics.csv` | CSV | R1@0.5/0.7/mean_IoU by moment length |
| `position_bucket_metrics.csv` | CSV | R1@0.5/0.7/mean_IoU by temporal position |
| `prediction_center_distribution.csv` | CSV | Where each model's predictions cluster |
| `bias_baselines.json` | JSON | Center, train-distribution, random baselines |
| `{model}_per_sample.csv` | CSV | Per-query IoU + metadata for each model |
| `qd_detr_failure_cases.json` | JSON | 10 worst QD-DETR predictions |
| `analysis_summary.json` | JSON | Full machine-readable summary |
| `*.png` | PNG | Plots (from plot_results.py) |

## Notes

- These scripts are **read-only** with respect to model predictions and data.
- No training, no GPU, no data download needed.
- The "best" prediction files correspond to the **shared baseline** results (Moment-DETR 54.06, QD-DETR 62.06, CG-DETR 62.84 R1@0.5).
- The HDF5-backed run predictions (47.35/61.29/67.35) are not stored as separate `.jsonl` files on SCC. Only their aggregate metrics were recorded.
