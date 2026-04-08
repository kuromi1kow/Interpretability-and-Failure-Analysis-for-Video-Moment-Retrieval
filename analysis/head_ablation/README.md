# Head Ablation Study — QD-DETR on QVHighlights

Identifies which attention heads are most important for moment boundary prediction.

## Method

1. Run baseline eval (no ablation)
2. For each multi-head attention module and each head: zero out that head's output projection
3. Measure R1@0.5 drop — heads with largest drops are "boundary heads"

## SCC Commands

```bash
# Submit GPU job (6 hours, may need more time depending on head count)
qsub /projectnb/cs585/projects/VMR/vmr_project/analysis/head_ablation/run_head_ablation.sh

# Check results
cat /projectnb/cs585/projects/VMR/vmr_project/analysis/head_ablation/outputs/head_ablation_results.json
```

## Outputs

- `outputs/head_ablation_results.json` — per-head R1@0.5 delta
