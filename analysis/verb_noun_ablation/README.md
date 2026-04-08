# Verb/Noun Sensitivity Ablation — QVHighlights

Probes whether QD-DETR relies more on action verbs or object nouns for moment retrieval.

## Method

1. **spaCy POS tagging** identifies verbs and nouns in each query
2. **CLIP tokenizer alignment** maps words to subword token positions in the pre-extracted features
3. **Feature zeroing**: verb tokens or noun tokens are set to zero in `last_hidden_state`
4. **QD-DETR eval** runs with original, verb-masked, and noun-masked text features

## SCC Commands

```bash
# Step 1: Prepare masked features (CPU, already done)
source /projectnb/cs585/projects/VMR/vmr_env/bin/activate
cd /projectnb/cs585/projects/VMR/vmr_project
python analysis/verb_noun_ablation/prepare_ablation.py

# Step 2: Submit GPU eval job
qsub analysis/verb_noun_ablation/run_ablation.sh

# Step 3: Check results
cat analysis/verb_noun_ablation/outputs/ablation_comparison.json
```

## Outputs

- `outputs/query_composition.csv` — POS analysis for all queries
- `outputs/ablation_prep_summary.json` — preparation stats
- `outputs/ablation_comparison.json` — metrics comparison (after GPU eval)
- `outputs/eval_*/` — per-condition predictions and metrics
