#!/bin/bash -l
# =============================================================================
# VMR Project: Evaluate all models on QVHighlights val
# Submit with: qsub scripts/eval_all.sh
# =============================================================================

#$ -P <YOUR_PROJECT>          # ← CHANGE THIS
#$ -l gpus=1
#$ -l gpu_type=V100|A100
#$ -l h_rt=02:00:00           # Inference is fast (~10 min per model)
#$ -l mem_per_cpu=16G
#$ -pe omp 4
#$ -N vmr_eval_all
#$ -o logs/eval_all_$JOB_ID.log
#$ -j y

module load miniconda
conda activate vmr
cd /projectnb/<YOUR_PROJECT>/vmr    # ← CHANGE THIS

echo "Evaluation started: $(date)"

GT_FILE="data/qvhighlights/annotations/highlight_val_release.jsonl"
OUTPUT_CSV="results/eval_results.csv"

# Initialize CSV
echo "model,R1@0.3,R1@0.5,R1@0.7,mAP,HIT@1" > "$OUTPUT_CSV"

# Evaluate each model
# Uncomment and update checkpoint paths once training is done:

# echo "--- Evaluating Moment-DETR ---"
# python src/eval/evaluate.py \
#     --pred checkpoints/moment_detr_*/predictions.jsonl \
#     --gt $GT_FILE \
#     --model_name "Moment-DETR" \
#     --output $OUTPUT_CSV

# echo "--- Evaluating QD-DETR ---"
# python src/eval/evaluate.py \
#     --pred checkpoints/qd_detr_*/predictions.jsonl \
#     --gt $GT_FILE \
#     --model_name "QD-DETR" \
#     --output $OUTPUT_CSV

# echo "--- Evaluating CG-DETR ---"
# python src/eval/evaluate.py \
#     --pred checkpoints/cg_detr_*/predictions.jsonl \
#     --gt $GT_FILE \
#     --model_name "CG-DETR" \
#     --output $OUTPUT_CSV

echo ""
echo "Results saved to: $OUTPUT_CSV"
cat "$OUTPUT_CSV"
echo ""
echo "Evaluation complete: $(date)"
