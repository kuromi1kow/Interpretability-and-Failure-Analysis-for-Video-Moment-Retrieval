#!/bin/bash -l
# =============================================================================
# VMR Project: Train CG-DETR on QVHighlights
# Submit with: qsub scripts/train_cg_detr.sh
#
# NOTE: CG-DETR may need more VRAM than Moment-DETR.
# If you get OOM errors, reduce BATCH_SIZE to 16.
# =============================================================================

#$ -P cs585
#$ -l gpus=1
#$ -l gpu_type=A100           # A100 preferred for CG-DETR (more VRAM)
#$ -l h_rt=08:00:00
#$ -l mem_per_cpu=16G
#$ -pe omp 4
#$ -N vmr_cg_detr
#$ -o logs/train_cg_detr_$JOB_ID.log
#$ -j y
#$ -m ea

module load miniconda
conda activate vmr
PROJECT_ROOT="${PROJECT_ROOT:-/projectnb/cs585/projects/VMR/vmr_project}"
cd "$PROJECT_ROOT"

echo "Job started: $(date) | Host: $(hostname) | Job ID: $JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MODEL="cg_detr"
FEATURES="clip"
SEED=0
EPOCHS=200
BATCH_SIZE=32    # Reduce to 16 if OOM
LR=1e-4
SAVE_DIR="checkpoints/${MODEL}_${FEATURES}_seed${SEED}_$(date +%Y%m%d)"

mkdir -p "$SAVE_DIR" logs

echo "Training $MODEL | Features: $FEATURES | Seed: $SEED | BS: $BATCH_SIZE"

# Uncomment when ready:
# python -m lighthouse.train \
#     --model $MODEL \
#     --dataset qvhighlights \
#     --features $FEATURES \
#     --epochs $EPOCHS \
#     --batch_size $BATCH_SIZE \
#     --lr $LR \
#     --seed $SEED \
#     --save_dir $SAVE_DIR \
#     --num_workers 4

echo "Training complete: $(date)"
echo "$MODEL,$FEATURES,$SEED,$EPOCHS,$BATCH_SIZE,$LR,$SAVE_DIR,$(date)" \
    >> results/experiment_log.csv
