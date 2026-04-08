#!/bin/bash -l
# =============================================================================
# VMR Project: Train QD-DETR on QVHighlights
# Submit with: qsub scripts/train_qd_detr.sh
#
# Default behavior is intentionally non-destructive: it finetunes from the
# existing best QD-DETR checkpoint so we do not wipe the original baseline run.
# =============================================================================

#$ -P cs585
#$ -l gpus=1
#$ -l gpu_type=A100|A100-80G
#$ -l h_rt=08:00:00
#$ -l mem_per_core=16G
#$ -l h_vmem=16G
#$ -pe omp 8
#$ -N vmr_qd_detr
#$ -o /projectnb/cs585/projects/VMR/vmr_project/logs/train_qd_detr_$JOB_ID.log
#$ -j y
#$ -m ea

set -euo pipefail

module load miniconda
conda activate vmr
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PROJECT_ROOT="${PROJECT_ROOT:-/projectnb/cs585/projects/VMR/vmr_project}"
LIGHTHOUSE_ROOT="$PROJECT_ROOT/lighthouse"
BASE_CKPT="${BASE_CKPT:-$LIGHTHOUSE_ROOT/results/qd_detr/qvhighlight/clip_slowfast/best.ckpt}"
LIGHTHOUSE_RESULTS_ROOT="${LIGHTHOUSE_RESULTS_ROOT:-/projectnb/cs505am/students/${USER}/lighthouse_results}"
LIGHTHOUSE_BSZ="${LIGHTHOUSE_BSZ:-4}"
LIGHTHOUSE_EVAL_BSZ="${LIGHTHOUSE_EVAL_BSZ:-16}"
LIGHTHOUSE_NUM_WORKERS="${LIGHTHOUSE_NUM_WORKERS:-0}"
export LIGHTHOUSE_RESULTS_ROOT LIGHTHOUSE_BSZ LIGHTHOUSE_EVAL_BSZ LIGHTHOUSE_NUM_WORKERS
mkdir -p "$LIGHTHOUSE_RESULTS_ROOT"

cd "$LIGHTHOUSE_ROOT"

echo "Job started: $(date) | Host: $(hostname) | Job ID: ${JOB_ID:-interactive}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "QD-DETR finetune dataset: qvhighlight"
echo "QD-DETR finetune feature: clip_slowfast"
echo "Train batch size: $LIGHTHOUSE_BSZ | Eval batch size: $LIGHTHOUSE_EVAL_BSZ | Workers: $LIGHTHOUSE_NUM_WORKERS"
echo "Visual features resolve through lighthouse/training/config.py to shared HDF5 under:"
echo "  $PROJECT_ROOT/data/qvhighlights/hdf5/clip_slowfast_features.h5"

if [ ! -f "$BASE_CKPT" ]; then
    echo "Missing base checkpoint: $BASE_CKPT" >&2
    echo "Refusing to start a scratch run that could overwrite the original baseline results." >&2
    exit 1
fi

echo "Finetuning from checkpoint: $BASE_CKPT"
python training/train.py \
    --model qd_detr \
    --dataset qvhighlight \
    --feature clip_slowfast \
    --resume "$BASE_CKPT"
