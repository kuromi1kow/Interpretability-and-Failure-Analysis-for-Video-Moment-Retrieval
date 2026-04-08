#!/bin/bash -l
# =============================================================================
# VMR Project: Train Moment-DETR on QVHighlights
# Submit with: qsub scripts/train_moment_detr.sh
# =============================================================================

# --- SCC Job Configuration ---
#$ -P cs585
#$ -l gpus=1                  # Request 1 GPU
#$ -l gpu_type=V100|A100      # V100 or A100 (either works)
#$ -l h_rt=08:00:00           # 8 hour walltime (training ~2-4 hrs, buffer for safety)
#$ -l mem_free=64G            # RAM per CPU
#$ -l h_vmem=64G              # Hard Memory Limit
#$ -pe omp 4                  # 4 CPU cores (for data loading workers)
#$ -N vmr_moment_detr         # Job name
#$ -o logs/train_moment_detr_$JOB_ID.log   # Stdout log
#$ -j y                       # Merge stderr into stdout
#$ -m ea                      # Email on end/abort
#$ -M aperez00@bu.edu       # ← Uncomment and set your email

# --- Environment Setup ---
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $JOB_ID"

module load miniconda
# uncomment the below line if  you haven't activated the vmr yet
# conda activate vmr

# Navigate to project directory
PROJECT_ROOT="${PROJECT_ROOT:-/projectnb/cs585/projects/VMR/vmr_project}"
cd "$PROJECT_ROOT"

echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- Training Configuration ---
MODEL="moment_detr"
FEATURES="clip"                    # "clip" or "clip_slowfast"
SEED=0
EPOCHS=200
BATCH_SIZE=32
LR=1e-4
SAVE_DIR="checkpoints/${MODEL}_${FEATURES}_seed${SEED}_$(date +%Y%m%d)"

mkdir -p "$SAVE_DIR"
mkdir -p logs

# --- Option A: Train via Lighthouse (recommended) ---
# Lighthouse provides a unified interface for all DETR variants.
# Check lighthouse docs for exact CLI syntax after install.
echo "============================================"
echo "Training $MODEL with $FEATURES features"
echo "Epochs: $EPOCHS | Batch: $BATCH_SIZE | LR: $LR | Seed: $SEED"
echo "Save dir: $SAVE_DIR"
echo "============================================"

# Uncomment ONE of the following training commands:

# Option A: Lighthouse-based training
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

# Option B: Original Moment-DETR training script
 cd "$PROJECT_ROOT/moment_detr"
 python -m moment_detr.train \
     --dset_name hl \
     --train_path "$PROJECT_ROOT/data/qvhighlights/annotations/highlight_train_release.jsonl" \
     --eval_path "$PROJECT_ROOT/data/qvhighlights/annotations/highlight_val_release.jsonl" \
     --v_feat_dirs "$PROJECT_ROOT/lighthouse/features_tmp/qvhighlight/clip_features" \
     --t_feat_dir "$PROJECT_ROOT/lighthouse/features_tmp/qvhighlight/clip_text_features" \
     --v_feat_dim 512 \
     --t_feat_dim 512 \
     --results_root "$SAVE_DIR" \
     --exp_id moment_detr_run \
     --n_epoch "$EPOCHS" \
     --bsz "$BATCH_SIZE" \
     --lr "$LR" \
     --seed "$SEED" \
     --num_workers 2

echo "============================================"
echo "Training complete: $(date)"
echo "Checkpoint saved to: $SAVE_DIR"
echo "============================================"

# --- Log results ---
mkdir -p results
echo "$MODEL,$FEATURES,$SEED,$EPOCHS,$BATCH_SIZE,$LR,$SAVE_DIR,$(date)" \
    >> results/experiment_log.csv
