#!/bin/bash -l
# Head ablation study for QD-DETR
# Submit: qsub /projectnb/cs585/projects/VMR/vmr_project/analysis/head_ablation/run_head_ablation.sh

#$ -P cs585
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -l h_rt=06:00:00
#$ -l mem_per_core=8G
#$ -pe omp 2
#$ -N vmr_head_ablation
#$ -o /projectnb/cs585/projects/VMR/vmr_project/logs/head_ablation_$JOB_ID.log
#$ -j y

set -euo pipefail

module load miniconda
conda activate vmr

PROJECT_ROOT="/projectnb/cs585/projects/VMR/vmr_project"
cd "$PROJECT_ROOT/lighthouse"

echo "Job started: $(date) | Host: $(hostname) | Job ID: ${JOB_ID:-interactive}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "Running head ablation study..."
python ../analysis/head_ablation/head_ablation.py

echo "Job completed: $(date)"
