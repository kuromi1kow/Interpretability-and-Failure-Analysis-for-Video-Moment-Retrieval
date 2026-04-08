#!/bin/bash
# =============================================================================
# VMR Project: Environment Setup Script for BU SCC
# Run this ONCE to create the conda environment with all dependencies.
# Usage: bash scripts/setup_env.sh
# =============================================================================

set -e  # exit on error

echo "============================================"
echo "VMR Project — Environment Setup"
echo "============================================"

# --- Step 1: Load miniconda module ---
echo "[1/6] Loading miniconda module..."
module load miniconda
echo "  ✓ miniconda loaded"

# --- Step 2: Create conda environment ---
ENV_NAME="vmr"
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "[2/6] Conda env '$ENV_NAME' already exists. Skipping creation."
    echo "  (To recreate: conda remove -n $ENV_NAME --all)"
else
    echo "[2/6] Creating conda environment '$ENV_NAME' with Python 3.10..."
    conda create -n "$ENV_NAME" python=3.10 -y
    echo "  ✓ environment created"
fi

# --- Step 3: Activate and install PyTorch with CUDA ---
echo "[3/6] Activating environment and installing PyTorch..."
conda activate "$ENV_NAME"

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

echo "  ✓ PyTorch installed with CUDA 11.8"

# --- Step 4: Install core ML/NLP packages ---
echo "[4/6] Installing core packages..."
pip install \
    transformers==4.36.2 \
    einops==0.7.0 \
    tensorboard==2.15.1 \
    scipy==1.11.4 \
    pandas==2.1.4 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    tqdm==4.66.1 \
    pyyaml==6.0.1 \
    h5py==3.10.0 \
    spacy==3.7.2 \
    scikit-learn==1.3.2 \
    jupyter==1.0.0 \
    ipykernel==6.27.1

echo "  ✓ core packages installed"

# --- Step 5: Install spaCy English model (for linguistic analysis) ---
echo "[5/6] Installing spaCy English model..."
python -m spacy download en_core_web_sm
echo "  ✓ spaCy model installed"

# --- Step 6: Clone required repos ---
echo "[6/6] Cloning required repositories..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

if [ ! -d "lighthouse" ]; then
    git clone https://github.com/line-corporation/lighthouse.git
    echo "  ✓ Lighthouse cloned"
else
    echo "  Lighthouse already exists, skipping"
fi

if [ ! -d "moment_detr" ]; then
    git clone https://github.com/jayleicn/moment_detr.git
    echo "  ✓ Moment-DETR cloned"
else
    echo "  Moment-DETR already exists, skipping"
fi

# Install Lighthouse in editable mode
cd lighthouse
pip install -e .
cd "$PROJECT_DIR"
echo "  ✓ Lighthouse installed"

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Verify GPU: qrsh -P <project> -l gpus=1 -l h_rt=00:30:00"
echo "     conda activate vmr && python scripts/verify_gpu.py"
echo "  2. Normalize shared data layout: bash scripts/normalize_project_layout.sh"
echo "  3. Verify data: python scripts/verify_data.py"
echo "  4. Train baseline: qsub scripts/train_moment_detr.sh"
echo ""
