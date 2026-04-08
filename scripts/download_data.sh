#!/bin/bash
# =============================================================================
# VMR Project: Data Download Script
# Downloads QVHighlights annotations and pre-extracted features.
#
# Run this from the project root directory.
# Usage: bash scripts/download_data.sh
#
# IMPORTANT: Run inside tmux/screen so downloads survive disconnects!
#   tmux new -s download
#   bash scripts/download_data.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data/qvhighlights"

echo "============================================"
echo "VMR Project — Data Download"
echo "============================================"
echo "Project dir: $PROJECT_DIR"
echo "Data dir:    $DATA_DIR"
echo ""

mkdir -p "$DATA_DIR/annotations"
mkdir -p "$DATA_DIR/features"

if [ -d "$PROJECT_DIR/lighthouse/data/qvhighlight" ] && [ -d "$PROJECT_DIR/lighthouse/features/qvhighlight" ]; then
    echo "Found local Lighthouse data and features."
    echo "Normalizing the shared project layout instead of re-downloading."
    echo ""
    bash "$PROJECT_DIR/scripts/normalize_project_layout.sh"
    exit 0
fi

# --- Option A: Use Lighthouse built-in download (recommended) ---
echo "============================================"
echo "Option A: Lighthouse auto-download"
echo "============================================"
echo ""
echo "Lighthouse can auto-download everything when you first run training."
echo "If you prefer manual download, continue below."
echo ""

# --- Option B: Manual download ---
echo "============================================"
echo "Option B: Manual download"
echo "============================================"

# Step 1: QVHighlights annotations from Moment-DETR repo
echo "[1/3] Downloading QVHighlights annotations..."
cd "$DATA_DIR/annotations"

if [ ! -f "highlight_train_release.jsonl" ]; then
    # Annotations are in the moment_detr repo data/ directory
    # Download directly from the GitHub repo
    wget -c "https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_train_release.jsonl" \
        -O highlight_train_release.jsonl 2>/dev/null || echo "  Note: may need to get from cloned repo"
    wget -c "https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_val_release.jsonl" \
        -O highlight_val_release.jsonl 2>/dev/null || echo "  Note: may need to get from cloned repo"
    wget -c "https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_test_release.jsonl" \
        -O highlight_test_release.jsonl 2>/dev/null || echo "  Note: may need to get from cloned repo"
    echo "  ✓ annotations downloaded"
else
    echo "  annotations already exist, skipping"
fi

# If wget from GitHub failed, copy from cloned repo
if [ ! -f "highlight_train_release.jsonl" ] && [ -d "$PROJECT_DIR/moment_detr/data" ]; then
    echo "  Copying from cloned moment_detr repo..."
    cp "$PROJECT_DIR/moment_detr/data/highlight_"*".jsonl" . 2>/dev/null || true
fi

# Step 2: Pre-extracted features
echo "[2/3] Downloading pre-extracted features..."
echo ""
echo "  QVHighlights features are hosted on Google Drive / the Moment-DETR repo."
echo "  The features include:"
echo "    - SlowFast visual features (~4-6 GB)"
echo "    - CLIP visual+text features (~2-3 GB)"
echo ""
echo "  ========================================================"
echo "  MANUAL STEP REQUIRED:"
echo "  ========================================================"
echo ""
echo "  The pre-extracted features must be downloaded manually."
echo "  Follow these steps:"
echo ""
echo "  1. Go to the Moment-DETR GitHub page:"
echo "     https://github.com/jayleicn/moment_detr"
echo ""
echo "  2. In the README, find the 'Features' download links."
echo "     They typically link to Google Drive."
echo ""
echo "  3. Download the feature files to:"
echo "     $DATA_DIR/features/"
echo ""
echo "  4. Alternative: Use gdown (pip install gdown) to download"
echo "     from Google Drive via command line:"
echo "     gdown <GOOGLE_DRIVE_FILE_ID> -O features/"
echo ""
echo "  5. If using Lighthouse, check:"
echo "     https://github.com/line-corporation/lighthouse"
echo "     Lighthouse may auto-download features on first run."
echo ""
echo "  ========================================================"
echo ""

# Step 3: Verify what we have
echo "[3/3] Checking data directory..."
echo ""
echo "Annotations:"
ls -lh "$DATA_DIR/annotations/" 2>/dev/null || echo "  (empty)"
echo ""
echo "Features:"
ls -lh "$DATA_DIR/features/" 2>/dev/null || echo "  (empty — see manual step above)"
echo ""

cd "$PROJECT_DIR"

echo "============================================"
echo "Download script complete."
echo "============================================"
echo ""
echo "Verify annotations exist with:"
echo "  python scripts/verify_data.py"
echo ""
echo "If Lighthouse data already exists locally, prefer:"
echo "  bash scripts/normalize_project_layout.sh"
echo ""
