#!/bin/bash
# =============================================================================
# VMR Project: Quick Status Checklist
# Run this to see what's done and what still needs to be done.
# Usage: bash scripts/checklist.sh
# =============================================================================

echo ""
echo "============================================"
echo "  VMR Project — Setup Checklist"
echo "============================================"
echo ""

# Change this to your actual project path
PROJECT_DIR="${VMR_PROJECT_DIR:-$(pwd)}"

check() {
    local label="$1"
    local condition="$2"
    if eval "$condition" 2>/dev/null; then
        echo "  [✓] $label"
    else
        echo "  [✗] $label"
    fi
}

echo "--- Environment ---"
check "Conda env 'vmr' exists" "conda info --envs 2>/dev/null | grep -q vmr"
check "PyTorch installed" "python -c 'import torch' 2>/dev/null"
check "CUDA available" "python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null"
check "transformers installed" "python -c 'import transformers' 2>/dev/null"
check "spaCy installed" "python -c 'import spacy' 2>/dev/null"
check "spaCy en model" "python -c 'import spacy; spacy.load(\"en_core_web_sm\")' 2>/dev/null"

echo ""
echo "--- Repositories ---"
check "Lighthouse cloned" "test -d $PROJECT_DIR/lighthouse"
check "Moment-DETR cloned" "test -d $PROJECT_DIR/moment_detr"
check "Lighthouse installed" "python -c 'import lighthouse' 2>/dev/null"

echo ""
echo "--- Data ---"
check "Annotations dir exists" "test -d $PROJECT_DIR/data/qvhighlights/annotations"
check "Train annotations" "test -f $PROJECT_DIR/data/qvhighlights/annotations/highlight_train_release.jsonl"
check "Val annotations" "test -f $PROJECT_DIR/data/qvhighlights/annotations/highlight_val_release.jsonl"
check "Features dir exists" "test -d $PROJECT_DIR/data/qvhighlights/features"
check "Features downloaded" "test -n \"\$(ls -A $PROJECT_DIR/data/qvhighlights/features/ 2>/dev/null)\""

echo ""
echo "--- Project Structure ---"
check "src/ directory" "test -d $PROJECT_DIR/src"
check "configs/ directory" "test -d $PROJECT_DIR/configs"
check "scripts/ directory" "test -d $PROJECT_DIR/scripts"
check "checkpoints/ directory" "test -d $PROJECT_DIR/checkpoints"
check "logs/ directory" "test -d $PROJECT_DIR/logs"
check "results/ directory" "test -d $PROJECT_DIR/results"
check "experiment_log.csv" "test -f $PROJECT_DIR/results/experiment_log.csv"

echo ""
echo "--- Training Status ---"
check "Moment-DETR checkpoint" "test -n \"\$(find $PROJECT_DIR/checkpoints -name '*moment_detr*' 2>/dev/null | head -1)\""
check "QD-DETR checkpoint" "test -n \"\$(find $PROJECT_DIR/checkpoints -name '*qd_detr*' 2>/dev/null | head -1)\""
check "CG-DETR checkpoint" "test -n \"\$(find $PROJECT_DIR/checkpoints -name '*cg_detr*' 2>/dev/null | head -1)\""

echo ""
echo "============================================"
echo "  [✓] = done   [✗] = not yet"
echo "============================================"
echo ""
