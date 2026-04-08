#!/bin/bash
# =============================================================================
# VMR Project: Normalize project layout for SCC work
# Creates a single data layout under data/qvhighlights and points it at the
# already-downloaded Lighthouse assets so preprocessing and training use the
# same paths.
#
# Usage: bash scripts/normalize_project_layout.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
QV_DIR="$PROJECT_DIR/data/qvhighlights"
ANNO_DIR="$QV_DIR/annotations"
FEAT_DIR="$QV_DIR/features"
TXT_DIR="$QV_DIR/txt_features"
SUB_DIR="$QV_DIR/sub_features"
HDF5_DIR="$QV_DIR/hdf5"
ARCHIVE_DIR="$PROJECT_DIR/artifacts/archives"
LIGHTHOUSE_DIR="$PROJECT_DIR/lighthouse"
MOMENT_DETR_DIR="$PROJECT_DIR/moment_detr"

echo "============================================"
echo "VMR Project — Normalize Layout"
echo "============================================"
echo "Project dir: $PROJECT_DIR"
echo ""

mkdir -p "$ANNO_DIR" "$FEAT_DIR" "$TXT_DIR" "$SUB_DIR" "$HDF5_DIR" "$ARCHIVE_DIR"
rm -f "$ANNO_DIR/.gitkeep" "$FEAT_DIR/.gitkeep" "$TXT_DIR/.gitkeep"

if [ -d "$LIGHTHOUSE_DIR/data/qvhighlight" ]; then
    cp -f "$LIGHTHOUSE_DIR/data/qvhighlight/highlight_train_release.jsonl" "$ANNO_DIR/"
    cp -f "$LIGHTHOUSE_DIR/data/qvhighlight/highlight_val_release.jsonl" "$ANNO_DIR/"
    cp -f "$LIGHTHOUSE_DIR/data/qvhighlight/highlight_test_release.jsonl" "$ANNO_DIR/"
    cp -f "$LIGHTHOUSE_DIR/data/qvhighlight/subs_train.jsonl" "$ANNO_DIR/"
    echo "  ✓ annotations copied from lighthouse/data/qvhighlight"
elif [ -d "$MOMENT_DETR_DIR/data" ]; then
    cp -f "$MOMENT_DETR_DIR/data/highlight_train_release.jsonl" "$ANNO_DIR/"
    cp -f "$MOMENT_DETR_DIR/data/highlight_val_release.jsonl" "$ANNO_DIR/"
    cp -f "$MOMENT_DETR_DIR/data/highlight_test_release.jsonl" "$ANNO_DIR/"
    if [ -f "$MOMENT_DETR_DIR/data/subs_train.jsonl" ]; then
        cp -f "$MOMENT_DETR_DIR/data/subs_train.jsonl" "$ANNO_DIR/"
    fi
    echo "  ✓ annotations copied from moment_detr/data"
else
    echo "  ! No annotation source found under lighthouse/ or moment_detr/"
fi

if [ -d "$LIGHTHOUSE_DIR/features/qvhighlight/clip" ]; then
    ln -sfn ../../../lighthouse/features/qvhighlight/clip "$FEAT_DIR/clip"
    echo "  ✓ linked clip features"
fi

if [ -d "$LIGHTHOUSE_DIR/features/qvhighlight/slowfast" ]; then
    ln -sfn ../../../lighthouse/features/qvhighlight/slowfast "$FEAT_DIR/slowfast"
    echo "  ✓ linked slowfast features"
fi

if [ -d "$LIGHTHOUSE_DIR/features/qvhighlight/pann" ]; then
    ln -sfn ../../../lighthouse/features/qvhighlight/pann "$FEAT_DIR/pann"
    echo "  ✓ linked pann features"
fi

if [ -d "$LIGHTHOUSE_DIR/features/qvhighlight/clip_text" ]; then
    ln -sfn ../../../lighthouse/features/qvhighlight/clip_text "$TXT_DIR/clip_text"
    echo "  ✓ linked clip text features"
fi

if [ -d "$LIGHTHOUSE_DIR/features_tmp/qvhighlight/clip_sub_features" ]; then
    ln -sfn ../../../lighthouse/features_tmp/qvhighlight/clip_sub_features "$SUB_DIR/clip_sub"
    echo "  ✓ linked subtitle-aligned features"
fi

if [ -e "$PROJECT_DIR/lighthouse.zip" ]; then
    mv -f "$PROJECT_DIR/lighthouse.zip" "$ARCHIVE_DIR/"
    echo "  ✓ moved lighthouse.zip to artifacts/archives"
fi

if [ -e "$PROJECT_DIR/moment_detr.zip" ]; then
    mv -f "$PROJECT_DIR/moment_detr.zip" "$ARCHIVE_DIR/"
    echo "  ✓ moved moment_detr.zip to artifacts/archives"
fi

if [ -e "$LIGHTHOUSE_DIR/qvhighlight_features.tar.gz" ] && [ ! -L "$LIGHTHOUSE_DIR/qvhighlight_features.tar.gz" ]; then
    mv -f "$LIGHTHOUSE_DIR/qvhighlight_features.tar.gz" "$ARCHIVE_DIR/qvhighlight_features.zip"
    echo "  ✓ moved qvhighlight feature archive to artifacts/archives"
fi

if [ -e "$ARCHIVE_DIR/qvhighlight_features.zip" ]; then
    ln -sfn ../artifacts/archives/qvhighlight_features.zip "$LIGHTHOUSE_DIR/qvhighlight_features.tar.gz"
    echo "  ✓ preserved legacy archive path via symlink"
fi

rm -rf "$PROJECT_DIR/tmp_inspect"
rm -f "$PROJECT_DIR/tmp_sample.npz"

echo ""
echo "Normalized layout:"
find "$QV_DIR" -maxdepth 2 -print | sort
echo ""
echo "Done."
