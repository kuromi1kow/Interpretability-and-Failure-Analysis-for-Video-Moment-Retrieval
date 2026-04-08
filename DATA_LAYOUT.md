# Data Layout and Preprocessing Guide

This file explains where the shared data lives on SCC, what each feature type means, and which files teammates should use for training.

## Project Root on SCC

```bash
/projectnb/cs585/projects/VMR/vmr_project
```

All paths below are relative to that folder unless noted otherwise.

## High-Level Idea

The team does **not** train on raw `.mp4` videos in the current setup.

Instead, the project uses:

- annotation files (`.jsonl`)
- extracted feature files (`.npz` / `.npy`)
- packed HDF5 files (`.h5`) plus JSON indices for faster loading

Think of the pipeline as:

```text
annotations + extracted features -> HDF5 packing -> model training
```

## Main Shared Data Layout

```text
data/
├── qvhighlights/
│   ├── annotations/
│   ├── features/
│   │   ├── clip/
│   │   ├── slowfast/
│   │   └── pann/
│   ├── txt_features/
│   │   └── clip_text/
│   ├── sub_features/
│   │   └── clip_sub/
│   └── hdf5/
└── charades_sta/
    ├── annotations/
    ├── features/
    └── hdf5/
```

## What Each Folder Means

### `data/qvhighlights/annotations`

Official split files for QVHighlights:

- `highlight_train_release.jsonl`
- `highlight_val_release.jsonl`
- `highlight_test_release.jsonl`
- `subs_train.jsonl`

These define the train/val/test examples and subtitle metadata.

### `data/qvhighlights/features/clip`

Visual CLIP features for video segments.

- one file per video segment
- usually stored as `.npz`
- key inside file: `features`

Use this when a model expects CLIP-based visual features.

### `data/qvhighlights/features/slowfast`

Visual SlowFast features for video segments.

- one file per video segment
- usually stored as `.npz`
- key inside file: `features`

Use this when a model expects motion-heavy visual features.

### `data/qvhighlights/features/pann`

Audio features.

- one file per video segment
- usually stored as `.npy`

Use this if the model includes audio.

### `data/qvhighlights/txt_features/clip_text`

Query-text CLIP features.

Each file usually contains:

- `last_hidden_state`
- `pooler_output`

These are text embeddings for the query sentence.

### `data/qvhighlights/sub_features/clip_sub`

Subtitle-text CLIP features.

Each file usually contains:

- `last_hidden_state`
- `pooler_output`

These are text embeddings for subtitle chunks from the video.

## Meaning of `last_hidden_state` vs `pooler_output`

These come from transformer-style text encoders and are not duplicates.

- `last_hidden_state`
  - sequence-level representation
  - shape is usually something like `(num_tokens, 512)`
  - keeps more token-level detail

- `pooler_output`
  - single vector for the whole text
  - shape is usually `(512,)`
  - smaller and more compact

Some models prefer token-level features, others only need one vector per text item.

## Why HDF5 Exists

The raw extracted features are spread across many small files.

That works, but it is inconvenient for training because:

- opening hundreds of thousands of small files is slow
- shared training code becomes harder to manage
- random access is less efficient

So preprocessing packs them into:

- one `.h5` file containing all rows
- one `_index.json` mapping item ids to offsets

This makes training and data loading much cleaner.

## Training-Ready HDF5 Files

The shared HDF5 outputs live in:

```bash
data/qvhighlights/hdf5
```

Typical ready files include:

- `clip_features.h5`
- `clip_features_index.json`
- `slowfast_features.h5`
- `slowfast_features_index.json`
- `clip_slowfast_features.h5`
- `clip_slowfast_features_index.json`
- `pann_features.h5`
- `pann_features_index.json`
- `clip_text_last_hidden_state.h5`
- `clip_text_last_hidden_state_index.json`
- `clip_text_pooler_output.h5`
- `clip_text_pooler_output_index.json`
- `clip_sub_last_hidden_state.h5`
- `clip_sub_last_hidden_state_index.json`
- `clip_sub_pooler_output.h5`
- `clip_sub_pooler_output_index.json`

If the last two `clip_sub_pooler_output` files are missing, preprocessing is still finishing.

## Recommended Feature Choices

For visual-only training:

- `clip_features.h5`
- `slowfast_features.h5`
- or the merged `clip_slowfast_features.h5`

For audio:

- `pann_features.h5`

For query text:

- `clip_text_last_hidden_state.h5` if token-level text is needed
- `clip_text_pooler_output.h5` if one vector per query is enough

For subtitle text:

- `clip_sub_last_hidden_state.h5` if token-level subtitle encoding is needed
- `clip_sub_pooler_output.h5` if one vector per subtitle item is enough

## Recommended Default for Teammates

If someone just wants a clean visual input for baseline training, the simplest option is:

```python
self.v_feat_dirs = [
    "/projectnb/cs585/projects/VMR/vmr_project/data/qvhighlights/hdf5/clip_slowfast_features.h5",
]
```

If they want separate visual sources:

```python
self.v_feat_dirs = [
    "/projectnb/cs585/projects/VMR/vmr_project/data/qvhighlights/hdf5/clip_features.h5",
    "/projectnb/cs585/projects/VMR/vmr_project/data/qvhighlights/hdf5/slowfast_features.h5",
]
```

## Helper Code

The helper loader lives in:

- `src/utils/hdf5_features.py`
- `src/utils/training_feature_loader.py`

Patched training datasets now support `.h5` inputs in:

- `moment_detr/moment_detr/start_end_dataset.py`
- `lighthouse/training/dataset.py`
- `lighthouse/training/cg_detr_dataset.py`

## Quick Checks

List packed outputs:

```bash
ls -lh data/qvhighlights/hdf5
```

Load one item from HDF5:

```bash
python scripts/demo_hdf5_loading.py \
  --h5 data/qvhighlights/hdf5/clip_features.h5 \
  --index data/qvhighlights/hdf5/clip_features_index.json \
  --video-id AO3sNhzP2Tg_510.0_660.0
```

Check whether a long preprocessing job is still running:

```bash
ps -u kuromiqo -f | grep preprocess_to_hdf5.py
```

Watch a preprocessing log:

```bash
tail -f logs/preprocess_clip_sub_last_hidden_state.log
```

## `charades_sta` Status

`data/charades_sta/annotations` currently contains the split files, but the project does **not** currently include prepared Charades feature files in the shared layout.

That means:

- annotations are available
- feature preprocessing for Charades is not fully set up in this project yet

## One-Sentence Summary

If you are training on the shared SCC project, use the packed files in `data/qvhighlights/hdf5` whenever possible, and treat `features/`, `txt_features/`, and `sub_features/` as the raw extracted feature sources behind them.
