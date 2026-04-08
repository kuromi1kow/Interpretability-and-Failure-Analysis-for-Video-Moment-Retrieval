# Video Moment Retrieval: Interpretability and Failure Analysis

**CS585: Image and Video Computing — Boston University**

Team: Assylkhan Geniyat, Pablo Bello, Jaime Bernal, Zukhriddin Fakhriddinov, Aidan Perez

## Overview

This project studies **Video Moment Retrieval (VMR)** — localizing temporal segments in untrimmed videos given natural language queries. We reproduce three DETR-style baselines on QVHighlights and conduct interpretability analyses to understand how these models work and fail.

## Key Results

| Model | R1@0.5 | R1@0.7 | mAP |
|-------|--------|--------|-----|
| Moment-DETR | 54.06 | 36.52 | 32.80 |
| QD-DETR | 62.06 | 45.48 | 40.56 |
| CG-DETR | **62.84** | **46.84** | **42.00** |

### Our Contributions

1. **Temporal Bias Analysis**: Models achieve 7x above trivial baselines but systematically fail on short moments (<10s), over-predicting length by 7x.
2. **Verb vs. Noun Sensitivity**: Object nouns are 2.6x more important than action verbs for moment grounding in QD-DETR.
3. **Head Ablation**: No specialized boundary heads found — temporal grounding is a distributed emergent property across all 32 attention heads.

## Project Goals

1. **Reproduce** strong baselines (Moment-DETR, QD-DETR, CG-DETR) on QVHighlights
2. **Intellectual contribution**: temporal interpretability, bias analysis, verb-vs-noun sensitivity

## Quick Start on SCC

```bash
# 1. Go to the shared SCC project folder
cd /projectnb/cs585/projects/VMR/vmr_project

# 2. Run the setup script (creates conda env + installs everything)
bash scripts/setup_env.sh

# 3. Normalize the project layout so everyone uses the same data paths
bash scripts/normalize_project_layout.sh

# 4. Verify annotations and linked features
python scripts/verify_data.py

# 5. Optional preprocessing: pack clip features into one HDF5 file
python scripts/preprocess_to_hdf5.py \
  --src-dir data/qvhighlights/features/clip \
  --out-h5 data/qvhighlights/hdf5/clip_features.h5 \
  --out-index data/qvhighlights/hdf5/clip_features_index.json

# 5b. Optional preprocessing: pack slowfast features too
python scripts/preprocess_to_hdf5.py \
  --src-dir /projectnb/cs585/projects/VMR/vmr_project/lighthouse/features_tmp/qvhighlight/slowfast_features \
  --out-h5 data/qvhighlights/hdf5/slowfast_features.h5 \
  --out-index data/qvhighlights/hdf5/slowfast_features_index.json

# 6. Verify GPU access
qrsh -P cs585 -l gpus=1 -l h_rt=00:30:00
conda activate vmr
python scripts/verify_gpu.py

# 7. Train first baseline
qsub scripts/train_moment_detr.sh
```

## Notes

- This project requires at least Python 3.10.  If you don't have this run the command: `module load python3/3.10.12`

- The project name in the qrsh command is likely something like "cs585"

- If you're version of Python is outdated and you update it, you'll also probably want to update pip.  Run `python -m pip install --upgrade pip` to do that

- If `conda activate vmr` doesn't work, you can run `source vmr/bin/activate`

- If neither of the above two commands work, you'll want to run `python -m venv vmr`, then one of the above two commands should work after that

- If when you run `python scripts/verify_gpu.py` it says any of the libraries are not found, you'll want run the following commands:

```bash
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126

python -m pip install \
  transformers einops spacy scipy pandas matplotlib seaborn \
  h5py tensorboard scikit-learn

python -m spacy download en_core_web_sm
```

- If lighthouse says it is not found, you'll want to run the following commands:

```bash
cd /projectnb/cs585/projects/VMR/vmr_project/lighthouse
pip install -e .
cd ..
```

- Read through the train_moment_detr.sh script to understand how the training is set up, and modify it as needed for your experiments.  You can submit it with `qsub` as shown above, or you can run the commands interactively in a `qrsh` session for faster iteration.  There may be some places where it says to uncomment.  Be sure to do that if you want to run those parts of the code.

## Directory Structure

```
vmr/
├── data/
│   ├── qvhighlights/
│   │   ├── annotations/       # train/val/test jsonl + subs_train.jsonl
│   │   ├── features/          # clip/, slowfast/, pann/
│   │   ├── txt_features/      # clip_text/
│   │   ├── sub_features/      # clip_sub/
│   │   └── hdf5/              # packed HDF5 features + index json
│   └── charades_sta/          # optional generalization dataset
├── src/
│   ├── models/                # model code (from Lighthouse / custom)
│   ├── analysis/              # interpretability, bias, linguistic scripts
│   ├── eval/                  # evaluation scripts
│   └── utils/                 # data loaders, plotting, helpers
├── configs/                   # experiment configs (YAML)
├── checkpoints/               # saved model weights
├── logs/                      # TensorBoard + job output logs
├── results/                   # eval tables, predictions, figures
├── scripts/                   # SCC batch job scripts
├── notebooks/                 # Jupyter notebooks for exploration
└── README.md
```

## Experiment Tracking

All experiment results are logged to:
- `logs/` — TensorBoard training curves
- `results/experiment_log.csv` — metrics summary table
- Each checkpoint directory includes its config YAML

## Data Workflow

Use one shared layout under `data/qvhighlights` for both preprocessing and training:

- `bash scripts/normalize_project_layout.sh` copies annotations and links feature folders from the local Lighthouse checkout.
- `python scripts/verify_data.py` verifies that annotations and feature files are visible from the shared layout.
- `python scripts/preprocess_to_hdf5.py ...` converts per-video `.npz` features into one HDF5 file plus an offset index.
- `python scripts/demo_hdf5_loading.py ...` loads one video's features from the packed HDF5 using its `video_id`.
- `DATA_LAYOUT.md` explains the shared SCC data structure, feature types, and which HDF5 files teammates should use.

## HDF5 Training Patch

If a teammate wants to keep the existing training code but read packed HDF5 features instead of thousands of `.npz` files, they can use [training_feature_loader.py](/Users/assylkhan/Downloads/vmr_project/src/utils/training_feature_loader.py).

Minimal replacement for `_get_video_feat_by_vid` in a dataset class:

```python
from src.utils.training_feature_loader import load_video_features_for_training

def _get_video_feat_by_vid(self, vid):
    return load_video_features_for_training(
        video_id=vid,
        sources=self.v_feat_dirs,
        max_v_l=self.max_v_l,
        normalize=self.normalize_v,
    )
```

Example HDF5 source paths:

```python
self.v_feat_dirs = [
    "data/qvhighlights/hdf5/clip_features.h5",
]
```

or for concatenated features:

```python
self.v_feat_dirs = [
    "data/qvhighlights/hdf5/clip_features.h5",
    "data/qvhighlights/hdf5/slowfast_features.h5",
]
```

## Team Assignments

| Person | Primary Task | Secondary |
|--------|-------------|-----------|
| Assylkhan | Infrastructure, SCC setup, Lighthouse | Coordination |
| Pablo | Data pipeline, feature verification | Bias analysis |
| Jaime | Evaluation scripts, result tables | Report |
| Zukhriddin | Attention extraction, saliency viz | Erasure tests |
| Aidan | Linguistic ablation (spaCy + re-encoding) | Verb/noun analysis |
