#!/usr/bin/env python3
"""
VMR Project: Data Verification and Statistics
Verifies that QVHighlights data is correctly downloaded and computes
dataset statistics needed for the report.

Usage: python scripts/verify_data.py
"""

import json
import os
import sys
import glob
import numpy as np
from pathlib import Path
from collections import Counter

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "qvhighlights"
ANNO_DIR = DATA_DIR / "annotations"
FEAT_DIR = DATA_DIR / "features"

FEATURE_SUFFIXES = {".npz", ".npy", ".h5", ".hdf5", ".pt", ".pth"}


def load_jsonl(path):
    """Load a JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def collect_feature_files(root_dir):
    """Collect feature files without recursing endlessly through symlinked dirs."""
    found_files = []
    visited_dirs = set()

    def scan_dir(path):
        real_path = path.resolve()
        real_key = str(real_path)
        if real_key in visited_dirs:
            return
        visited_dirs.add(real_key)

        for current_root, _, files in os.walk(real_path, followlinks=False):
            for name in files:
                suffix = Path(name).suffix
                if suffix in FEATURE_SUFFIXES:
                    found_files.append(str(Path(current_root) / name))

    if not root_dir.exists():
        return found_files

    for entry in root_dir.iterdir():
        if entry.is_file() and entry.suffix in FEATURE_SUFFIXES:
            found_files.append(str(entry.resolve()))
        elif entry.is_dir():
            scan_dir(entry)

    return sorted(set(found_files))


def check_annotations():
    """Check annotation files exist and print statistics."""
    print("=" * 60)
    print("ANNOTATION CHECK")
    print("=" * 60)

    splits = {
        "train": "highlight_train_release.jsonl",
        "val": "highlight_val_release.jsonl",
        "test": "highlight_test_release.jsonl",
    }

    all_ok = True
    stats = {}

    for split_name, filename in splits.items():
        path = ANNO_DIR / filename
        if not path.exists():
            print(f"  [✗] {split_name}: {filename} NOT FOUND")
            all_ok = False
            continue

        data = load_jsonl(path)
        print(f"  [✓] {split_name}: {len(data)} samples")

        # Collect statistics
        durations = []
        query_lengths = []
        moment_durations = []
        moment_centers = []

        for item in data:
            vid_dur = item.get("duration", 0)
            durations.append(vid_dur)

            query = item.get("query", "")
            query_lengths.append(len(query.split()))

            # Relevant windows (moment annotations)
            windows = item.get("relevant_windows", [])
            for w in windows:
                if len(w) == 2:
                    start, end = w
                    moment_durations.append(end - start)
                    if vid_dur > 0:
                        center = ((start + end) / 2) / vid_dur
                        moment_centers.append(center)

        stats[split_name] = {
            "count": len(data),
            "vid_dur_mean": np.mean(durations) if durations else 0,
            "vid_dur_std": np.std(durations) if durations else 0,
            "vid_dur_min": np.min(durations) if durations else 0,
            "vid_dur_max": np.max(durations) if durations else 0,
            "query_len_mean": np.mean(query_lengths) if query_lengths else 0,
            "moment_dur_mean": np.mean(moment_durations) if moment_durations else 0,
            "moment_dur_std": np.std(moment_durations) if moment_durations else 0,
            "moment_center_mean": np.mean(moment_centers) if moment_centers else 0,
            "moment_center_std": np.std(moment_centers) if moment_centers else 0,
            "n_moments": len(moment_durations),
        }

    if stats:
        print()
        print("DATASET STATISTICS")
        print("-" * 60)
        print(f"{'Stat':<30} {'Train':>10} {'Val':>10} {'Test':>10}")
        print("-" * 60)
        for key in ["count", "vid_dur_mean", "vid_dur_std",
                     "query_len_mean", "moment_dur_mean", "moment_center_mean"]:
            row = f"{key:<30}"
            for split in ["train", "val", "test"]:
                if split in stats:
                    val = stats[split].get(key, "—")
                    if isinstance(val, float):
                        row += f" {val:>10.2f}"
                    else:
                        row += f" {val:>10}"
                else:
                    row += f" {'—':>10}"
            print(row)
        print("-" * 60)

    return all_ok, stats


def check_features():
    """Check that pre-extracted features exist."""
    print()
    print("=" * 60)
    print("FEATURE CHECK")
    print("=" * 60)

    if not FEAT_DIR.exists():
        print("  [✗] features/ directory does not exist")
        return False

    found_files = collect_feature_files(FEAT_DIR)

    if not found_files:
        print("  [✗] No feature files found in features/")
        print("      Expected: .npy, .h5, .pt files with SlowFast/CLIP features")
        print("      See scripts/download_data.sh for download instructions")
        return False

    print(f"  [✓] Found {len(found_files)} feature files")

    # Print summary by type
    by_ext = Counter(Path(f).suffix for f in found_files)
    for ext, count in by_ext.items():
        print(f"      {ext}: {count} files")

    # Check total size
    total_bytes = sum(os.path.getsize(f) for f in found_files)
    total_gb = total_bytes / (1024 ** 3)
    print(f"      Total size: {total_gb:.2f} GB")

    # Try to load one file to verify shape
    sample = found_files[0]
    ext = Path(sample).suffix
    try:
        if ext == ".npz":
            data = np.load(sample)
            keys = list(data.files)
            print(f"      Sample keys ({Path(sample).name}): {keys}")
            if keys:
                arr = data[keys[0]]
                print(f"      Sample shape ({keys[0]}): {arr.shape}")
        elif ext == ".npy":
            arr = np.load(sample)
            print(f"      Sample shape ({Path(sample).name}): {arr.shape}")
        elif ext in [".h5", ".hdf5"]:
            import h5py
            with h5py.File(sample, "r") as f:
                keys = list(f.keys())[:3]
                print(f"      Sample keys ({Path(sample).name}): {keys}")
    except Exception as e:
        print(f"      Could not inspect sample: {e}")

    return True


def save_stats(stats):
    """Save dataset statistics to results/."""
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "dataset_statistics.json"
    # Convert numpy types to python types
    clean_stats = {}
    for split, s in stats.items():
        clean_stats[split] = {k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                              for k, v in s.items()}
    with open(out_path, "w") as f:
        json.dump(clean_stats, f, indent=2)
    print(f"\nStats saved to {out_path}")


def main():
    print()
    print("VMR Project — Data Verification")
    print()

    anno_ok, stats = check_annotations()
    feat_ok = check_features()

    if stats:
        save_stats(stats)

    print()
    print("=" * 60)
    if anno_ok and feat_ok:
        print("ALL CHECKS PASSED ✓  Ready for training!")
    elif anno_ok:
        print("Annotations OK, but features missing.")
        print("Download features before training (see download_data.sh).")
    else:
        print("MISSING DATA — see above for details.")
    print("=" * 60)


if __name__ == "__main__":
    main()
