#!/usr/bin/env python3
"""
Convert per-video feature files (.npz/.npy) into a single HDF5 file plus
an index mapping video_id -> (offset, num_clips).

Usage:
  python scripts/preprocess_to_hdf5.py \
      --src-dir data/qvhighlights/features/clip \
      --out-h5 data/qvhighlights/hdf5/clip_features.h5 \
      --out-index data/qvhighlights/hdf5/clip_features_index.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", required=True, help="Directory containing .npz/.npy feature files")
    parser.add_argument("--out-h5", required=True, help="Output HDF5 file path")
    parser.add_argument("--out-index", required=True, help="Output JSON index path")
    parser.add_argument(
        "--feature-key",
        default="features",
        help="Preferred key inside .npz files. Falls back to the only available key if needed.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Storage dtype inside HDF5",
    )
    parser.add_argument(
        "--compression",
        default="lzf",
        choices=["lzf", "gzip", "none"],
        help="HDF5 compression",
    )
    parser.add_argument(
        "--gzip-level",
        type=int,
        default=2,
        help="Compression level when --compression=gzip",
    )
    parser.add_argument(
        "--allow-1d",
        action="store_true",
        help="Allow 1D arrays and store them as a single-row 2D feature matrix.",
    )
    return parser.parse_args()


def load_feature_array(path: Path, preferred_key: str, allow_1d: bool = False) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        if preferred_key in data.files:
            array = data[preferred_key]
        elif len(data.files) == 1:
            array = data[data.files[0]]
        else:
            raise ValueError(f"Could not choose feature key for {path.name}: {data.files}")
    elif path.suffix == ".npy":
        array = np.load(path)
    else:
        raise ValueError(f"Unsupported feature format: {path}")

    if array.ndim == 1 and allow_1d:
        array = array[None, :]

    if array.ndim != 2:
        raise ValueError(f"Expected 2D array in {path.name}, got shape {array.shape}")

    return array


def discover_feature_files(src_dir: Path) -> list[Path]:
    files = sorted(src_dir.glob("*.npz")) + sorted(src_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npz or .npy files found in {src_dir}")
    return files


def main() -> None:
    args = parse_args()

    src_dir = Path(args.src_dir).resolve()
    out_h5 = Path(args.out_h5).resolve()
    out_index = Path(args.out_index).resolve()

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    out_index.parent.mkdir(parents=True, exist_ok=True)

    files = discover_feature_files(src_dir)
    first = load_feature_array(files[0], args.feature_key, allow_1d=args.allow_1d)
    feature_dim = int(first.shape[1])
    storage_dtype = np.float16 if args.dtype == "float16" else np.float32
    compression = None if args.compression == "none" else args.compression
    compression_opts = args.gzip_level if args.compression == "gzip" else None
    chunk_rows = max(1, min(256, int(first.shape[0])))

    index: dict[str, dict[str, int | str]] = {}
    total_rows = 0

    with h5py.File(out_h5, "w") as h5_file:
        feature_ds = h5_file.create_dataset(
            "features",
            shape=(0, feature_dim),
            maxshape=(None, feature_dim),
            dtype=storage_dtype,
            chunks=(chunk_rows, feature_dim),
            compression=compression,
            compression_opts=compression_opts,
        )

        for path in tqdm(files, desc="Packing features"):
            array = load_feature_array(path, args.feature_key, allow_1d=args.allow_1d)
            if array.shape[1] != feature_dim:
                raise ValueError(
                    f"Feature dim mismatch for {path.name}: expected {feature_dim}, got {array.shape[1]}"
                )

            n_rows = int(array.shape[0])
            feature_ds.resize(total_rows + n_rows, axis=0)
            feature_ds[total_rows : total_rows + n_rows] = array.astype(storage_dtype, copy=False)

            index[path.stem] = {
                "offset": total_rows,
                "num_clips": n_rows,
                "feature_dim": feature_dim,
                "source_file": path.name,
            }
            total_rows += n_rows

        h5_file.attrs["feature_dim"] = feature_dim
        h5_file.attrs["total_videos"] = len(files)
        h5_file.attrs["total_clips"] = total_rows
        h5_file.attrs["storage_dtype"] = np.dtype(storage_dtype).name
        h5_file.attrs["source_dir"] = str(src_dir)

    payload = {
        "source_dir": str(src_dir),
        "out_h5": str(out_h5),
        "total_videos": len(files),
        "total_clips": total_rows,
        "feature_dim": feature_dim,
        "storage_dtype": np.dtype(storage_dtype).name,
        "index": index,
    }

    with open(out_index, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved HDF5 to {out_h5}")
    print(f"Saved index to {out_index}")
    print(f"Videos: {len(files)} | Total clips: {total_rows} | Feature dim: {feature_dim}")


if __name__ == "__main__":
    main()
