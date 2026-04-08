#!/usr/bin/env python3
"""
Merge multiple HDF5 feature stores by key and concatenate aligned features.

Usage:
  python scripts/merge_hdf5_features.py \
      --inputs data/qvhighlights/hdf5/clip_features.h5 data/qvhighlights/hdf5/slowfast_features.h5 \
      --out-h5 data/qvhighlights/hdf5/clip_slowfast_features.h5 \
      --out-index data/qvhighlights/hdf5/clip_slowfast_features_index.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.hdf5_features import HDF5FeatureStore
from src.utils.training_feature_loader import infer_index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Input HDF5 feature stores")
    parser.add_argument("--out-h5", required=True, help="Output merged HDF5 file path")
    parser.add_argument("--out-index", required=True, help="Output merged JSON index path")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Storage dtype inside merged HDF5",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_h5_paths = [Path(path).resolve() for path in args.inputs]
    out_h5 = Path(args.out_h5).resolve()
    out_index = Path(args.out_index).resolve()
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    out_index.parent.mkdir(parents=True, exist_ok=True)

    compression = None if args.compression == "none" else args.compression
    compression_opts = args.gzip_level if args.compression == "gzip" else None
    storage_dtype = np.float16 if args.dtype == "float16" else np.float32

    stores = []
    for h5_path in input_h5_paths:
        store = HDF5FeatureStore(h5_path, infer_index_path(h5_path))
        store.open()
        stores.append(store)

    try:
        shared_keys = sorted(set.intersection(*(set(store.keys()) for store in stores)))
        if not shared_keys:
            raise ValueError("No shared keys were found across the provided HDF5 stores.")

        first_arrays = [np.asarray(store.get(shared_keys[0]), dtype=np.float32) for store in stores]
        first_min_len = min(len(array) for array in first_arrays)
        first_merged = np.concatenate([array[:first_min_len] for array in first_arrays], axis=1)
        chunk_rows = max(1, min(256, int(first_merged.shape[0])))
        feature_dim = int(first_merged.shape[1])

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

            for key in tqdm(shared_keys, desc="Merging stores"):
                arrays = [np.asarray(store.get(key), dtype=np.float32) for store in stores]
                min_len = min(len(array) for array in arrays)
                merged = np.concatenate([array[:min_len] for array in arrays], axis=1)

                n_rows = int(merged.shape[0])
                feature_ds.resize(total_rows + n_rows, axis=0)
                feature_ds[total_rows : total_rows + n_rows] = merged.astype(storage_dtype, copy=False)

                index[key] = {
                    "offset": total_rows,
                    "num_clips": n_rows,
                    "feature_dim": feature_dim,
                }
                total_rows += n_rows

            h5_file.attrs["feature_dim"] = feature_dim
            h5_file.attrs["total_videos"] = len(shared_keys)
            h5_file.attrs["total_clips"] = total_rows
            h5_file.attrs["storage_dtype"] = np.dtype(storage_dtype).name
            h5_file.attrs["sources"] = json.dumps([str(path) for path in input_h5_paths])

        payload = {
            "sources": [str(path) for path in input_h5_paths],
            "out_h5": str(out_h5),
            "total_videos": len(shared_keys),
            "total_clips": total_rows,
            "feature_dim": feature_dim,
            "storage_dtype": np.dtype(storage_dtype).name,
            "index": index,
        }

        with open(out_index, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"Saved merged HDF5 to {out_h5}")
        print(f"Saved merged index to {out_index}")
        print(f"Keys: {len(shared_keys)} | Total clips: {total_rows} | Feature dim: {feature_dim}")
    finally:
        for store in stores:
            store.close()


if __name__ == "__main__":
    main()
