#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.hdf5_features import HDF5FeatureStore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--video-id", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with HDF5FeatureStore(args.h5, args.index) as store:
        features = store.get(args.video_id)
        print(f"video_id: {args.video_id}")
        print(f"shape: {features.shape}")
        print(f"dtype: {features.dtype}")


if __name__ == "__main__":
    main()
