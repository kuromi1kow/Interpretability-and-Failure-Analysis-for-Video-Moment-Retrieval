from __future__ import annotations

import json
from pathlib import Path

import h5py


def load_feature_index(index_path: str | Path) -> dict:
    with open(index_path, "r") as f:
        return json.load(f)


class HDF5FeatureStore:
    def __init__(self, h5_path: str | Path, index_path: str | Path):
        self.h5_path = Path(h5_path)
        self.index_path = Path(index_path)
        self._index_payload = load_feature_index(self.index_path)
        self.index = self._index_payload["index"]
        self._h5 = None

    def open(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __enter__(self) -> "HDF5FeatureStore":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.index)

    def keys(self):
        return self.index.keys()

    def get(self, video_id: str):
        self.open()
        meta = self.index[video_id]
        offset = meta["offset"]
        num_clips = meta["num_clips"]
        return self._h5["features"][offset : offset + num_clips]


def load_video_features(h5_path: str | Path, index_path: str | Path, video_id: str):
    with HDF5FeatureStore(h5_path, index_path) as store:
        return store.get(video_id)
