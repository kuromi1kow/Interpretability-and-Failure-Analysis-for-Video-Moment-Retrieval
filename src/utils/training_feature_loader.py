from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.utils.hdf5_features import HDF5FeatureStore


_HDF5_STORE_CACHE: dict[str, HDF5FeatureStore] = {}


def l2_normalize_np_array(array: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(array, axis=-1, keepdims=True)
    denom = np.clip(denom, a_min=1e-12, a_max=None)
    return array / denom


def infer_index_path(h5_path: str | Path) -> Path:
    h5_path = Path(h5_path)
    if h5_path.name.endswith("_features.h5"):
        return h5_path.with_name(h5_path.name.replace("_features.h5", "_features_index.json"))
    return h5_path.with_suffix(".json")


def get_hdf5_store(h5_path: str | Path, index_path: str | Path | None = None) -> HDF5FeatureStore:
    h5_path = Path(h5_path).resolve()
    if index_path is None:
        index_path = infer_index_path(h5_path)
    else:
        index_path = Path(index_path).resolve()

    cache_key = f"{h5_path}::{index_path}"
    store = _HDF5_STORE_CACHE.get(cache_key)
    if store is None:
        store = HDF5FeatureStore(h5_path, index_path)
        store.open()
        _HDF5_STORE_CACHE[cache_key] = store
    return store


def load_feature_array(source: str | Path, video_id: str, max_v_l: int | None = None, normalize: bool = True,
                       index_path: str | Path | None = None) -> np.ndarray:
    source = Path(source)

    if source.suffix == ".h5":
        store = get_hdf5_store(source, index_path=index_path)
        array = np.asarray(store.get(video_id), dtype=np.float32)
    else:
        feat_path = source / f"{video_id}.npz"
        array = np.load(feat_path)["features"].astype(np.float32)

    if max_v_l is not None:
        array = array[:max_v_l]

    if normalize:
        array = l2_normalize_np_array(array)

    return array


def load_video_features_for_training(
    video_id: str,
    sources: list[str] | list[Path] | str | Path,
    max_v_l: int | None = None,
    normalize: bool = True,
    index_paths: list[str | Path | None] | None = None,
) -> torch.Tensor:
    if isinstance(sources, (str, Path)):
        sources = [sources]

    if index_paths is None:
        index_paths = [None] * len(sources)

    arrays = []
    for source, index_path in zip(sources, index_paths):
        arrays.append(
            load_feature_array(
                source=source,
                video_id=video_id,
                max_v_l=max_v_l,
                normalize=normalize,
                index_path=index_path,
            )
        )

    min_len = min(len(array) for array in arrays)
    arrays = [array[:min_len] for array in arrays]
    merged = np.concatenate(arrays, axis=1)
    return torch.from_numpy(merged)
