"""Use the same window representations as the frozen script-03 protocol."""

from __future__ import annotations

import numpy as np


def tabular_features(windows: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    windows = np.asarray(windows, dtype="float32")
    if windows.ndim != 3 or windows.shape[1] < 2 or windows.shape[2] != len(names):
        raise ValueError("Expected windows with shape (N, minutes, core_features)")
    statistics = [
        ("mean", windows.mean(axis=1)), ("std", windows.std(axis=1)),
        ("min", windows.min(axis=1)), ("max", windows.max(axis=1)),
        ("last", windows[:, -1, :]),
        ("trend_per_minute", (windows[:, -1, :] - windows[:, 0, :]) / (windows.shape[1] - 1)),
    ]
    arrays = [values for _, values in statistics]
    columns = [f"{name}__{statistic}" for statistic, _ in statistics for name in names]
    for left, right, name in [
        ("supply_temp_c", "mixed_temp_c", "supply_minus_mixed_c"),
        ("mixed_temp_c", "return_temp_c", "mixed_minus_return_c"),
        ("outdoor_temp_c", "return_temp_c", "outdoor_minus_return_c"),
    ]:
        delta = windows[:, :, names.index(left)] - windows[:, :, names.index(right)]
        arrays.extend([delta.mean(axis=1, keepdims=True), delta[:, -1:]])
        columns.extend([f"{name}__mean", f"{name}__last"])
    return np.concatenate(arrays, axis=1).astype("float32"), columns


def model_inputs(raw_windows: np.ndarray, scaler: dict, protocol: dict) -> tuple[np.ndarray, np.ndarray]:
    """Canonical Celsius/fraction windows -> temporal and classical inputs."""
    windows = np.asarray(raw_windows, dtype="float32")
    if windows.ndim == 2:
        windows = windows[None, :, :]
    features = protocol["core_features"]
    expected = (protocol["window_minutes"], len(features))
    if windows.ndim != 3 or windows.shape[1:] != expected or len(windows) == 0:
        raise ValueError(f"Expected raw windows shaped (N, {expected[0]}, {expected[1]})")
    if not np.isfinite(windows).all():
        raise ValueError("Required window inputs contain missing or nonfinite values")
    for name in ["fan_status", "occupied"]:
        if not np.all(windows[:, :, features.index(name)] == 1):
            raise ValueError("The first model supports windows that are occupied with the fan running")
    table, table_names = tabular_features(windows, features)
    if table_names != protocol["tabular_features"] or scaler["features"] != features:
        raise ValueError("Saved model feature order does not match the input contract")
    mean = np.asarray(scaler["mean"], dtype="float64")
    scale = np.asarray(scaler["scale"], dtype="float64")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid source normalization parameters")
    sequence = ((windows.astype("float64") - mean) / scale).astype("float32")
    return sequence, table
