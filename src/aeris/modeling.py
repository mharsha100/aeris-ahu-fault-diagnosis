"""Hybrid inference with explicit, backward-compatible temporal architectures."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
import torch
from torch import nn
from torch.nn import functional as F

from .features import model_inputs


CLASS_NAMES = ["normal", "damper_fault", "cooling_valve_fault", "heating_valve_fault"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    values = np.asarray(logits, dtype="float64") / temperature
    values -= values.max(axis=1, keepdims=True)
    exponents = np.exp(values)
    return exponents / exponents.sum(axis=1, keepdims=True)


def temperature_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(np.log(np.clip(probabilities, 1e-12, 1.0)), temperature)


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    # A one-parameter calibration fit; no model weights or target labels enter it.
    temperatures = np.unique(np.concatenate([np.geomspace(0.25, 4.0, 81), [1.0]]))
    best = min(temperatures, key=lambda value: log_loss(labels, temperature_probabilities(probabilities, float(value)), labels=[0, 1, 2, 3]))
    return float(best)


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        keep = (confidence >= low) & (confidence <= high if index == bins - 1 else confidence < high)
        if keep.any():
            error += float(keep.mean()) * abs(float(correct[keep].mean()) - float(confidence[keep].mean()))
    return float(error)


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype="int64")
    probabilities = np.asarray(probabilities, dtype="float64")
    predictions = probabilities.argmax(axis=1)
    one_hot = np.eye(4)[labels]
    return {
        "windows": len(labels), "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1, 2, 3])),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece_10_bins": expected_calibration_error(labels, probabilities),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2, 3]).tolist(),
        "per_class": classification_report(labels, predictions, labels=[0, 1, 2, 3], target_names=CLASS_NAMES, output_dict=True, zero_division=0),
    }


class CausalConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size, dilation=dilation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(inputs, (self.left_padding, 0)))


class TemporalBlock(nn.Module):
    def __init__(self, input_channels: int, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.first = CausalConv(input_channels, channels, 3, dilation)
        self.second = CausalConv(channels, channels, 3, dilation)
        # Normalize channels at each time step, keeping time ordering causal.
        self.first_norm = nn.LayerNorm(channels)
        self.second_norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(input_channels, channels, 1) if input_channels != channels else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.first(inputs)
        values = self.first_norm(values.transpose(1, 2)).transpose(1, 2)
        values = self.dropout(F.gelu(values))
        values = self.second(values)
        values = self.second_norm(values.transpose(1, 2)).transpose(1, 2)
        return F.gelu(self.dropout(values) + self.skip(inputs))


class TemporalTCN(nn.Module):
    def __init__(self, input_features: int = 10, channels: int = 32,
                 dropout: float = 0.15, dilations: tuple = (1, 2, 4), classes: int = 4):
        super().__init__()
        self.input_features = input_features
        blocks, previous = [], input_features
        for dilation in dilations:
            blocks.append(TemporalBlock(previous, channels, dilation, dropout))
            previous = channels
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(channels, classes)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[2] != self.input_features:
            raise ValueError("Temporal input must have shape (batch, minutes, features)")
        encoded = self.blocks(sequence.transpose(1, 2))
        return self.classifier(encoded[:, :, -1])


class AttentionTemporalTCN(TemporalTCN):
    """V1: learn a weighted summary of all observed time steps in the window.

    The encoder is causal. Pooling uses the complete, already observed window;
    the output describes its endpoint. Attention is not a causal explanation.
    """

    def __init__(self, input_features: int = 10, channels: int = 32,
                 dropout: float = 0.15, dilations: tuple = (1, 2, 4), classes: int = 4):
        super().__init__(input_features, channels, dropout, dilations, classes)
        self.attention_score = nn.Sequential(nn.Linear(channels, channels), nn.Tanh(),
                                             nn.Linear(channels, 1, bias=False))

    def encode_and_pool(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence.ndim != 3 or sequence.shape[2] != self.input_features:
            raise ValueError("Temporal input must have shape (batch, minutes, features)")
        encoded = self.blocks(sequence.transpose(1, 2)).transpose(1, 2)
        weights = torch.softmax(self.attention_score(encoded).squeeze(-1), dim=1)
        pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.encode_and_pool(sequence)
        return self.classifier(pooled)


def create_temporal(kind: str, architecture: dict) -> TemporalTCN:
    if kind == "causal_tcn_last":
        return TemporalTCN(**architecture)
    if kind == "causal_tcn_attention":
        return AttentionTemporalTCN(**architecture)
    raise ValueError(f"Unsupported temporal architecture: {kind}")


def create_forest(seed: int = 42, trees: int = 200, jobs: int = -1) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=trees, max_depth=18, min_samples_leaf=3,
                                  max_features="sqrt", class_weight="balanced_subsample",
                                  bootstrap=True, n_jobs=jobs, random_state=seed)


def neural_logits(model: TemporalTCN, sequence: np.ndarray, device: str | torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    parts = []
    with torch.inference_mode():
        for offset in range(0, len(sequence), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(sequence[offset:offset + batch_size], dtype="float32")).to(device)
            parts.append(model(batch).cpu().numpy())
    if not parts:
        raise ValueError("No temporal inputs to predict")
    return np.concatenate(parts, axis=0)


@dataclass
class HybridPredictor:
    forest: RandomForestClassifier
    temporal: TemporalTCN
    configuration: dict
    scaler: dict
    protocol: dict
    device: str

    def predict(self, X_seq: np.ndarray, X_tab: np.ndarray) -> dict[str, np.ndarray]:
        if len(X_seq) != len(X_tab) or len(X_seq) == 0:
            raise ValueError("Classical and temporal inputs must have matching, nonempty rows")
        expected = (self.protocol["window_minutes"], len(self.protocol["core_features"]))
        if X_seq.ndim != 3 or X_seq.shape[1:] != expected or X_tab.ndim != 2 or X_tab.shape[1] != len(self.protocol["tabular_features"]):
            raise ValueError("Hybrid input shapes do not match the saved feature contract")
        if not np.isfinite(X_seq).all() or not np.isfinite(X_tab).all():
            raise ValueError("Hybrid inputs contain nonfinite values")
        temperatures = self.configuration["temperatures"]
        forest_probs = temperature_probabilities(self.forest.predict_proba(X_tab), temperatures["forest"])
        temporal_probs = softmax(neural_logits(self.temporal, X_seq, self.device), temperatures["temporal"])
        weight = self.configuration["forest_weight"]
        combined = weight * forest_probs + (1 - weight) * temporal_probs
        probabilities = temperature_probabilities(combined, temperatures["hybrid"])
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        disagreement = forest_probs.argmax(axis=1) != temporal_probs.argmax(axis=1)
        review = (confidence < self.configuration["review_confidence_threshold"]) | disagreement
        escalation = (predictions != 0) & ~review
        return {"forest_probabilities": forest_probs, "temporal_probabilities": temporal_probs,
                "probabilities": probabilities, "class_id": predictions, "confidence": confidence,
                "branch_disagreement": disagreement, "needs_review": review, "confident_fault_escalation": escalation}

    def predict_raw(self, raw_windows: np.ndarray) -> dict[str, np.ndarray]:
        sequence, table = model_inputs(raw_windows, self.scaler, self.protocol)
        return self.predict(sequence, table)


def load_hybrid(folder: str | Path, device: str = "cpu") -> HybridPredictor:
    folder = Path(folder)
    if not (folder / "READY.json").is_file():
        raise FileNotFoundError(f"Model bundle is incomplete: {folder}")
    ready = json.loads((folder / "READY.json").read_text(encoding="utf-8"))
    if sha256_file(folder / "hybrid.json") != ready["hybrid_sha256"]:
        raise ValueError("Model configuration changed after the bundle was saved")
    config = json.loads((folder / "hybrid.json").read_text(encoding="utf-8"))
    if config["artifact_version"] != "aeris-hybrid-v1":
        raise ValueError("Unsupported hybrid artifact version")
    required = {"forest.joblib", "temporal_state.pt", "source_scaler.json", "window_protocol.json"}
    if set(config["artifact_sha256"]) != required:
        raise ValueError("The model bundle does not contain the expected artifacts")
    for filename, digest in config["artifact_sha256"].items():
        if sha256_file(folder / filename) != digest:
            raise ValueError(f"Model artifact changed: {filename}")
    forest = joblib.load(folder / "forest.joblib")
    if not np.array_equal(forest.classes_, np.arange(4)):
        raise ValueError("Forest class order does not match AERIS class IDs")
    temporal = create_temporal(config.get("temporal_kind", "causal_tcn_last"), config["temporal_architecture"])
    state = torch.load(folder / "temporal_state.pt", map_location="cpu", weights_only=True)
    temporal.load_state_dict(state)
    temporal.to(device).eval()
    scaler = json.loads((folder / "source_scaler.json").read_text(encoding="utf-8"))
    protocol = json.loads((folder / "window_protocol.json").read_text(encoding="utf-8"))
    return HybridPredictor(forest, temporal, config, scaler, protocol, device)
