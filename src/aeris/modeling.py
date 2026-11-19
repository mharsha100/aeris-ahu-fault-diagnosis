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


class CombinedTemporalTCN(AttentionTemporalTCN):
    """V2 combines the final encoded state and attention-pooled history."""

    def __init__(self, input_features: int = 10, channels: int = 32,
                 dropout: float = 0.15, dilations: tuple = (1, 2, 4), classes: int = 4):
        super().__init__(input_features, channels, dropout, dilations, classes)
        self.classifier = nn.Sequential(nn.Linear(2 * channels, channels), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(channels, classes))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[2] != self.input_features:
            raise ValueError("Temporal input must have shape (batch, minutes, features)")
        encoded = self.blocks(sequence.transpose(1, 2)).transpose(1, 2)
        attention = torch.softmax(self.attention_score(encoded).squeeze(-1), dim=1)
        pooled = (encoded * attention.unsqueeze(-1)).sum(dim=1)
        return self.classifier(torch.cat([encoded[:, -1, :], pooled], dim=1))


class PhysicalTemporal(nn.Module):
    """Retain each temporal pooling design and add a common physical context head."""
    def __init__(self,kind,raw_mean,raw_scale,context_width=64,**architecture):
        super().__init__();self.base=create_temporal(kind,architecture)
        self.register_buffer('raw_mean',torch.tensor(raw_mean,dtype=torch.float32))
        self.register_buffer('raw_scale',torch.tensor(raw_scale,dtype=torch.float32))
        self.context=nn.Sequential(nn.Linear(54,context_width),nn.GELU(),nn.Dropout(architecture['dropout']),nn.Linear(context_width,4))
    def forward(self,sequence):
        raw=sequence*self.raw_scale+self.raw_mean
        pieces=[sequence.mean(1),sequence[:,-1],sequence.std(1,unbiased=False)]
        for x in [raw.mean(1),raw[:,-1]]:
            s,o,m,r=x[:,0],x[:,1],x[:,2],x[:,3];d,c,h=x[:,6],x[:,7],x[:,8]
            span=o-r;safe=torch.where(span.abs()>=2,span,torch.where(span<0,-2.,2.))
            coil=(s-m)/10;ratio=((m-r)/safe).clamp(-2,3)
            pieces.append(torch.stack([coil,(s-r)/10,(m-r)/10,span/30,(m-(d*o+(1-d)*r))/10,ratio,ratio-d,coil*c,coil*h,c*h,coil*(1-c),coil*(1-h)],dim=1))
        return self.base(sequence)+self.context(torch.cat(pieces,dim=1))


def create_temporal(kind: str, architecture: dict) -> TemporalTCN:
    if kind.startswith('physical_'):
        return PhysicalTemporal(kind[len('physical_'):],**architecture)
    if kind == "causal_tcn_last":
        return TemporalTCN(**architecture)
    if kind == "causal_tcn_attention":
        return AttentionTemporalTCN(**architecture)
    if kind == "causal_tcn_combined":
        return CombinedTemporalTCN(**architecture)
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
        factors=np.asarray(self.configuration.get('class_factors',[1.,1.,1.,1.]),dtype=float)
        combined=combined*factors;combined=combined/combined.sum(axis=1,keepdims=True)
        probabilities = temperature_probabilities(combined, temperatures["hybrid"])
        parent_probabilities=probabilities.copy()
        boost_config=self.configuration.get('boosting')
        boosted_probs=None
        if boost_config:
            boosted_probs=temperature_probabilities(self.forest.boosted_probabilities(X_seq,X_tab),boost_config['temperature'])
            w=boost_config['weight'];refined=((1-w)*probabilities+w*boosted_probs)*np.asarray(boost_config['class_factors'])
            refined=refined/refined.sum(axis=1,keepdims=True)
            probabilities=temperature_probabilities(refined,boost_config['hybrid_temperature'])
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        disagreement = forest_probs.argmax(axis=1) != temporal_probs.argmax(axis=1)
        if boosted_probs is not None:
            disagreement=disagreement|(boosted_probs.argmax(axis=1)!=temporal_probs.argmax(axis=1))|(predictions!=temporal_probs.argmax(axis=1))
        refinement_review = ((parent_probabilities.max(axis=1)<self.configuration['review_confidence_threshold']) | (parent_probabilities.argmax(axis=1)!=predictions)) if boost_config else np.zeros(len(predictions),dtype=bool)
        review = (confidence < self.configuration["review_confidence_threshold"]) | disagreement | refinement_review
        escalation = (predictions != 0) & ~review
        result={"forest_probabilities": forest_probs, "temporal_probabilities": temporal_probs,
                "probabilities": probabilities, "class_id": predictions, "confidence": confidence,
                "branch_disagreement": disagreement, "needs_review": review, "confident_fault_escalation": escalation}
        if boosted_probs is not None:result['boosted_probabilities']=boosted_probs;result['refinement_review']=refinement_review
        return result

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
    boost=config.get('boosting')
    if boost:
        factors=np.asarray(boost.get('class_factors'),dtype=float)
        if not 0<boost.get('weight',0)<1 or factors.shape!=(4,) or not np.isfinite(factors).all() or np.any(factors<=0):raise ValueError('Invalid boosted refinement factors or weight')
        if any(not np.isfinite(boost.get(k,0)) or boost.get(k,0)<=0 for k in ['temperature','hybrid_temperature']):raise ValueError('Invalid boosted calibration temperatures')
    if config["artifact_version"] != "aeris-hybrid-v1":
        raise ValueError("Unsupported hybrid artifact version")
    if config.get('classes') != CLASS_NAMES or not 0 < config.get('forest_weight', 0) < 1:
        raise ValueError('AERIS requires the ordered four classes and both RF and temporal branches')
    factors=np.asarray(config.get('class_factors',[1.,1.,1.,1.]),dtype=float)
    if factors.shape!=(4,) or not np.isfinite(factors).all() or np.any(factors<=0):
        raise ValueError('Class factors must be four finite positive values')
    if set(config.get('temperatures', {})) != {'forest','temporal','hybrid'} or any(not np.isfinite(t) or t <= 0 for t in config['temperatures'].values()):
        raise ValueError('All branch and hybrid calibration temperatures must be finite and positive')
    if not 0 < config.get('review_confidence_threshold', 0) < 1:
        raise ValueError('Invalid review confidence threshold')
    required = {"forest.joblib", "temporal_state.pt", "source_scaler.json", "window_protocol.json"}
    if set(config["artifact_sha256"]) != required:
        raise ValueError("The model bundle does not contain the expected artifacts")
    for filename, digest in config["artifact_sha256"].items():
        if sha256_file(folder / filename) != digest:
            raise ValueError(f"Model artifact changed: {filename}")
    forest = joblib.load(folder / "forest.joblib")
    if boost and (not hasattr(forest,'boosted_probabilities') or not np.array_equal(forest.booster.classes_,np.arange(4))):raise ValueError('Boosted artifact does not match its declared contract')
    if not np.array_equal(forest.classes_, np.arange(4)):
        raise ValueError("Forest class order does not match AERIS class IDs")
    temporal = create_temporal(config.get("temporal_kind", "causal_tcn_last"), config["temporal_architecture"])
    state = torch.load(folder / "temporal_state.pt", map_location="cpu", weights_only=True)
    temporal.load_state_dict(state)
    temporal.to(device).eval()
    scaler = json.loads((folder / "source_scaler.json").read_text(encoding="utf-8"))
    protocol = json.loads((folder / "window_protocol.json").read_text(encoding="utf-8"))
    return HybridPredictor(forest, temporal, config, scaler, protocol, device)
