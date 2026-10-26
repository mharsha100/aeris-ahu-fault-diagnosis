"""Fresh hybrid architecture training; shared source protocol for V0 and V1."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import time
import zipfile
from datetime import datetime, timezone
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .modeling import CLASS_NAMES, HybridPredictor, TemporalTCN, create_temporal, create_forest, fit_temperature, load_hybrid, metrics, neural_logits, sha256_file, softmax, temperature_probabilities


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=lambda item: item.item() if isinstance(item, np.generic) else str(item)), encoding="utf-8")


def environment_check(requested_device: str = "cuda", temporal_kind: str = "causal_tcn_last") -> dict:
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment. Verify the cu128 PyTorch installation before GPU training.")
    device = "cuda" if requested_device == "cuda" else "cpu"
    # Exercise both the device runtime and the actual temporal network operators.
    probe = create_temporal(temporal_kind, {}).to(device).eval()
    with torch.inference_mode():
        result = probe(torch.zeros((2, 15, 10), device=device))
    if not torch.isfinite(result).all().item():
        raise RuntimeError("The temporal device smoke check returned nonfinite values")
    report = {"python": platform.python_version(), "python_executable": str(Path(os.sys.executable)),
              "device": device, "cuda_available": bool(torch.cuda.is_available()),
              "torch_cuda_runtime": torch.version.cuda, "device_test": "passed", "packages": {}}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "torch"]:
        report["packages"][package] = importlib.metadata.version(package)
    if device == "cuda":
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        report["cudnn_version"] = torch.backends.cudnn.version()
    del probe
    return report


def load_source_partition(windows_root: Path, partition: str, protocol: dict) -> dict:
    # Test/adaptation/feedback files cannot be passed to the fitting loader.
    if partition not in {"source_train", "source_validation", "source_calibration"}:
        raise ValueError("Initial hybrid fitting only accepts source train/validation/calibration partitions")
    with np.load(windows_root / f"{partition}.npz", allow_pickle=False) as stored:
        pack = {name: stored[name].copy() for name in stored.files}
    for key in ["X_seq", "X_tab", "y", "window_id"]:
        if key not in pack:
            raise ValueError(f"Missing {key} in {partition}")
    n = len(pack["y"])
    if pack["X_seq"].shape != (n, protocol["window_minutes"], len(protocol["core_features"])):
        raise ValueError(f"Wrong temporal shape in {partition}")
    if pack["X_tab"].shape != (n, len(protocol["tabular_features"])) or len(pack["window_id"]) != n:
        raise ValueError(f"Wrong classical shape or alignment in {partition}")
    if set(np.unique(pack["y"])) != {0, 1, 2, 3} or len(set(pack["window_id"])) != n:
        raise ValueError(f"Missing classes or duplicate windows in {partition}")
    if not np.isfinite(pack["X_seq"]).all() or not np.isfinite(pack["X_tab"]).all():
        raise ValueError(f"Nonfinite model inputs in {partition}")
    pack["X_seq"] = pack["X_seq"].astype("float32", copy=False)
    pack["X_tab"] = pack["X_tab"].astype("float32", copy=False)
    pack["y"] = pack["y"].astype("int64", copy=False)
    return pack


def train_temporal(model: TemporalTCN, train: dict, validation: dict, *, device: str,
                   epochs: int, batch_size: int, patience: int, seed: int) -> tuple[list[dict], int]:
    class_counts = np.bincount(train["y"], minlength=4)
    weights = len(train["y"]) / (4 * class_counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    dataset = TensorDataset(torch.from_numpy(train["X_seq"]), torch.from_numpy(train["y"]))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator,
                        num_workers=0, pin_memory=device == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=0.00001)
    best_score, best_loss, stale, best_epoch = -1.0, float("inf"), 0, 0
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        total_loss, total_rows = 0.0, 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=device == "cuda")
            y_batch = y_batch.to(device, non_blocking=device == "cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X_batch), y_batch)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Temporal training produced a nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(y_batch)
            total_rows += len(y_batch)
        probabilities = softmax(neural_logits(model, validation["X_seq"], device))
        score = float(f1_score(validation["y"], probabilities.argmax(axis=1), labels=[0, 1, 2, 3], average="macro", zero_division=0))
        validation_loss = float(log_loss(validation["y"], probabilities, labels=[0, 1, 2, 3]))
        improved = score > best_score + 0.0001 or (abs(score - best_score) <= 0.0001 and validation_loss < best_loss - 0.0001)
        if improved:
            best_score, best_loss, stale, best_epoch = score, validation_loss, 0, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        scheduler.step(score)
        row = {"epoch": epoch, "train_weighted_loss": total_loss / total_rows,
               "validation_macro_f1": score, "validation_log_loss": validation_loss,
               "learning_rate": optimizer.param_groups[0]["lr"],
               "seconds": time.perf_counter() - started, "selected_checkpoint": improved}
        history.append(row)
        print(f"Epoch {epoch:02d}/{epochs} | loss={row['train_weighted_loss']:.4f} | val macro F1={score:.4f} | {row['seconds']:.1f}s", flush=True)
        if patience > 0 and epoch >= min(10, epochs) and stale >= patience:
            print(f"Early stopping; restoring epoch {best_epoch}.", flush=True)
            break
    if best_state is None:
        raise RuntimeError("No valid temporal checkpoint was produced")
    model.load_state_dict(best_state)
    model.to(device).eval()
    return history, best_epoch


def train_source_hybrid(project_root: Path, *, model_version: str = "v0", device: str = "cuda", epochs: int = 50,
             batch_size: int = 256, patience: int = 8, seed: int = 42,
             trees: int = 200, jobs: int = -1, overwrite: bool = False) -> Path:
    if model_version not in {"v0", "v1"}:
        raise ValueError("Only the implemented V0 and V1 architectures can be trained")
    if epochs < 1 or batch_size < 1 or patience < 0 or trees < 1:
        raise ValueError("Training counts must be positive; patience=0 disables early stopping")
    temporal_kind = "causal_tcn_last" if model_version == "v0" else "causal_tcn_attention"
    architecture_id = "rf_tcn_last_v0" if model_version == "v0" else "rf_tcn_attention_v1"
    bundle_name = "v0" if model_version == "v0" else "v1_attention"
    windows_root = project_root / "data" / "aeris_windows_v1"
    output = project_root / "data" / "aeris_models" / bundle_name
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Model output already exists: {output}. Existing results are protected.")
    if not (windows_root / "READY.json").exists():
        raise FileNotFoundError("Complete script 03 before starting model training")
    ready = json.loads((windows_root / "READY.json").read_text(encoding="utf-8"))
    if sha256_file(windows_root / "window_protocol.json") != ready["protocol_sha256"]:
        raise ValueError("The frozen window protocol changed after preparation")
    protocol = json.loads((windows_root / "window_protocol.json").read_text(encoding="utf-8"))
    if protocol["protocol_version"] != "aeris-windows-v1":
        raise ValueError("Unsupported window protocol")
    if protocol["window_minutes"] != 15:
        raise ValueError("The first reviewed training configuration expects 15-minute windows")
    baseline_manifest_sha = None
    if model_version == "v1":
        # Read provenance only. Neither V0 forest nor V0 neural weights is loaded.
        baseline = project_root / "data/aeris_models/v0"
        baseline_config = json.loads((baseline / "hybrid.json").read_text(encoding="utf-8"))
        baseline_ready = json.loads((baseline / "READY.json").read_text(encoding="utf-8"))
        baseline_manifest_sha = sha256_file(baseline / "hybrid.json")
        if baseline_manifest_sha != baseline_ready["hybrid_sha256"] or baseline_config["model_version"] != "v0":
            raise ValueError("V0 provenance changed or does not describe the baseline")
        for name in ["window_protocol.json", "source_scaler.json"]:
            if sha256_file(windows_root / name) != baseline_config["artifact_sha256"][name]:
                raise ValueError(f"{name} differs from the V0 comparison protocol")
        baseline_run = json.loads((baseline / "training_run.json").read_text(encoding="utf-8"))
        expected_roles = {"source_train", "source_validation", "source_calibration"}
        if set(baseline_run["source_pack_sha256"]) != expected_roles:
            raise ValueError("V0 source pack provenance is incomplete")
        for role, digest in baseline_run["source_pack_sha256"].items():
            if sha256_file(windows_root / f"{role}.npz") != digest:
                raise ValueError(f"{role} differs from V0; keep the frozen comparison split")
    environment = environment_check(device, temporal_kind)
    print(f"Architecture: {architecture_id} | fresh RF fit + fresh neural initialization", flush=True)
    print(f"Temporal budget: {epochs} epochs | early stopping: {'disabled' if patience == 0 else patience}", flush=True)
    print(f"Training device: {environment.get('gpu', device)} | PyTorch {environment['packages']['torch']}", flush=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    train = load_source_partition(windows_root, "source_train", protocol)
    validation = load_source_partition(windows_root, "source_validation", protocol)
    calibration = load_source_partition(windows_root, "source_calibration", protocol)
    sets = [set(pack["window_id"]) for pack in [train, validation, calibration]]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Fitting, validation and calibration windows overlap")
    print(f"Source windows: train={len(train['y']):,}, validation={len(validation['y']):,}, calibration={len(calibration['y']):,}", flush=True)
    print("[1/3] Training the classical Random Forest...", flush=True)
    forest_started = time.perf_counter()
    forest = create_forest(seed=seed, trees=trees, jobs=jobs)
    forest.fit(train["X_tab"], train["y"])
    forest_seconds = time.perf_counter() - forest_started
    if not np.array_equal(forest.classes_, np.arange(4)):
        raise ValueError("Unexpected forest class order")
    print(f"Random Forest finished in {forest_seconds:.1f}s.", flush=True)
    print("[2/3] Training the temporal neural network...", flush=True)
    architecture = {"input_features": len(protocol["core_features"]), "channels": 32,
                    "dropout": 0.15, "dilations": [1, 2, 4], "classes": 4}
    temporal = create_temporal(temporal_kind, architecture).to(device)
    history, best_epoch = train_temporal(temporal, train, validation, device=device, epochs=epochs,
                                        batch_size=batch_size, patience=patience, seed=seed)
    print("[3/3] Selecting fusion weight and fitting source-only calibration...", flush=True)
    raw_val_forest = forest.predict_proba(validation["X_tab"])
    raw_cal_forest = forest.predict_proba(calibration["X_tab"])
    raw_val_temporal = softmax(neural_logits(temporal, validation["X_seq"], device))
    raw_cal_temporal = softmax(neural_logits(temporal, calibration["X_seq"], device))
    forest_temperature = fit_temperature(raw_cal_forest, calibration["y"])
    temporal_temperature = fit_temperature(raw_cal_temporal, calibration["y"])
    val_forest = temperature_probabilities(raw_val_forest, forest_temperature)
    val_temporal = temperature_probabilities(raw_val_temporal, temporal_temperature)
    candidates = []
    for weight in [0.25, 0.5, 0.75]:
        mixed = weight * val_forest + (1 - weight) * val_temporal
        result = metrics(validation["y"], mixed)
        candidates.append({"forest_weight": weight, "temporal_weight": 1 - weight,
                           "validation_macro_f1": result["macro_f1"], "validation_log_loss": result["log_loss"]})
    selected = max(candidates, key=lambda row: (row["validation_macro_f1"], -row["validation_log_loss"], -abs(row["forest_weight"] - 0.5)))
    weight = selected["forest_weight"]
    mixed_calibration = weight * temperature_probabilities(raw_cal_forest, forest_temperature) + (1 - weight) * temperature_probabilities(raw_cal_temporal, temporal_temperature)
    hybrid_temperature = fit_temperature(mixed_calibration, calibration["y"])
    config = {"artifact_version": "aeris-hybrid-v1", "model_version": model_version,
              "architecture_id": architecture_id, "temporal_kind": temporal_kind,
              "learning_stage": "initial_source_training", "initialization": "fresh",
              "parent_weights_loaded": False,
              "comparison_baseline_hybrid_sha256": baseline_manifest_sha,
              "classes": CLASS_NAMES, "temporal_architecture": architecture,
              "forest_weight": weight, "temporal_weight": 1 - weight,
              "temperatures": {"forest": forest_temperature, "temporal": temporal_temperature, "hybrid": hybrid_temperature},
              "review_confidence_threshold": 0.9,
              "review_policy": "Review below the provisional confidence threshold or when the two branches disagree. This is not a validated target error guarantee.",
              "training_scope": protocol["scope"],
              "training_partitions": ["source_train", "source_validation", "source_calibration"],
              "target_data_used": False, "test_data_used": False,
              "selected_temporal_epoch": best_epoch, "seed": seed,
              "created_utc": datetime.now(timezone.utc).isoformat(), "artifact_sha256": {}}
    scaler = json.loads((windows_root / "source_scaler.json").read_text(encoding="utf-8"))
    predictor = HybridPredictor(forest, temporal, config, scaler, protocol, device)
    predictions = predictor.predict(validation["X_seq"], validation["X_tab"])
    detailed = {name: metrics(validation["y"], probabilities) for name, probabilities in [
        ("classical", predictions["forest_probabilities"]), ("temporal", predictions["temporal_probabilities"]),
        (f"hybrid_{model_version}", predictions["probabilities"])]}
    accepted = ~predictions["needs_review"]
    decision = {"provisional_review_threshold": 0.9, "review_windows": int((~accepted).sum()),
                "accepted_windows": int(accepted.sum()), "accepted_coverage": float(accepted.mean()),
                "accepted_accuracy": float(accuracy_score(validation["y"][accepted], predictions["class_id"][accepted])) if accepted.any() else None,
                "branch_disagreement_windows": int(predictions["branch_disagreement"].sum())}
    # Source validation is development evidence. Final test packs stay unopened.
    meta = pd.read_csv(windows_root / "window_metadata.csv.gz")
    meta = meta.loc[meta["partition"].eq("source_validation")].set_index("window_id")
    if set(meta.index) != set(validation["window_id"]):
        raise ValueError("Source-validation metadata does not match model inputs")
    prediction_table = meta.loc[validation["window_id"]].reset_index()
    prediction_table["true_class_id"] = validation["y"]
    prediction_table["hybrid_class_id"] = predictions["class_id"]
    prediction_table["confidence"] = predictions["confidence"]
    prediction_table["needs_review"] = predictions["needs_review"]
    prediction_table["branch_disagreement"] = predictions["branch_disagreement"]
    per_day = []
    for day, indices in prediction_table.groupby("condition_group", sort=True).groups.items():
        ix = np.asarray(list(indices))
        labels = validation["y"][ix]
        probs = predictions["probabilities"][ix]
        per_day.append({"condition_group": day, "windows": len(ix),
                        "observed_class_ids": ",".join(map(str, np.unique(labels))),
                        "accuracy": float(accuracy_score(labels, probs.argmax(axis=1))),
                        "macro_f1_present_classes": float(f1_score(labels, probs.argmax(axis=1), labels=np.unique(labels), average="macro", zero_division=0))})

    with TemporaryDirectory(prefix=f"aeris_{model_version}_", dir=project_root / "data") as temp:
        stage = Path(temp)
        joblib.dump(forest, stage / "forest.joblib", compress=3)
        torch.save({name: value.detach().cpu() for name, value in temporal.state_dict().items()}, stage / "temporal_state.pt")
        shutil.copyfile(windows_root / "source_scaler.json", stage / "source_scaler.json")
        shutil.copyfile(windows_root / "window_protocol.json", stage / "window_protocol.json")
        config["artifact_sha256"] = {name: sha256_file(stage / name) for name in [
            "forest.joblib", "temporal_state.pt", "source_scaler.json", "window_protocol.json"]}
        write_json(stage / "hybrid.json", config)
        write_json(stage / "READY.json", {"model_version": model_version, "architecture_id": architecture_id,
                   "hybrid_sha256": sha256_file(stage / "hybrid.json")})
        restored = load_hybrid(stage, device=device)
        before = predictor.predict(validation["X_seq"][:32], validation["X_tab"][:32])["probabilities"]
        after = restored.predict(validation["X_seq"][:32], validation["X_tab"][:32])["probabilities"]
        reload_difference = float(np.max(np.abs(before - after)))
        if not np.allclose(before, after, atol=0.00001, rtol=0.00001):
            raise RuntimeError("Saved/reloaded hybrid predictions differ from the trained model")
        pd.DataFrame(history).to_csv(stage / "training_history.csv", index=False)
        pd.DataFrame(candidates).to_csv(stage / "fusion_candidates.csv", index=False)
        write_json(stage / "source_validation_metrics.json", detailed)
        metric_rows = [{"component": name, **{key: value for key, value in values.items() if key not in {"confusion_matrix", "per_class"}}} for name, values in detailed.items()]
        pd.DataFrame(metric_rows).to_csv(stage / "source_validation_metrics.csv", index=False)
        prediction_table.to_csv(stage / "source_validation_predictions.csv.gz", index=False, compression="gzip")
        pd.DataFrame(per_day).to_csv(stage / "source_validation_by_day.csv", index=False)
        pd.DataFrame({"feature": protocol["tabular_features"], "global_impurity_importance": forest.feature_importances_}).sort_values("global_impurity_importance", ascending=False).to_csv(stage / "forest_global_importance.csv", index=False)
        write_json(stage / "source_validation_review_policy.json", decision)
        write_json(stage / "training_environment.json", environment)
        run = {"model_version": model_version, "architecture_id": architecture_id,
               "learning_stage": "initial_source_training", "initialization": "fresh",
               "parent_weights_loaded": False, "completed_temporal_epochs": len(history),
               "early_stopping_enabled": patience > 0,
               "comparison_baseline_hybrid_sha256": baseline_manifest_sha,
               "training_windows": len(train["y"]), "validation_windows": len(validation["y"]),
               "calibration_windows": len(calibration["y"]), "best_temporal_epoch": best_epoch,
               "temporal_parameter_count": sum(parameter.numel() for parameter in temporal.parameters()),
               "forest_fit_seconds": forest_seconds, "seed": seed,
               "forest_parameters": forest.get_params(), "temporal_training": {"maximum_epochs": epochs, "batch_size": batch_size, "patience": patience, "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001, "class_weighted_loss": True},
               "window_protocol_sha256": sha256_file(windows_root / "window_protocol.json"),
               "source_pack_sha256": {role: sha256_file(windows_root / f"{role}.npz") for role in ["source_train", "source_validation", "source_calibration"]},
               "source_only": True, "final_tests_opened": False, "saved_reload_max_abs_difference": reload_difference,
               "notes": ["Validation results were used for model/fusion choices and are not final generalization estimates.",
                         "Calibration parameters use source calibration days only; target calibration is not guaranteed.",
                         "The review confidence threshold is provisional. Branch agreement does not establish label correctness.",
                         "Global forest importances are not local or causal explanations."]}
        write_json(stage / "training_run.json", run)
        lines = [f"AERIS - hybrid {model_version.upper()} training complete", f"Architecture: {architecture_id}",
                 "Fresh Random Forest and temporal network trained; no parent weights loaded.",
                 f"Completed temporal epochs: {len(history)} / {epochs}", f"Device: {environment.get('gpu', device)}",
                 f"Training windows: {len(train['y']):,}", f"Best temporal epoch: {best_epoch}",
                 f"Fusion: {weight:.2f} Random Forest + {1-weight:.2f} temporal network", "", "Source validation macro F1:"]
        lines += [f"  {name}: {values['macro_f1']:.4f}" for name, values in detailed.items()]
        lines += ["", "Saved/reloaded prediction check: passed", "Target adaptation, feedback and final test data were not used.",
                  f"Saved model: {output}", f"Review attachment: {model_version}_review.zip (reports only)",
                  "Next: review architecture training, then run cross-setting adaptation within AERIS."]
        summary = "\n".join(lines) + "\n"
        (stage / f"{model_version}_summary.txt").write_text(summary, encoding="utf-8")
        review_files = [f"{model_version}_summary.txt", "training_run.json", "training_environment.json", "training_history.csv",
                        "fusion_candidates.csv", "source_validation_metrics.json", "source_validation_metrics.csv",
                        "source_validation_by_day.csv", "source_validation_predictions.csv.gz",
                        "forest_global_importance.csv", "source_validation_review_policy.json", "hybrid.json"]
        with zipfile.ZipFile(stage / f"{model_version}_review.zip", "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name in review_files:
                bundle.write(stage / name, arcname=name)
        output.mkdir(parents=True, exist_ok=True)
        (output / "READY.json").unlink(missing_ok=True)
        for path in sorted(stage.iterdir()):
            if path.name != "READY.json":
                os.replace(path, output / path.name)
        os.replace(stage / "READY.json", output / "READY.json")
    print("\n" + summary)
    return output


def train_v0(project_root: Path, **options) -> Path:
    return train_source_hybrid(project_root, model_version="v0", **options)


def train_v1(project_root: Path, **options) -> Path:
    options.setdefault("patience", 0)
    return train_source_hybrid(project_root, model_version="v1", **options)
