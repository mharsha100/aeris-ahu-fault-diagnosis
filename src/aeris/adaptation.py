"""One conservative, unlabeled-target adaptation round for the shared hybrid."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
import shutil
import time
import zipfile
from datetime import datetime, timezone
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .modeling import (CLASS_NAMES, HybridPredictor, fit_temperature, load_hybrid,
                       metrics, neural_logits, sha256_file, softmax,
                       temperature_probabilities)
from .training import environment_check, load_source_partition, write_json


# Fixed before examining target predictions. None is tuned against target labels.
POLICY = {
    "hybrid_min_confidence": 0.95,
    "each_branch_min_confidence": 0.90,
    "maximum_source_standard_deviations": 6.0,
    "stable_consecutive_windows": 3,
    "maximum_per_day": 8,
    "maximum_per_predicted_class": 24,
    "maximum_per_mode_and_predicted_class": 16,
    "minimum_selected_windows": 12,
    "minimum_selected_days": 2,
    "minimum_predicted_classes": 2,
    "forest_target_sample_weight": 5.0,
    "temporal_target_loss_fraction": 0.10,
    "temporal_learning_rate": 0.0001,
    "maximum_source_macro_f1_drop": 0.02,
    "maximum_source_per_class_recall_drop": 0.05,
}
META_COLUMNS = ["window_id", "partition", "dataset", "condition_group", "run_id",
                "segment_id", "start_time_s", "end_time_s"]


def adaptation_inputs(folder: Path, protocol: dict) -> tuple[dict, pd.DataFrame]:
    # Only this literal target role is opened. Evaluation/reference files are unused.
    with np.load(folder / "target_adapt.npz", allow_pickle=False) as stored:
        if "y" in stored.files:
            raise ValueError("The unlabeled adaptation pack unexpectedly contains y")
        pack = {name: stored[name].copy() for name in ["X_seq", "X_tab", "window_id"]}
    n = len(pack["window_id"])
    if n == 0 or len(set(pack["window_id"])) != n:
        raise ValueError("Empty adaptation pack or duplicate adaptation window IDs")
    if pack["X_seq"].shape != (n, protocol["window_minutes"], len(protocol["core_features"])):
        raise ValueError("Wrong temporal shape in the adaptation pack")
    if pack["X_tab"].shape != (n, len(protocol["tabular_features"])):
        raise ValueError("Wrong classical shape in the adaptation pack")
    if not np.isfinite(pack["X_seq"]).all() or not np.isfinite(pack["X_tab"]).all():
        raise ValueError("Adaptation inputs contain nonfinite values")
    pack["X_seq"] = pack["X_seq"].astype("float32", copy=False)
    pack["X_tab"] = pack["X_tab"].astype("float32", copy=False)
    meta = pd.read_csv(folder / "window_metadata.csv.gz", usecols=META_COLUMNS)
    meta = meta.loc[meta["partition"].eq("target_adapt")].set_index("window_id")
    if meta.index.duplicated().any() or set(meta.index) != set(pack["window_id"]):
        raise ValueError("Adaptation metadata and arrays are not aligned")
    meta = meta.loc[pack["window_id"]].reset_index()
    expected_groups = {group for group, role in protocol["group_assignments"].items()
                       if role == "target_adapt"}
    if set(meta["condition_group"]) != expected_groups:
        raise ValueError("The adaptation day assignment changed")
    if not set(meta["dataset"]).issubset({"SZVAV", "SZCAV"}):
        raise ValueError("Unexpected target AHU mode")
    duration = (protocol["window_minutes"] - 1) * 60
    if not np.allclose(meta["end_time_s"] - meta["start_time_s"], duration):
        raise ValueError("Adaptation windows do not have the frozen duration")
    return pack, meta


def select_pseudo_labels(pack: dict, meta: pd.DataFrame, predictions: dict,
                         protocol: dict) -> pd.DataFrame:
    audit = meta.copy()
    audit["pack_index"] = np.arange(len(meta))
    labels = predictions["class_id"]
    audit["pseudo_class_id"] = labels
    audit["pseudo_class_name"] = [CLASS_NAMES[index] for index in labels]
    audit["hybrid_confidence"] = predictions["confidence"]
    rows = np.arange(len(labels))
    audit["forest_class_probability"] = predictions["forest_probabilities"][rows, labels]
    audit["temporal_class_probability"] = predictions["temporal_probabilities"][rows, labels]
    audit["branches_agree"] = ~predictions["branch_disagreement"]
    audit["confidence_pass"] = (
        (audit["hybrid_confidence"] >= POLICY["hybrid_min_confidence"])
        & (audit["forest_class_probability"] >= POLICY["each_branch_min_confidence"])
        & (audit["temporal_class_probability"] >= POLICY["each_branch_min_confidence"])
    )
    # A broad source-scale plausibility filter, not a target correctness guarantee.
    audit["largest_source_z"] = np.abs(pack["X_seq"]).max(axis=(1, 2))
    audit["source_envelope_pass"] = audit["largest_source_z"].le(
        POLICY["maximum_source_standard_deviations"])
    audit["base_gate_pass"] = (audit["confidence_pass"] & audit["branches_agree"]
                              & audit["source_envelope_pass"])
    audit["stability_pass"] = False
    stride_seconds = protocol["stride_minutes"] * 60
    stable_length = POLICY["stable_consecutive_windows"]
    for _, group in audit.groupby(["run_id", "segment_id", "condition_group"], sort=True):
        group = group.sort_values(["start_time_s", "window_id"])
        streak, previous_time, previous_class = 0, None, None
        for index, row in group.iterrows():
            continuous = previous_time is not None and abs(row["start_time_s"] - previous_time - stride_seconds) < 0.000001
            if row["base_gate_pass"]:
                streak = streak + 1 if continuous and row["pseudo_class_id"] == previous_class else 1
            else:
                streak = 0
            audit.at[index, "stability_pass"] = streak >= stable_length
            previous_time, previous_class = row["start_time_s"], row["pseudo_class_id"]
    audit["selection_score"] = audit[["hybrid_confidence", "forest_class_probability",
                                     "temporal_class_probability"]].min(axis=1)
    audit["selected"] = False
    audit["selection_status"] = "failed_confidence_agreement_or_envelope"
    audit.loc[audit["base_gate_pass"], "selection_status"] = "insufficient_temporal_stability"
    audit.loc[audit["stability_pass"], "selection_status"] = "not_selected"
    day_counts, class_counts, mode_class_counts, intervals = {}, {}, {}, {}
    eligible = audit.loc[audit["stability_pass"]].sort_values(
        ["selection_score", "window_id"], ascending=[False, True])
    for index, row in eligible.iterrows():
        day, label = row["condition_group"], int(row["pseudo_class_id"])
        mode_class = (row["dataset"], label)
        if (day_counts.get(day, 0) >= POLICY["maximum_per_day"]
                or class_counts.get(label, 0) >= POLICY["maximum_per_predicted_class"]
                or mode_class_counts.get(mode_class, 0) >= POLICY["maximum_per_mode_and_predicted_class"]):
            audit.at[index, "selection_status"] = "diversity_cap"
            continue
        # Chosen pseudo training windows share no minutes within the same run/day.
        interval_key = (row["run_id"], day)
        previous = intervals.get(interval_key, [])
        if any(row["start_time_s"] <= end and row["end_time_s"] >= start for start, end in previous):
            audit.at[index, "selection_status"] = "overlaps_selected_window"
            continue
        audit.at[index, "selected"] = True
        audit.at[index, "selection_status"] = "selected_unverified_pseudo_label"
        day_counts[day] = day_counts.get(day, 0) + 1
        class_counts[label] = class_counts.get(label, 0) + 1
        mode_class_counts[mode_class] = mode_class_counts.get(mode_class, 0) + 1
        intervals.setdefault(interval_key, []).append((row["start_time_s"], row["end_time_s"]))
    return audit


def enough_candidates(audit: pd.DataFrame) -> tuple[bool, dict]:
    selected = audit.loc[audit["selected"]]
    report = {
        "adaptation_windows": len(audit),
        "base_gate_windows": int(audit["base_gate_pass"].sum()),
        "temporally_stable_windows": int(audit["stability_pass"].sum()),
        "selected_windows": len(selected),
        "selected_days": int(selected["condition_group"].nunique()),
        "selected_predicted_classes": int(selected["pseudo_class_id"].nunique()),
        "selected_by_predicted_class": {CLASS_NAMES[int(key)]: int(count) for key, count in
                                       selected["pseudo_class_id"].value_counts().items()},
        "selection_labels_are_unverified": True,
    }
    passed = (len(selected) >= POLICY["minimum_selected_windows"]
              and report["selected_days"] >= POLICY["minimum_selected_days"]
              and report["selected_predicted_classes"] >= POLICY["minimum_predicted_classes"])
    report["enough_candidates"] = passed
    return passed, report


def fine_tune_temporal(temporal, train: dict, validation: dict, pseudo: dict, *,
                       device: str, epochs: int, batch_size: int, seed: int) -> tuple[list, int]:
    counts = np.bincount(train["y"], minlength=4)
    class_weights = torch.tensor(len(train["y"]) / (4 * counts), dtype=torch.float32, device=device)
    source_loss = nn.CrossEntropyLoss(weight=class_weights)
    target_loss = nn.CrossEntropyLoss(reduction="none")
    loader = DataLoader(TensorDataset(torch.from_numpy(train["X_seq"]), torch.from_numpy(train["y"])),
                        batch_size=batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed), pin_memory=device == "cuda")
    optimizer = torch.optim.AdamW(temporal.parameters(), lr=POLICY["temporal_learning_rate"],
                                 weight_decay=0.0001)
    pseudo_counts = np.bincount(pseudo["y"], minlength=4)
    sampling = 1.0 / pseudo_counts[pseudo["y"]]
    sampling /= sampling.sum()
    rng = np.random.default_rng(seed)
    before = softmax(neural_logits(temporal, validation["X_seq"], device))
    baseline_score = float(f1_score(validation["y"], before.argmax(axis=1), average="macro",
                                   labels=[0, 1, 2, 3], zero_division=0))
    retained = {key: value.detach().cpu().clone() for key, value in temporal.state_dict().items()}
    retained_epoch, history = 0, []
    for epoch in range(1, epochs + 1):
        started, total, batches = time.perf_counter(), 0.0, 0
        temporal.train()
        for X_source, y_source in loader:
            indices = rng.choice(len(pseudo["y"]), size=min(32, len(pseudo["y"])),
                                 replace=True, p=sampling)
            X_target = torch.from_numpy(pseudo["X_seq"][indices]).to(device)
            y_target = torch.from_numpy(pseudo["y"][indices]).to(device)
            confidence = torch.from_numpy(pseudo["confidence"][indices].astype("float32")).to(device)
            optimizer.zero_grad(set_to_none=True)
            source_term = source_loss(temporal(X_source.to(device)), y_source.to(device))
            target_term = (target_loss(temporal(X_target), y_target) * confidence).sum() / confidence.sum()
            fraction = POLICY["temporal_target_loss_fraction"]
            loss = (1 - fraction) * source_term + fraction * target_term
            if not torch.isfinite(loss).item():
                raise RuntimeError("Adaptation fine-tuning produced a nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(temporal.parameters(), 1.0)
            optimizer.step()
            total, batches = total + float(loss.item()), batches + 1
        probabilities = softmax(neural_logits(temporal, validation["X_seq"], device))
        score = float(f1_score(validation["y"], probabilities.argmax(axis=1), average="macro",
                               labels=[0, 1, 2, 3], zero_division=0))
        passes = score >= baseline_score - POLICY["maximum_source_macro_f1_drop"]
        if passes:
            retained = {key: value.detach().cpu().clone() for key, value in temporal.state_dict().items()}
            retained_epoch = epoch
        history.append({"epoch": epoch, "combined_training_loss": total / batches,
                        "source_validation_temporal_macro_f1": score, "retained": passes,
                        "seconds": time.perf_counter() - started})
        print(f"Adaptation epoch {epoch:02d}/{epochs} | source temporal F1={score:.4f} | retained={passes}", flush=True)
        if not passes:
            print("Stopping fine-tuning and restoring the last retained temporal state.", flush=True)
            break
    temporal.load_state_dict(retained)
    temporal.to(device).eval()
    return history, retained_epoch


def recalibrate(candidate: HybridPredictor, calibration: dict) -> None:
    forest = candidate.forest.predict_proba(calibration["X_tab"])
    temporal = softmax(neural_logits(candidate.temporal, calibration["X_seq"], candidate.device))
    forest_t = fit_temperature(forest, calibration["y"])
    temporal_t = fit_temperature(temporal, calibration["y"])
    weight = candidate.configuration["forest_weight"]
    combined = (weight * temperature_probabilities(forest, forest_t)
                + (1 - weight) * temperature_probabilities(temporal, temporal_t))
    candidate.configuration["temperatures"] = {
        "forest": forest_t, "temporal": temporal_t,
        "hybrid": fit_temperature(combined, calibration["y"]),
    }


def source_guardrail(baseline: dict, candidate: dict) -> dict:
    f1_drop = baseline["macro_f1"] - candidate["macro_f1"]
    recall_drops = {name: baseline["per_class"][name]["recall"] - candidate["per_class"][name]["recall"]
                    for name in CLASS_NAMES}
    passed = (f1_drop <= POLICY["maximum_source_macro_f1_drop"] + 1e-12
              and max(recall_drops.values()) <= POLICY["maximum_source_per_class_recall_drop"] + 1e-12)
    return {"passed": passed, "source_macro_f1_drop": f1_drop,
            "source_per_class_recall_drop": recall_drops,
            "interpretation": "A source-retention check; it does not establish a target-domain benefit."}


def adapt_hybrid(project_root: Path, *, model_name: str = "v1_attention", device: str = "cuda", epochs: int = 5,
             batch_size: int = 256, seed: int = 43, overwrite: bool = False) -> Path:
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    windows = project_root / "data/aeris_windows_v1"
    if model_name not in {"v0", "v1_attention"}:
        raise ValueError("Choose a supported, initially trained architecture: v0 or v1_attention")
    parent = project_root / "data/aeris_models" / model_name
    output = project_root / "data/aeris_models" / f"{model_name}_cross_adapted"
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Adaptation output already exists: {output}")
    environment = environment_check(device)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    baseline = load_hybrid(parent, device=device)
    expected_version = "v0" if model_name == "v0" else "v1"
    if baseline.configuration["model_version"] != expected_version:
        raise ValueError("The adaptation parent does not match the requested architecture")
    if model_name == "v1_attention" and baseline.configuration.get("architecture_id") != "rf_tcn_attention_v1":
        raise ValueError("V1 must be the freshly trained attention architecture")
    parent_sha = sha256_file(parent / "hybrid.json")
    parent_artifacts = {name: sha256_file(parent / name) for name in
                        ["hybrid.json", "READY.json", *baseline.configuration["artifact_sha256"]]}
    protocol_sha = sha256_file(windows / "window_protocol.json")
    if protocol_sha != baseline.configuration["artifact_sha256"]["window_protocol.json"]:
        raise ValueError("Current windows and saved parent use different protocols")
    ready = json.loads((windows / "READY.json").read_text(encoding="utf-8"))
    if ready["protocol_sha256"] != protocol_sha:
        raise ValueError("The window protocol is no longer frozen")
    if sha256_file(windows / "source_scaler.json") != baseline.configuration["artifact_sha256"]["source_scaler.json"]:
        raise ValueError("Source normalization changed after parent training")
    parent_run = json.loads((parent / "training_run.json").read_text(encoding="utf-8"))
    if set(parent_run["source_pack_sha256"]) != {"source_train", "source_validation", "source_calibration"}:
        raise ValueError("The parent provenance does not contain the three expected source packs")
    for role, digest in parent_run["source_pack_sha256"].items():
        if role not in {"source_train", "source_validation", "source_calibration"}:
            raise ValueError("Unexpected fitting role in the parent provenance")
        if sha256_file(windows / f"{role}.npz") != digest:
            raise ValueError(f"{role} changed after the parent training run")
    protocol = baseline.protocol
    train = load_source_partition(windows, "source_train", protocol)
    validation = load_source_partition(windows, "source_validation", protocol)
    calibration = load_source_partition(windows, "source_calibration", protocol)
    target, meta = adaptation_inputs(windows, protocol)
    used_ids = set(train["window_id"]) | set(validation["window_id"]) | set(calibration["window_id"])
    if used_ids & set(target["window_id"]):
        raise ValueError("Adaptation and source windows overlap")
    print(f"Adaptation device: {environment.get('gpu', device)}", flush=True)
    print("Auditing the unlabeled target_adapt pool with the selected frozen architecture...", flush=True)
    target_predictions = baseline.predict(target["X_seq"], target["X_tab"])
    audit = select_pseudo_labels(target, meta, target_predictions, protocol)
    enough, selection = enough_candidates(audit)
    print(f"Selected {selection['selected_windows']} unverified pseudo-label windows, "
          f"{selection['selected_days']} days, {selection['selected_predicted_classes']} predicted classes.", flush=True)
    baseline_predictions = baseline.predict(validation["X_seq"], validation["X_tab"])
    baseline_metrics = metrics(validation["y"], baseline_predictions["probabilities"])
    deployed, candidate_metrics, guard, history, retained_epoch = baseline, None, None, [], 0
    status = "skipped_insufficient_candidates"
    if enough:
        chosen = audit.loc[audit["selected"]].sort_values("pack_index")
        ix = chosen["pack_index"].to_numpy(dtype="int64")
        pseudo = {"X_seq": target["X_seq"][ix], "X_tab": target["X_tab"][ix],
                  "y": chosen["pseudo_class_id"].to_numpy(dtype="int64"),
                  "confidence": chosen["selection_score"].to_numpy(dtype="float64")}
        print("[1/3] Refitting the classical branch with full source replay and gated pseudo-labels...", flush=True)
        forest = RandomForestClassifier(**baseline.forest.get_params())
        forest.fit(np.concatenate([train["X_tab"], pseudo["X_tab"]]),
                   np.concatenate([train["y"], pseudo["y"]]),
                   sample_weight=np.concatenate([np.ones(len(train["y"])),
                       POLICY["forest_target_sample_weight"] * pseudo["confidence"]]))
        print("[2/3] Fine-tuning the temporal branch with source replay...", flush=True)
        temporal = copy.deepcopy(baseline.temporal)
        history, retained_epoch = fine_tune_temporal(
            temporal, train, validation, pseudo, device=device, epochs=epochs,
            batch_size=batch_size, seed=seed)
        candidate = HybridPredictor(forest, temporal, copy.deepcopy(baseline.configuration),
                                    baseline.scaler, baseline.protocol, device)
        print("[3/3] Source-only recalibration and source-retention check...", flush=True)
        recalibrate(candidate, calibration)
        candidate_predictions = candidate.predict(validation["X_seq"], validation["X_tab"])
        candidate_metrics = metrics(validation["y"], candidate_predictions["probabilities"])
        guard = source_guardrail(baseline_metrics, candidate_metrics)
        if guard["passed"]:
            deployed, status = candidate, "adapted"
        else:
            status = "rejected_source_validation"
            print("The candidate failed the source-retention check; the adapted bundle will retain its parent predictions.", flush=True)
    else:
        print("Insufficient gated candidate coverage; the adapted bundle will retain its parent predictions.", flush=True)
    config = copy.deepcopy(deployed.configuration)
    config.update({
        "model_version": expected_version, "parent_version": expected_version, "parent_hybrid_sha256": parent_sha,
        "learning_stage": "cross_setting_adaptation", "parent_bundle": model_name,
        "architecture_id": baseline.configuration.get("architecture_id", "rf_tcn_last_v0"),
        "initial_training_initialization": baseline.configuration.get("initialization", "fresh"),
        "initialization": "parent_for_adaptation", "parent_weights_loaded": True,
        "adaptation_status": status, "target_data_used": status == "adapted",
        "target_adaptation_attempted": enough, "target_data_audited": True,
        "target_labels_used": False, "test_data_used": False,
        "adaptation_policy": POLICY, "adaptation_seed": seed,
        "temporal_adaptation_retained_epoch": retained_epoch if status == "adapted" else 0,
        "adaptation_training_partitions": ["source_train", "target_adapt"] if enough else [],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    deployed = HybridPredictor(deployed.forest, deployed.temporal, config,
                               deployed.scaler, deployed.protocol, device)
    predictions = deployed.predict(validation["X_seq"], validation["X_tab"])
    deployed_metrics = metrics(validation["y"], predictions["probabilities"])
    if status != "adapted":
        np.testing.assert_allclose(predictions["probabilities"], baseline_predictions["probabilities"],
                                   atol=1e-7, rtol=1e-7)
    if any(sha256_file(parent / name) != digest for name, digest in parent_artifacts.items()):
        raise RuntimeError("A frozen parent artifact changed during adaptation")
    with TemporaryDirectory(prefix="aeris_adaptation_", dir=project_root / "data") as temp:
        stage = Path(temp)
        if status == "adapted":
            joblib.dump(deployed.forest, stage / "forest.joblib", compress=3)
            torch.save({key: value.detach().cpu() for key, value in deployed.temporal.state_dict().items()},
                       stage / "temporal_state.pt")
        else:
            for name in ["forest.joblib", "temporal_state.pt"]:
                shutil.copyfile(parent / name, stage / name)
        for name in ["source_scaler.json", "window_protocol.json"]:
            shutil.copyfile(parent / name, stage / name)
        config["artifact_sha256"] = {name: sha256_file(stage / name) for name in
                                    ["forest.joblib", "temporal_state.pt", "source_scaler.json", "window_protocol.json"]}
        write_json(stage / "hybrid.json", config)
        write_json(stage / "READY.json", {"model_version": expected_version,
                   "hybrid_sha256": sha256_file(stage / "hybrid.json")})
        restored = load_hybrid(stage, device=device)
        before = predictions["probabilities"][:32]
        after = restored.predict(validation["X_seq"][:32], validation["X_tab"][:32])["probabilities"]
        np.testing.assert_allclose(before, after, atol=1e-5, rtol=1e-5)
        audit.to_csv(stage / "pseudo_label_audit.csv.gz", index=False, compression="gzip")
        per_day = audit.groupby(["dataset", "condition_group"], sort=True).agg(
            windows=("window_id", "size"), gate_pass=("base_gate_pass", "sum"),
            stable=("stability_pass", "sum"), selected=("selected", "sum")).reset_index()
        per_day.to_csv(stage / "pseudo_labels_by_day.csv", index=False)
        write_json(stage / "selection_summary.json", selection)
        write_json(stage / "adaptation_policy.json", POLICY)
        write_json(stage / "source_retention_check.json", guard or
                   {"passed": None, "reason": "No candidate was trained"})
        detailed = {"before_adaptation": baseline_metrics, "after_adaptation": deployed_metrics}
        if candidate_metrics is not None:
            detailed["adaptation_candidate"] = candidate_metrics
        write_json(stage / "source_validation_metrics.json", detailed)
        metric_rows = [{"component": name, **{key: value for key, value in entry.items()
                       if key not in {"confusion_matrix", "per_class"}}} for name, entry in detailed.items()]
        pd.DataFrame(metric_rows).to_csv(stage / "source_validation_metrics.csv", index=False)
        pd.DataFrame(history, columns=["epoch", "combined_training_loss",
            "source_validation_temporal_macro_f1", "retained", "seconds"]).to_csv(stage / "adaptation_training_history.csv", index=False)
        write_json(stage / "training_environment.json", environment)
        run = {
            "model_version": expected_version, "architecture_id": config.get("architecture_id", "rf_tcn_last_v0"),
            "learning_stage": "cross_setting_adaptation", "parent_bundle": model_name,
            "parent_weights_loaded": True,
            "adaptation_status": status, "completed_adaptation_epochs": len(history),
            "parent_hybrid_sha256": parent_sha, "parent_artifacts_unchanged": True,
            "training_windows": len(train["y"]), "selected_pseudo_windows": selection["selected_windows"],
            "candidate_training_attempted": enough, "deployed_target_update": status == "adapted",
            "temporal_retained_epoch": retained_epoch, "maximum_epochs": epochs,
            "batch_size": batch_size, "seed": seed, "target_labels_opened": False,
            "feedback_pool_opened": False, "final_tests_opened": False,
            "source_pack_sha256": parent_run["source_pack_sha256"],
            "target_adapt_sha256": sha256_file(windows / "target_adapt.npz"),
            "saved_reload_max_abs_difference": float(np.max(np.abs(before - after))),
            "notes": [
                "Pseudo-labels are model guesses; agreement and source calibration do not verify real target labels.",
                "Selection uses nonoverlapping windows with fixed confidence/stability/diversity rules.",
                "All source training windows are replayed; source validation monitors retention, not target accuracy.",
                "Parent fusion weights, feature contract and source scaler remain fixed.",
                "No source or target final test, target reference labels, or feedback input is opened.",
                "A skipped or rejected attempt is reported honestly and preserves parent predictions.",
            ],
        }
        write_json(stage / "training_run.json", run)
        lines = [
            f"AERIS - cross-setting adaptation complete for {model_name}", f"Status: {status}",
            f"Selected unverified pseudo-label windows: {selection['selected_windows']}",
            f"Selected days: {selection['selected_days']}",
            f"Selected predicted classes: {selection['selected_predicted_classes']}",
            f"Retained temporal adaptation epoch: {retained_epoch if status == 'adapted' else 0}",
            f"Before-adaptation source-validation macro F1: {baseline_metrics['macro_f1']:.4f}",
            f"After-adaptation source-validation macro F1: {deployed_metrics['macro_f1']:.4f}",
            "Frozen parent unchanged: passed", "Saved/reloaded prediction check: passed",
            "Target labels, feedback pool and final test packs were not opened.",
            "These reports do not establish target accuracy or successful transfer.",
            f"Saved model: {output}", "Review attachment: cross_adaptation_review.zip",
        ]
        summary = "\n".join(lines) + "\n"
        (stage / "cross_adaptation_summary.txt").write_text(summary, encoding="utf-8")
        review = ["cross_adaptation_summary.txt", "hybrid.json", "training_run.json", "training_environment.json",
                  "adaptation_policy.json", "selection_summary.json", "pseudo_label_audit.csv.gz",
                  "pseudo_labels_by_day.csv", "source_retention_check.json",
                  "source_validation_metrics.json", "source_validation_metrics.csv", "adaptation_training_history.csv"]
        with zipfile.ZipFile(stage / "cross_adaptation_review.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in review:
                archive.write(stage / name, name)
        output.mkdir(parents=True, exist_ok=True)
        (output / "READY.json").unlink(missing_ok=True)
        for path in sorted(stage.iterdir()):
            if path.name != "READY.json":
                os.replace(path, output / path.name)
        os.replace(stage / "READY.json", output / "READY.json")
    print("\n" + summary, flush=True)
    return output


def adapt_v1(*args, **kwargs):
    raise RuntimeError("Legacy version mapping retired. Train V1 with 05_train_v1.py, then use 06_cross_adapt.py.")
