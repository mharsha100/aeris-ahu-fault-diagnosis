"""Freeze AERIS day splits and create matching classical/temporal inputs.

Requires only the existing numpy/pandas environment and outputs of script 02.
Default experiment: valid occupied/fan-on periods, 15 one-minute samples per
window, starts every 5 minutes. Models are not trained by this script.

Source: first four complete weather days per season train; subsequent days
validate, calibrate, and test. Every scenario and baseline for a weather day
stays in the same partition. This tests new days within the existing scenarios,
not previously unseen source fault severities or buildings.

Target: freeze complete days before any model is trained. Adaptation and
technician-feedback pools have no y arrays. Target window labels are stored
separately for later evaluation/simulated technician review.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARED = PROJECT_ROOT / "data" / "aeris_prepared"
OUTPUT = PROJECT_ROOT / "data" / "aeris_windows_v1"
SOURCE_ROLES = ["source_train", "source_validation", "source_calibration", "source_test"]
TARGET_ROLES = ["target_adapt", "target_feedback", "target_test"]

# Fixed using class/day coverage, before model fitting or performance inspection.
# The test set contains normal/damper/cooling/heating days in BOTH real AHU modes.
TARGET_PLAN = {
    "target_adapt": [
        "target:SZVAV:2017-09-11", "target:SZVAV:2017-09-14", "target:SZVAV:2017-09-18",
        "target:SZVAV:2017-09-20", "target:SZVAV:2017-09-21",
        "target:SZCAV:2017-03-11", "target:SZCAV:2017-03-15", "target:SZCAV:2017-03-18",
        "target:SZCAV:2017-03-22", "target:SZCAV:2017-03-25",
    ],
    "target_feedback": [
        "target:SZVAV:2017-09-12", "target:SZVAV:2017-09-23",
        "target:SZCAV:2017-03-16", "target:SZCAV:2017-03-20", "target:SZCAV:2017-03-21",
        "target:SZCAV:2017-03-23", "target:SZCAV:2017-03-26", "target:SZCAV:2017-03-31",
    ],
    "target_test": [
        "target:SZVAV:2017-09-15", "target:SZVAV:2017-09-19", "target:SZVAV:2017-09-22",
        "target:SZVAV:2017-09-24", "target:SZCAV:2017-03-12", "target:SZCAV:2017-03-19",
        "target:SZCAV:2017-03-24", "target:SZCAV:2017-04-01",
    ],
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_values(series: pd.Series, column: str) -> pd.Series:
    values = series.astype(str).str.lower()
    if not values.isin(["true", "false"]).all():
        raise ValueError(f"Unexpected boolean values in {column}")
    return values.eq("true")


def read_observations(domain: str, features: list[str]) -> pd.DataFrame:
    metadata = ["row_id", "domain", "dataset", "season", "run_id", "condition_group",
                "segment_id", "sample_time_s", "core_valid", "active_occupied"]
    data = pd.read_csv(PREPARED / f"{domain}_observations.csv.gz", usecols=metadata + features,
                       dtype={feature: "float32" for feature in features})
    if data.empty or data["row_id"].duplicated().any() or not data["domain"].eq(domain).all():
        raise ValueError(f"Invalid or duplicate {domain} observation records")
    data["core_valid"] = bool_values(data["core_valid"], "core_valid")
    data["active_occupied"] = bool_values(data["active_occupied"], "active_occupied")
    return data


def assign_groups(source: pd.DataFrame, target: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    mapping, records = {}, []
    for season, season_rows in source.groupby("season", sort=True):
        eligible = season_rows[season_rows["core_valid"] & season_rows["active_occupied"]]
        days = sorted(eligible["condition_group"].unique(), key=lambda name: int(name.rsplit("_", 1)[1]))
        if len(days) != 7:
            raise ValueError(f"Expected 7 active source weather days in {season}; found {len(days)}")
        for index, day in enumerate(days):
            role = "source_train" if index < 4 else SOURCE_ROLES[index - 3]
            mapping[day] = role
        for day, rows in season_rows.groupby("condition_group", sort=True):
            records.append({"domain": "source", "dataset": "G36-1wk", "season": season,
                            "condition_group": day, "partition": mapping.get(day, "excluded_no_active_data"),
                            "raw_rows": len(rows),
                            "eligible_rows": int((rows["core_valid"] & rows["active_occupied"]).sum())})

    target_mapping = {}
    for role, days in TARGET_PLAN.items():
        for day in days:
            if day in target_mapping:
                raise ValueError(f"Target day assigned twice: {day}")
            target_mapping[day] = role
    actual_days = set(target["condition_group"])
    if actual_days != set(target_mapping):
        raise ValueError(f"Target days do not match the frozen plan; extra={sorted(actual_days-set(target_mapping))}, missing={sorted(set(target_mapping)-actual_days)}")
    mapping.update(target_mapping)
    for day, rows in target.groupby("condition_group", sort=True):
        records.append({"domain": "target", "dataset": rows["dataset"].iloc[0],
                        "season": rows["season"].iloc[0], "condition_group": day,
                        "partition": mapping[day], "raw_rows": len(rows),
                        "eligible_rows": int((rows["core_valid"] & rows["active_occupied"]).sum())})
    manifest = pd.DataFrame(records)
    if manifest["condition_group"].duplicated().any():
        raise ValueError("A condition group appears more than once in the split plan")
    return mapping, manifest


def tabular_inputs(windows: np.ndarray, features: list[str]) -> tuple[np.ndarray, list[str]]:
    """Raw-unit summaries for trees; the exact same windows feed the neural branch."""
    statistics = [
        ("mean", windows.mean(axis=1)), ("std", windows.std(axis=1)),
        ("min", windows.min(axis=1)), ("max", windows.max(axis=1)),
        ("last", windows[:, -1, :]),
        ("trend_per_minute", (windows[:, -1, :] - windows[:, 0, :]) / (windows.shape[1] - 1)),
    ]
    arrays = [values for _, values in statistics]
    names = [f"{feature}__{name}" for name, _ in statistics for feature in features]
    for left, right, name in [
        ("supply_temp_c", "mixed_temp_c", "supply_minus_mixed_c"),
        ("mixed_temp_c", "return_temp_c", "mixed_minus_return_c"),
        ("outdoor_temp_c", "return_temp_c", "outdoor_minus_return_c"),
    ]:
        delta = windows[:, :, features.index(left)] - windows[:, :, features.index(right)]
        arrays.extend([delta.mean(axis=1, keepdims=True), delta[:, -1:]])
        names.extend([f"{name}__mean", f"{name}__last"])
    return np.concatenate(arrays, axis=1).astype("float32"), names


def create_windows(frame: pd.DataFrame, features: list[str], length: int, stride: int,
                   partition: str, mean: np.ndarray, scale: np.ndarray) -> tuple[dict, pd.DataFrame, list[str], dict]:
    frame = frame.loc[frame["core_valid"] & frame["active_occupied"]].copy()
    if frame.empty:
        raise ValueError(f"No eligible rows for {partition}")
    values = frame[features].to_numpy(dtype="float32")
    if not np.isfinite(values).all():
        raise ValueError(f"Eligible rows contain nonfinite features in {partition}")
    # Filtering an off/unoccupied row must create a gap, even within an old segment.
    boundary = frame["run_id"].ne(frame["run_id"].shift())
    boundary |= frame["segment_id"].ne(frame["segment_id"].shift())
    boundary |= frame["condition_group"].ne(frame["condition_group"].shift())
    boundary |= frame["sample_time_s"].diff().sub(60).abs().gt(0.000001)
    frame["_block"] = boundary.cumsum()
    sequences, source_labels, meta_frames = [], [], []
    short_blocks = 0
    for _, block in frame.groupby("_block", sort=False):
        if len(block) < length:
            short_blocks += 1
            continue
        if block["condition_group"].nunique() != 1 or block["run_id"].nunique() != 1:
            raise ValueError("A time block crosses a day or scenario boundary")
        times = block["sample_time_s"].to_numpy(dtype="float64")
        if not np.allclose(np.diff(times), 60, rtol=0, atol=0.000001):
            raise ValueError("A time block contains a sampling gap")
        raw = block[features].to_numpy(dtype="float32")
        windows = np.lib.stride_tricks.sliding_window_view(raw, window_shape=length, axis=0)
        windows = windows.transpose(0, 2, 1)[::stride].copy()
        starts = np.arange(0, len(block) - length + 1, stride)
        ends = starts + length - 1
        start_ids = block["row_id"].iloc[starts].to_numpy(dtype=str)
        end_ids = block["row_id"].iloc[ends].to_numpy(dtype=str)
        ids = [hashlib.sha256(f"{a}|{b}".encode()).hexdigest()[:24] for a, b in zip(start_ids, end_ids)]
        sequences.append(windows)
        meta_frames.append(pd.DataFrame({
            "window_id": ids, "partition": partition,
            "domain": block["domain"].iloc[0], "dataset": block["dataset"].iloc[0],
            "season": block["season"].iloc[0], "run_id": block["run_id"].iloc[0],
            "condition_group": block["condition_group"].iloc[0],
            "segment_id": block["segment_id"].iloc[0],
            "start_row_id": start_ids, "end_row_id": end_ids,
            "start_time_s": times[starts], "end_time_s": times[ends],
        }))
        if partition.startswith("source_"):
            labels = block["_class_id"].to_numpy(dtype="int64")
            if len(np.unique(labels)) != 1:
                raise ValueError("A source time block has inconsistent fault labels")
            source_labels.append(labels[ends])
    if not sequences:
        raise ValueError(f"No complete {length}-minute windows for {partition}")
    raw_windows = np.concatenate(sequences, axis=0)
    table, table_names = tabular_inputs(raw_windows, features)
    neural = ((raw_windows.astype("float64") - mean) / scale).astype("float32")
    metadata = pd.concat(meta_frames, ignore_index=True)
    if metadata["window_id"].duplicated().any() or not np.isfinite(neural).all() or not np.isfinite(table).all():
        raise ValueError(f"Invalid window IDs or nonfinite arrays in {partition}")
    arrays = {"X_seq": neural, "X_tab": table, "window_id": metadata["window_id"].to_numpy(dtype=str)}
    if source_labels:
        arrays["y"] = np.concatenate(source_labels)
        if set(np.unique(arrays["y"])) != {0, 1, 2, 3}:
            raise ValueError(f"Not all source classes have windows in {partition}")
    return arrays, metadata, table_names, {"eligible_rows": len(frame), "short_blocks_skipped": short_blocks}


def training_environment() -> dict:
    result = {"python_executable": sys.executable, "python_version": platform.python_version(), "packages": {}}
    for name in ["numpy", "pandas", "scikit-learn", "torch", "xgboost", "streamlit"]:
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        result["nvidia_smi_returncode"] = gpu.returncode
        result["nvidia_smi"] = (gpu.stdout or gpu.stderr).strip()[:4000]
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["nvidia_smi"] = f"Unavailable: {type(exc).__name__}"
    if result["packages"]["torch"] is not None:
        try:
            import torch
            result["torch_cuda_runtime"] = torch.version.cuda
            result["torch_cuda_available"] = bool(torch.cuda.is_available())
            if result["torch_cuda_available"]:
                result["torch_gpu_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            result["torch_check_error"] = f"{type(exc).__name__}: {exc}"
    return result


def main(length: int = 15, stride: int = 5) -> int:
    if length < 2 or stride < 1:
        raise ValueError("Window length must be >= 2 and stride >= 1")
    if not (PREPARED / "READY.json").is_file():
        raise FileNotFoundError("Run 02_prepare_data.py successfully before this script")
    contract = json.loads((PREPARED / "data_contract.json").read_text(encoding="utf-8"))
    prepared_summary = json.loads((PREPARED / "preparation_summary.json").read_text(encoding="utf-8"))
    features = contract["core_features"]
    if contract["contract_version"] != "aeris-data-v1" or len(features) != 10 or len(set(features)) != 10:
        raise ValueError("Unexpected prepared feature contract")
    print("Loading prepared observations...", flush=True)
    source = read_observations("source", features)
    target = read_observations("target", features)
    if len(source) != prepared_summary["source_rows"] or len(target) != prepared_summary["target_rows"]:
        raise ValueError("Prepared observation counts differ from the preparation summary")
    if set(source["row_id"]) & set(target["row_id"]):
        raise ValueError("Source and target row IDs overlap")
    group_mapping, group_manifest = assign_groups(source, target)
    source["_partition"] = source["condition_group"].map(group_mapping)
    target["_partition"] = target["condition_group"].map(group_mapping)
    eligible_source = source["core_valid"] & source["active_occupied"]
    if source.loc[eligible_source, "_partition"].isna().any() or target["_partition"].isna().any():
        raise ValueError("An eligible row has no partition")
    source_labels = pd.read_csv(PREPARED / "source_labels.csv.gz", usecols=["row_id", "class_id"])
    if not source_labels["row_id"].equals(source["row_id"]):
        raise ValueError("Source labels are not aligned with prepared observations")
    if not source_labels["class_id"].isin([0, 1, 2, 3]).all():
        raise ValueError("Invalid source class IDs")
    source["_class_id"] = source_labels["class_id"]
    # These are the only rows used to fit normalization.
    fit_rows = source.loc[eligible_source & source["_partition"].eq("source_train"), features]
    fit_values = fit_rows.to_numpy(dtype="float64")
    if not np.isfinite(fit_values).all() or len(fit_values) == 0:
        raise ValueError("Invalid source training inputs for normalization")
    mean, scale = fit_values.mean(axis=0), fit_values.std(axis=0)
    constants = scale < 0.00000001
    scale[constants] = 1.0
    scaler = {"fit_partition": "source_train", "fit_rows": len(fit_values), "features": features,
              "mean": mean.tolist(), "scale": scale.tolist(), "std_ddof": 0,
              "constant_features": [name for name, constant in zip(features, constants) if constant],
              "applies_to": "X_seq only; X_tab retains Celsius/fraction/binary units"}
    print(f"Normalization fitted on {len(fit_values):,} source-training rows only.", flush=True)

    with TemporaryDirectory(prefix="aeris_windows_", dir=PROJECT_ROOT / "data") as temp:
        stage = Path(temp)
        (stage / "evaluation").mkdir()
        metas, counts, table_names = [], [], None
        for role in SOURCE_ROLES + TARGET_ROLES:
            observations = source if role.startswith("source_") else target
            frame = observations.loc[observations["_partition"].eq(role)]
            arrays, metadata, names, info = create_windows(frame, features, length, stride, role, mean, scale)
            if table_names is not None and table_names != names:
                raise ValueError("Classical feature order differs between partitions")
            table_names = names
            if role.startswith("target_") and "y" in arrays:
                raise ValueError("Target labels must not enter training-input packs")
            np.savez_compressed(stage / f"{role}.npz", **arrays)
            metas.append(metadata)
            counts.append({"partition": role, "windows": len(metadata),
                           "condition_groups": metadata["condition_group"].nunique(), **info})
            print(f"{role}: {len(metadata):,} windows | {metadata['condition_group'].nunique()} whole-day groups", flush=True)
        metadata = pd.concat(metas, ignore_index=True)
        if metadata["window_id"].duplicated().any():
            raise ValueError("Window IDs overlap between partitions")
        if metadata.groupby("condition_group")["partition"].nunique().max() != 1:
            raise ValueError("A source weather day or target day leaked across partitions")
        expected_duration = 60 * (length - 1)
        if not np.allclose(metadata["end_time_s"] - metadata["start_time_s"], expected_duration, rtol=0, atol=0.000001):
            raise ValueError("A window spans a time gap")

        # Target ground truth is read only after all transforms and window packs exist.
        reference = pd.read_csv(PREPARED / "target_reference_labels.csv.gz")
        if not reference["row_id"].equals(target["row_id"]) or reference["row_id"].duplicated().any():
            raise ValueError("Target reference labels are not aligned")
        if reference.groupby("condition_group")["class_id"].nunique().max() != 1:
            raise ValueError("Target day labels change inside a day")
        target_meta = metadata[metadata["domain"].eq("target")].copy()
        reference = reference.set_index("row_id")
        labels_at_end = reference.loc[target_meta["end_row_id"]].reset_index(drop=True)
        target_meta = target_meta.reset_index(drop=True)
        for column in ["class_id", "class_name", "mechanism", "intensity", "documented_domain_pair"]:
            target_meta[column] = labels_at_end[column].to_numpy()
        for role in TARGET_ROLES:
            if set(target_meta.loc[target_meta["partition"].eq(role), "class_id"]) != {0, 1, 2, 3}:
                raise ValueError(f"Not all four classes have usable windows in {role}")
        target_meta.to_csv(stage / "evaluation" / "target_window_reference.csv.gz", index=False, compression="gzip")
        metadata.to_csv(stage / "window_metadata.csv.gz", index=False, compression="gzip")
        group_manifest.to_csv(stage / "split_manifest.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(counts).to_csv(stage / "window_counts.csv", index=False, encoding="utf-8-sig")

        assignments = pd.concat([
            source.loc[eligible_source, ["row_id", "domain", "condition_group", "_partition"]],
            target.loc[target["core_valid"] & target["active_occupied"], ["row_id", "domain", "condition_group", "_partition"]],
        ], ignore_index=True).rename(columns={"_partition": "partition"})
        if assignments["row_id"].duplicated().any():
            raise ValueError("A raw row is assigned more than once")
        assignments.to_csv(stage / "row_split_assignments.csv.gz", index=False, compression="gzip")

        class_rows = []
        class_names = {int(value): key for key, value in contract["class_ids"].items()}
        for role in SOURCE_ROLES:
            with np.load(stage / f"{role}.npz", allow_pickle=False) as pack:
                for class_id, n in zip(*np.unique(pack["y"], return_counts=True)):
                    class_rows.append({"partition": role, "class_name": class_names[int(class_id)], "windows": int(n)})
        for (role, class_name), rows in target_meta.groupby(["partition", "class_name"], sort=True):
            class_rows.append({"partition": role, "class_name": class_name, "windows": len(rows)})
        pd.DataFrame(class_rows).to_csv(stage / "split_class_counts.csv", index=False, encoding="utf-8-sig")
        target_days = target_meta[["partition", "dataset", "condition_group", "class_name", "documented_domain_pair"]].drop_duplicates()
        target_days.to_csv(stage / "target_day_plan.csv", index=False, encoding="utf-8-sig")
        (stage / "source_scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")

        protocol = {
            "protocol_version": "aeris-windows-v1", "data_contract_version": contract["contract_version"],
            "window_minutes": length, "stride_minutes": stride,
            "core_features": features, "tabular_features": table_names,
            "sequence_shape": ["N", length, len(features)], "sequence_order": "oldest to newest",
            "scope": "core-valid, occupied, fan-on; off/unoccupied cases are outside the first trained operating scope",
            "group_assignments": group_mapping,
            "source_evaluation_scope": "New blocked weather days within the same source scenario runs. Not a held-out-building or unseen-fault-severity test.",
            "source_validation_use": "Model/early-stopping/fusion choices; no target test feedback",
            "source_calibration_use": "Probability calibration using source held-out days",
            "target_adapt_use": "V1 unlabeled adaptation only; y is absent from NPZ",
            "target_feedback_use": "V2 candidate pool; reveal labels only for selected technician-reviewed examples",
            "target_test_use": "Common held-out benchmark for frozen V0/V1/V2; never train, tune, calibrate or select feedback with it",
            "target_reference_location": "evaluation/target_window_reference.csv.gz; evaluation/simulated-review only",
            "method_limits": [
                "Overlapping windows within a day are dependent; report per-day/mode metrics and do not treat windows as independent building trials.",
                "Target days were allocated using documented class/mode coverage before model performance was observed.",
                "This is a controlled retrospective adaptation experiment, not a chronological prospective deployment study.",
                "Two real test days per class provide limited coverage; calibration/transfer improvement is not guaranteed.",
            ],
            "prepared_inputs_sha256": {name: file_sha256(PREPARED / name) for name in [
                "data_contract.json", "input_manifest.csv", "source_observations.csv.gz", "source_labels.csv.gz",
                "target_observations.csv.gz", "target_reference_labels.csv.gz",
            ]},
        }
        (stage / "window_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        environment = training_environment()
        (stage / "training_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
        lines = ["AERIS - splits and windows complete", f"Window: {length} minutes | stride: {stride} minutes",
                 f"Neural inputs: {length} x {len(features)} per window", f"Classical inputs: {len(table_names)} per window",
                 "Scope: valid occupied periods with the fan running.", "", "Partitions:"]
        lines += [f"  {row['partition']}: {row['windows']:,} windows | {row['condition_groups']} day groups" for row in counts]
        lines += ["", f"Scaler fitted on: {len(fit_values):,} source-training rows only",
                  "Checks passed: disjoint groups/row assignments, continuous windows, all four classes in each partition.",
                  "Target input NPZ files contain no y labels.",
                  f"Output folder: {OUTPUT}", "Review attachment: windows_review.zip (small reports only)",
                  "Next: check the training environment, then train the classical and temporal parts of hybrid V0."]
        summary_text = "\n".join(lines) + "\n"
        (stage / "windows_summary.txt").write_text(summary_text, encoding="utf-8")
        review_files = ["windows_summary.txt", "window_counts.csv", "split_manifest.csv", "split_class_counts.csv",
                        "target_day_plan.csv", "source_scaler.json", "window_protocol.json", "training_environment.json"]
        with zipfile.ZipFile(stage / "windows_review.zip", "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for filename in review_files:
                bundle.write(stage / filename, arcname=filename)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "READY.json").unlink(missing_ok=True)
        (OUTPUT / "evaluation").mkdir(exist_ok=True)
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                destination = OUTPUT / path.relative_to(stage)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, destination)
        ready = {"protocol_version": protocol["protocol_version"], "window_minutes": length,
                 "stride_minutes": stride, "created_utc": datetime.now(timezone.utc).isoformat(),
                 "protocol_sha256": file_sha256(OUTPUT / "window_protocol.json")}
        (OUTPUT / "READY.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")
    print("\n" + summary_text)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-minutes", type=int, default=15)
    parser.add_argument("--stride-minutes", type=int, default=5)
    options = parser.parse_args()
    try:
        raise SystemExit(main(options.window_minutes, options.stride_minutes))
    except (ValueError, FileNotFoundError, OSError, KeyError) as exc:
        print(f"\nSTOP - split/window preparation needs review: {exc}", flush=True)
        raise SystemExit(1)
