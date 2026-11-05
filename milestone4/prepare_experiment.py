"""Check scope, source-to-target channels and provenance before M4 fitting."""

from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VERSION = "m4-within-recording-v1"
LABELS = ["normal", "damper_fault", "cooling_valve_fault", "heating_valve_fault"]
MEANINGS = {
    "supply_temp_c": "Supply-air temperature downstream in the AHU supply path",
    "outdoor_temp_c": "Outdoor-air temperature entering the AHU outside-air path",
    "mixed_temp_c": "Mixed-air temperature after outdoor and return air combine",
    "return_temp_c": "Return-air temperature returning from the conditioned space",
    "fan_status": "Supply fan on/off indication; not a measured airflow rate",
    "fan_speed_cmd": "Requested fan speed; CAV/VAV sequences differ, not actual speed or airflow",
    "oa_damper_cmd": "Requested outside-air damper control; not actual damper position",
    "cooling_valve_cmd": "Cooling-valve control request; not measured opening or delivered coil flow",
    "heating_valve_cmd": "Heating-valve control request; not measured opening or delivered coil flow",
    "occupied": "Controller occupancy-mode indicator; not independently measured human occupancy",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def scope_record():
    return {
        "version": VERSION,
        "question": "Within the selected AHU recordings, how do final-step, attention, and combined temporal summaries change descriptive diagnostic errors, class recall and review workload?",
        "data": "Selected 45 G36-1wk source runs; existing LBNL SZVAV/SZCAV files are exploratory target data",
        "prediction_unit": "15-minute sensor window, stride 5 minutes",
        "condition_unit": "Seasonal shared-weather week across baseline and fault severities",
        "source_condition_groups": 3,
        "within_condition_time_separation": "Earlier/later days in the same shared-weather week; not independent-condition train/validation separation",
        "retained_claim": "Descriptive, conditional results on these recordings and implemented workflow controls",
        "independent_generalization_claims_allowed": False,
        "target_labels_previously_exposed": True,
        "target_is_sealed_holdout": False,
        "target_benchmark_label": "Retrospective exploratory benchmark",
        "original_independent_case_floor": 5,
        "original_floor_lowered": False,
        "window_count_is_independent_case_count": False,
        "significance_or_population_CI_from_windows_allowed": False,
        "independent_case_comparison_status": "blocked_under_original_m3_rule",
        "fixed_training": {"epochs": 50, "patience": 0, "seed": 42, "batch_size": 256,
                           "rf_trees": 200, "rf_weight": 0.5, "temporal_weight": 0.5},
        "architecture_experiments": [
            {"name": "E1_pooling", "question": "Does attention pooling change errors relative to final-step pooling?", "variants": ["m4_v0", "m4_v1"], "change": "Temporal summary", "fixed": "Data, RF/encoder/optimizer settings, training budget, seed, fusion and scoring"},
            {"name": "E2_combination", "question": "Does combining final-step and attention summaries improve the diagnostic tradeoff?", "variants": ["m4_v1", "m4_v2"], "change": "Combine both summaries with a learned head", "fixed": "Same data, RF/encoder/optimizer settings, training budget, seed, fusion and scoring; parameter count changes are reported"},
        ],
        "interpretation_limits": [
            "Data share weather, occupancy and simulated system dynamics; time separation does not create new independent conditions.",
            "Macro F1 and per-class metrics are development descriptions, not estimates of transfer to unseen buildings.",
            "Published target schedules/labels were explored earlier; withholding y arrays later does not restore a sealed holdout.",
            "Matching units and nominal signal roles do not prove matching multi-zone and single-zone dynamics.",
            "Probability thresholds and rule tests demonstrate policy behavior, not verified diagnostic safety.",
        ],
    }


def supported_settings():
    return {
        "version": VERSION,
        "source_research_settings": {
            "CoolingSeason": ["normal", "damper_fault", "cooling_valve_fault"],
            "HeatingSeason": ["normal", "damper_fault", "heating_valve_fault"],
            "ShoulderSeason": ["normal", "damper_fault", "cooling_valve_fault"],
        },
        "source_modes": "Core-valid, occupied and supply fan on",
        "source_status": "Within-recording research predictions; not verified operation on unseen AHUs",
        "target_settings": {
            "SZVAV": {"mode": "Single-zone VAV, summer recordings", "status": "exploratory_only", "automatic_escalation_supported": False,
                      "publisher_pair_families": ["damper_fault", "cooling_valve_fault"]},
            "SZCAV": {"mode": "Single-zone CAV, winter recordings", "status": "exploratory_only", "automatic_escalation_supported": False,
                      "publisher_pair_families": ["damper_fault", "heating_valve_fault"]},
        },
        "target_conditions": "Any future adaptation/feedback demonstration must retain exploratory status and audit label exposure; no high score overrides unsupported scope.",
        "mapping_sources": ["G36-1wk/00_explanations.pdf pp.3,7-8,10", "DomainPairs.pdf p.2",
                            "LBNLDataSynthesisInventory.pdf pp.17-19", "data/aeris_prepared/data_contract.json"],
        "command_limitation": "Signals are controller requests. Fault overrides, leakage intensities and CAV/VAV operation can change their relation to actual positions and flows.",
    }


def prepare(root=ROOT, reproduction_report=None, fresh_rebuild=False):
    root = Path(root)
    data = root / "data"
    prepared, windows = data / "aeris_prepared", data / "aeris_windows_v1"
    output = root / "milestone4/evidence/readiness"
    # READY is published only after the audit finishes. Re-running a completed
    # audit is refused to preserve the configuration behind subsequent fits.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Readiness evidence exists; preserve it before a new version: {output}")
    contract = read(prepared / "data_contract.json")
    protocol = read(windows / "window_protocol.json")
    if digest(windows / "window_protocol.json") != read(windows / "READY.json")["protocol_sha256"]:
        raise ValueError("Window protocol changed after preparation")
    if contract["core_features"] != protocol["core_features"] or len(contract["core_features"]) != 10:
        raise ValueError("Feature contract changed")
    if fresh_rebuild:
        # Historical weights are unavailable here. Do not construct a counterfeit
        # V0 bundle: compare semantic audit records and version the rebuilt packs.
        historical = root / "experiments/model_exploration/data_audit"
        for name in ["source_scaler.json"]:
            if read(windows / name) != read(historical / name):
                raise ValueError(f"Rebuilt {name} differs semantically from historical data")
        for name, current in [("input_manifest.csv", prepared), ("split_manifest.csv", windows)]:
            left, right = pd.read_csv(current / name), pd.read_csv(historical / name)
            pd.testing.assert_frame_equal(left, right, check_dtype=False)
        baseline_run = {"source_pack_sha256": {role: digest(windows / f"{role}.npz")
            for role in ["source_train", "source_validation", "source_calibration"]}}
        data_rebuild = {"mode": "fresh_public_data_rebuild", "raw_file_and_split_records_match_historical": True,
            "scaler_semantically_identical": True, "serialized_pack_hashes_match_historical": False,
            "historical_weights_available": False,
            "note": "New pack hashes and protocol are frozen for ALL three new runs. Historical numbers remain separate; no historical model is reconstructed from metadata."}
    else:
        baseline = read(data / "aeris_models/v0/hybrid.json")
        baseline_run = read(data / "aeris_models/v0/training_run.json")
        for name in ["window_protocol.json", "source_scaler.json"]:
            if digest(windows / name) != baseline["artifact_sha256"][name]:
                raise ValueError(f"{name} differs from the historical baseline")
        for role in ["source_train", "source_validation", "source_calibration"]:
            if digest(windows / f"{role}.npz") != baseline_run["source_pack_sha256"][role]:
                raise ValueError(f"Source pack differs from historical baseline: {role}")
        data_rebuild = {"mode": "historical_frozen_packs", "serialized_pack_hashes_match_historical": True}
    old = read(root / "milestone3/evidence/split_status.json")
    if old["source_condition_groups"] != 3 or not old["target_labels_previously_exposed"]:
        raise ValueError("Original M3 condition/label-exposure record differs; review scope")
    if any(old[key] != 5 for key in ["minimum_final_cases_per_class_per_season",
           "source_train_minimum_per_class_per_season", "source_validation_minimum_per_class_per_season"]):
        raise ValueError("Preserve the original independent-case reporting floor")
    if reproduction_report is None:
        candidates = sorted((data / "m4_checks").glob("run_*/reproduction_report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError("Run the fresh-data reproduction check first")
        reproduction_report = candidates[0]
    reproduction_report = Path(reproduction_report)
    reproduction = read(reproduction_report)
    if reproduction["status"] != "passed" or not reproduction["m3_v1_unchanged"] or reproduction["new_temperature_parse_failures"] != 91:
        raise ValueError("A passing fresh-data reproduction record is required")
    expected_pins = {}
    for line in (root / "requirements.txt").read_text(encoding="utf-8-sig").splitlines():
        if line.strip().startswith(("numpy==", "pandas==")):
            name, value = line.strip().split("==", 1)
            expected_pins[name] = value
    if set(expected_pins) != {"numpy", "pandas"} or reproduction["installed_data_dependencies"] != expected_pins:
        raise ValueError("The fresh data check used different dependency pins")
    for name, expected in reproduction["m3_v1_original_hashes"].items():
        original = root / "milestone3" / name
        raw = original.read_bytes()
        windows_bytes = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        if expected not in {digest(original), hashlib.sha256(windows_bytes).hexdigest()}:
            raise ValueError(f"Original m3-v1 changed beyond Git LF/Windows CRLF: {name}")
    manifest = pd.read_csv(prepared / "input_manifest.csv")
    if set(manifest.loc[manifest.domain.eq("source"), "season"]) != {"CoolingSeason", "HeatingSeason", "ShoulderSeason"}:
        raise ValueError("Unexpected source seasonal scope")
    header_checks, sample_checks = [], []
    feature_checks = {f: [] for f in contract["core_features"]}
    observed = {}
    representatives = set()
    for domain in ["source", "target"]:
        subset = manifest.loc[manifest.domain.eq(domain)]
        for _, group in subset.groupby("season" if domain == "source" else "dataset"):
            representative = group.loc[group.class_name.eq("normal")] if domain == "source" else group
            representatives.add(str((representative if len(representative) else group).iloc[0].relative_path))
        frame = pd.read_csv(prepared / f"{domain}_observations.csv.gz", usecols=["row_id"] + contract["core_features"])
        observed[domain] = frame.set_index("row_id")
    for row in manifest.itertuples(index=False):
        path = (data / row.relative_path).resolve()
        if not path.is_relative_to(data.resolve()) or not path.is_file():
            raise ValueError(f"Raw input path is missing or outside data: {row.relative_path}")
        if digest(path) != row.sha256:
            raise ValueError(f"Raw file changed: {row.relative_path}")
        mapping = contract["source_columns"] if row.domain == "source" else contract["target_columns_after_whitespace_strip"]
        actual = {str(c).strip() for c in pd.read_csv(path, nrows=0).columns}
        if not set(mapping[f] for f in contract["core_features"]).issubset(actual):
            raise ValueError(f"Missing required input channels: {row.relative_path}")
        header_checks.append({"domain": row.domain, "dataset": row.dataset, "season": row.season,
                              "path": row.relative_path, "sha256": row.sha256, "required_headers_present": True})
        if row.relative_path not in representatives:
            continue
        raw = pd.read_csv(path, nrows=1440, usecols=lambda n: n.strip() in set(mapping[f] for f in contract["core_features"]), low_memory=False)
        raw.columns = raw.columns.str.strip()
        file_id = hashlib.sha256(str(row.relative_path).encode()).hexdigest()[:16]
        ids = [f"{file_id}:{i}" for i in range(len(raw))]
        canonical = observed[row.domain].loc[ids]
        for feature in contract["core_features"]:
            values = pd.to_numeric(raw[mapping[feature]], errors="coerce")
            if feature.endswith("_temp_c"):
                values = values - 273.15 if row.domain == "source" else (values - 32) * 5 / 9
            values = values.replace([np.inf, -np.inf], np.nan).to_numpy(dtype="float32")
            saved = canonical[feature].to_numpy(dtype="float32")
            passed = bool(np.allclose(values, saved, atol=0.00003, rtol=0.00001, equal_nan=True))
            if not passed:
                raise ValueError(f"Canonical conversion differs: {row.relative_path}/{feature}")
            feature_checks[feature].append({"domain": row.domain, "rows": len(raw)})
            sample_checks.append({"path": row.relative_path, "domain": row.domain, "feature": feature,
                                  "rows_checked": len(raw), "conversion_matches_prepared": passed,
                                  "missing_after_numeric_conversion": int(np.isnan(values).sum())})
    channel_rows = []
    for feature in contract["core_features"]:
        temperature = feature.endswith("_temp_c")
        channel_rows.append({"feature": feature, "physical_role": MEANINGS[feature],
            "source_column": contract["source_columns"][feature],
            "target_column": contract["target_columns_after_whitespace_strip"][feature],
            "source_unit": "Kelvin" if temperature else "binary" if feature in {"fan_status", "occupied"} else "fraction",
            "target_unit": "Fahrenheit" if temperature else "binary" if feature in {"fan_status", "occupied"} else "fraction",
            "canonical_unit": "Celsius" if temperature else "binary" if feature in {"fan_status", "occupied"} else "fraction",
            "conversion": "source K-273.15; target (F-32)*5/9" if temperature else "identity; preserve observed values and quality flags",
            "samples_verified": len(feature_checks[feature]), "nominal_role_mapping": "documented candidate",
            "dynamic_equivalence_established": False,
            "limitation": "Multi-zone source vs single-zone target; sensor location and control sequences differ. " + MEANINGS[feature]})
    # Check actual development-pack labels/IDs, never target-reference/test packs.
    counts, all_ids = [], []
    for role in ["source_train", "source_validation", "source_calibration"]:
        with np.load(windows / f"{role}.npz", allow_pickle=False) as pack:
            ids, y = pack["window_id"], pack["y"]
            if set(np.unique(y)) != {0, 1, 2, 3}:
                raise ValueError(f"A supported source class is missing: {role}")
            if pack["X_seq"].shape != (len(y), 15, 10) or pack["X_tab"].shape != (len(y), 66):
                raise ValueError(f"Invalid feature shapes: {role}")
            if len(set(ids)) != len(ids):
                raise ValueError(f"Duplicate IDs: {role}")
            all_ids.append(set(ids))
            for k, label in enumerate(LABELS):
                counts.append({"partition": role, "class": label, "windows": int((y == k).sum())})
    if any(all_ids[i] & all_ids[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError("Development partitions share window IDs")
    scope = scope_record()
    scope["data_rebuild"] = data_rebuild
    scope["source_runs_current"] = int(manifest.domain.eq("source").sum())
    scope["source_runs_in_original_m3"] = int(old.get("source_development_runs", 33))
    scope["source_pack_sha256"] = baseline_run["source_pack_sha256"]
    scope["window_protocol_sha256"] = digest(windows / "window_protocol.json")
    output.mkdir(parents=True)
    write(output / "scope.json", scope)
    write(output / "supported_settings.json", supported_settings())
    pd.DataFrame(channel_rows).to_csv(output / "channel_compatibility.csv", index=False)
    pd.DataFrame(header_checks).to_csv(output / "raw_header_checks.csv", index=False)
    pd.DataFrame(sample_checks).to_csv(output / "conversion_samples.csv", index=False)
    pd.DataFrame(counts).to_csv(output / "source_class_counts.csv", index=False)
    shutil.copyfile(reproduction_report, output / "reproduction_report.json")
    write(output / "audit.json", {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
          "raw_files_verified": len(header_checks), "representative_files_checked": len(representatives),
          "source_runs_current": scope["source_runs_current"], "source_condition_groups": 3,
          "feature_conversion_checks": len(sample_checks), "status": "passed_for_exploratory_experiments",
          "independent_case_claims_released": False, "target_automatic_escalation_released": False,
          "original_m3_v1_preserved": True, "new_test_or_target_labels_opened": False,
          "prepared_contract_sha256": digest(prepared / "data_contract.json"),
          "script_sha256": digest(__file__)})
    files = sorted(p for p in output.iterdir() if p.is_file())
    write(output / "READY.json", {"version": VERSION, "artifact_sha256": {p.name: digest(p) for p in files}})
    package = data / "m4_readiness_review.zip"
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            archive.write(path, path.name)
    print(f"Readiness passed for descriptive experiments. Raw files: {len(header_checks)}; conversion checks: {len(sample_checks)}")
    print("Independent-case performance claims remain blocked. Existing target data remain exploratory.")
    print(f"Evidence: {output}\nReview ZIP: {package}")
    return output


def verify_readiness(root=ROOT):
    root = Path(root)
    folder = root / "milestone4/evidence/readiness"
    ready = read(folder / "READY.json")
    if ready["version"] != VERSION:
        raise ValueError("Unsupported readiness version")
    for name, expected in ready["artifact_sha256"].items():
        if Path(name).name != name or digest(folder / name) != expected:
            raise ValueError(f"Readiness artifact changed: {name}")
    scope = read(folder / "scope.json")
    if scope["independent_generalization_claims_allowed"] or scope["target_is_sealed_holdout"]:
        raise ValueError("Unsupported performance claims in experiment scope")
    windows = root / "data/aeris_windows_v1"
    if digest(windows / "window_protocol.json") != scope["window_protocol_sha256"]:
        raise ValueError("Window protocol changed after readiness audit")
    for role, expected in scope["source_pack_sha256"].items():
        if role not in {"source_train", "source_validation", "source_calibration"} or digest(windows / f"{role}.npz") != expected:
            raise ValueError("Development inputs changed after readiness audit")
    return scope


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-rebuild", action="store_true")
    args = parser.parse_args()
    try:
        prepare(fresh_rebuild=args.fresh_rebuild)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"STOP - M4 readiness: {exc}")
