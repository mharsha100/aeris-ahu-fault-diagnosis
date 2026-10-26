"""Build the first documented AERIS source/target data tables.

Run from the existing project with Python 3.11 and pandas 2.2.3. Outputs go to
data/aeris_prepared/. Original CSVs are not edited. No model is trained here.

Evidence used for this contract:
* G36-1wk/00_explanations.pdf, pp. 3, 7-8: actuator faults and severities;
  pp. 1, 6, 10: sampling, continuous injection, and shared baseline weather.
* conventions.pdf, p. 2, and the actual CSV headers: time and feature units.
* DomainPairs.pdf, p. 2: the explicitly listed source/target fault pairs.
* LBNLDataSynthesisInventory.pdf, pp. 17-19: units and dated fault schedule.

Both branches of the future hybrid must use CORE_FEATURES explicitly. Metadata,
labels, filenames, timestamps, and simulation-only true positions are not inputs.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = DATA_ROOT / "aeris_prepared"
CLASS_IDS = {"normal": 0, "damper_fault": 1, "cooling_valve_fault": 2, "heating_valve_fault": 3}
CORE_FEATURES = [
    "supply_temp_c", "outdoor_temp_c", "mixed_temp_c", "return_temp_c",
    "fan_status", "fan_speed_cmd", "oa_damper_cmd", "cooling_valve_cmd",
    "heating_valve_cmd", "occupied",
]
AUXILIARY_FEATURES = ["return_damper_cmd", "exhaust_damper_cmd"]
TEMPERATURES = set(CORE_FEATURES[:4])
BINARY_FEATURES = {"fan_status", "occupied"}

SOURCE_COLUMNS = {
    "supply_temp_c": "AHU Supply Air Temperature (°K)",
    "outdoor_temp_c": "AHU Outdoor Air Dry Bulb Temperature (°K)",
    "mixed_temp_c": "AHU Mixed Air Temperature (°K)",
    "return_temp_c": "AHU Return Air Temperature (°K)",
    "fan_status": "AHU Supply Air Fan Status (0: off, 1: on)",
    "fan_speed_cmd": "AHU Supply Air Fan Speed Control Signal (fraction, 0: fan speed should be 0% to 1:fan speed should be 100%)",
    "oa_damper_cmd": "Control Signal for AHU Outdoor Air Damper from BAS (fraction, 0: damper should be fully closed to 1: damper should be fully open)",
    "cooling_valve_cmd": "Control Signal for AHU Cooling Coil Valve from BAS (fraction, 0: valve should be fully closed to 1: valve should be fully open)",
    "heating_valve_cmd": "Control Signal for AHU Heating Coil Valve (fraction, 0: valve should be fully closed to 1: valve should be fully open)",
    "occupied": "Indicator if the System Operates in Occupied Mode (0: unoccupied mode, 1: occupied mode)",
    "return_damper_cmd": "Control Signal for AHU Return Air Damper (fraction, 0: damper should be fully closed to 1: damper should be fully open)",
    "exhaust_damper_cmd": "Control Signal for AHU Exhaust Air Damper (fraction, 0: damper should be fully closed to 1: damper should be fully open)",
}
TARGET_COLUMNS = {
    "supply_temp_c": "AHU: Supply Air Temperature",
    "outdoor_temp_c": "AHU: Outdoor Air Temperature",
    "mixed_temp_c": "AHU: Mixed Air Temperature",
    "return_temp_c": "AHU: Return Air Temperature",
    "fan_status": "AHU: Supply Air Fan Status",
    "fan_speed_cmd": "AHU: Supply Air Fan Speed Control Signal",
    "oa_damper_cmd": "AHU: Outdoor Air Damper Control Signal",
    "cooling_valve_cmd": "AHU: Cooling Coil Valve Control Signal",
    "heating_valve_cmd": "AHU: Heating Coil Valve Control Signal",
    "occupied": "Occupancy Mode Indicator",
    "return_damper_cmd": "AHU: Return Air Damper Control Signal",
    "exhaust_damper_cmd": "AHU: Exhaust Air Damper Control Signal",
}

# Only these exact, documented scenario families enter the initial four classes.
STUCK_LEVELS = ("0", "5", "15", "65", "100")
LEAK_LEVELS = ("001", "01")
DOCUMENTED_SOURCE_PAIRS = {
    ("CoolingSeason", "OADamStuck_0"),
    ("CoolingSeason", "OADamStuck_100"),
    ("CoolingSeason", "CooCoiValStuck_100"),
    ("HeatingSeason", "HeaCoiValStuck_0"),
    ("HeatingSeason", "HeaCoiValStuck_5"),
    ("HeatingSeason", "HeaCoiValStuck_15"),
    ("HeatingSeason", "HeaCoiValStuck_100"),
    ("HeatingSeason", "OADamStuck_100"),
}
DOCUMENTED_TARGET_PAIRS = {
    ("SZVAV", "2017-09-18"), ("SZVAV", "2017-09-19"), ("SZVAV", "2017-09-22"),
    ("SZCAV", "2017-03-18"), ("SZCAV", "2017-03-24"), ("SZCAV", "2017-03-25"),
}


def target_schedule() -> dict[str, dict[str, tuple[str, str, str]]]:
    """Date -> (parent class, mechanism, publisher intensity wording)."""
    return {
        "SZVAV": {
            "2017-09-11": ("cooling_valve_fault", "leakage", "50% of max coil valve flow"),
            "2017-09-12": ("heating_valve_fault", "leakage", "40% of max coil valve flow"),
            "2017-09-14": ("heating_valve_fault", "stuck", "100% open"),
            "2017-09-15": ("heating_valve_fault", "stuck", "50% open"),
            "2017-09-18": ("damper_fault", "stuck", "minimum position"),
            "2017-09-19": ("damper_fault", "stuck", "100% open"),
            "2017-09-20": ("normal", "none", "not applicable"),
            "2017-09-21": ("normal", "none", "not applicable"),
            "2017-09-22": ("cooling_valve_fault", "stuck", "100% open"),
            "2017-09-23": ("normal", "none", "not applicable"),
            "2017-09-24": ("normal", "none", "not applicable"),
        },
        "SZCAV": {
            "2017-03-11": ("cooling_valve_fault", "stuck", "0% open"),
            "2017-03-12": ("cooling_valve_fault", "stuck", "100% open"),
            "2017-03-15": ("cooling_valve_fault", "leakage", "5% of max coil valve flow"),
            "2017-03-16": ("cooling_valve_fault", "leakage", "50% of max coil valve flow"),
            "2017-03-18": ("damper_fault", "stuck", "100% open"),
            "2017-03-19": ("damper_fault", "stuck", "50% open"),
            "2017-03-20": ("damper_fault", "leakage", "20% of max damper flow"),
            "2017-03-21": ("damper_fault", "leakage", "50% of max damper flow"),
            "2017-03-22": ("heating_valve_fault", "leakage", "5% of max coil valve flow"),
            "2017-03-23": ("heating_valve_fault", "leakage", "40% of max coil valve flow"),
            "2017-03-24": ("heating_valve_fault", "stuck", "0% open"),
            "2017-03-25": ("heating_valve_fault", "stuck", "100% open"),
            "2017-03-26": ("heating_valve_fault", "stuck", "50% open"),
            "2017-03-31": ("cooling_valve_fault", "stuck", "50% open"),
            "2017-04-01": ("normal", "none", "not applicable"),
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_file(filename: str) -> Path:
    paths = [p for p in DATA_ROOT.rglob("*") if p.is_file() and p.name.casefold() == filename.casefold()
             and not p.is_relative_to(OUTPUT_ROOT)]
    if len(paths) != 1:
        raise ValueError(f"Expected one extracted {filename}; found {len(paths)}: {paths}")
    return paths[0]


def source_specs() -> list[dict]:
    roots = [p for p in DATA_ROOT.rglob("04_G36-1wk") if p.is_dir()]
    if len(roots) != 1:
        raise ValueError(f"Expected one 04_G36-1wk directory; found {len(roots)}")
    specs = []
    for season in ("CoolingSeason", "HeatingSeason", "ShoulderSeason"):
        folder = roots[0] / season
        available = {p.name.casefold(): p for p in folder.glob("*.csv")}
        wanted = [("BaselineSystem", "normal", "none", "not applicable")]
        families = [("OADam", "damper_fault")]
        families.append(("HeaCoiVal", "heating_valve_fault") if season == "HeatingSeason"
                        else ("CooCoiVal", "cooling_valve_fault"))
        for prefix, parent_class in families:
            for level in STUCK_LEVELS:
                wanted.append((f"{prefix}Stuck_{level}", parent_class, "stuck", f"{level}% open"))
            for level in LEAK_LEVELS:
                ratio = "0.001" if level == "001" else "0.01"
                wanted.append((f"{prefix}Lea_{level}", parent_class, "leakage", f"flow-coefficient ratio {ratio}"))
        for stem, parent_class, mechanism, intensity in wanted:
            key = f"{stem}.csv".casefold()
            if key not in available:
                raise FileNotFoundError(f"Documented source file is missing: {folder / (stem + '.csv')}")
            specs.append({"path": available[key], "domain": "source", "dataset": "G36-1wk",
                          "season": season, "class_name": parent_class, "mechanism": mechanism,
                          "intensity": intensity, "scenario": stem})
    return specs


def read_and_convert(spec: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    mapping = SOURCE_COLUMNS if spec["domain"] == "source" else TARGET_COLUMNS
    needed = set(mapping.values()) | ({"time (s)"} if spec["domain"] == "source"
                                      else {"Datetime", "Fault Detection Ground Truth"})
    raw = pd.read_csv(spec["path"], usecols=lambda name: name.strip() in needed, low_memory=False)
    raw.columns = raw.columns.str.strip()
    missing = needed - set(raw.columns)
    if missing or raw.columns.duplicated().any() or raw.empty:
        raise ValueError(f"Invalid schema or empty file: {spec['path']}; missing={sorted(missing)}")

    converted = pd.DataFrame(index=raw.index)
    core_valid = pd.Series(True, index=raw.index)
    audits = []
    for feature, column in mapping.items():
        values = pd.to_numeric(raw[column], errors="coerce")
        bad_text = raw[column].notna() & values.isna()
        nonfinite = values.isin([float("inf"), float("-inf")])
        values = values.mask(nonfinite)
        if feature in TEMPERATURES:
            values = values - 273.15 if spec["domain"] == "source" else (values - 32) * (5 / 9)
            # Broad research sanity bounds; these are not equipment operating limits.
            outside = values.notna() & ~values.between(-60, 100)
        elif feature in BINARY_FEATURES:
            outside = values.notna() & ~values.isin([0, 1])
        else:
            outside = values.notna() & ~values.between(-0.000001, 1.000001)
        valid = values.notna() & ~outside
        if feature in CORE_FEATURES:
            core_valid &= valid
        converted[feature] = values.astype("float32")
        audits.append({
            "domain": spec["domain"], "dataset": spec["dataset"],
            "relative_path": spec["path"].relative_to(DATA_ROOT).as_posix(),
            "feature": feature, "role": "core" if feature in CORE_FEATURES else "auxiliary",
            "raw_missing": int(raw[column].isna().sum()), "nonnumeric": int(bad_text.sum()),
            "nonfinite": int(nonfinite.sum()), "out_of_range": int(outside.sum()),
            "example_nonnumeric": json.dumps(raw.loc[bad_text, column].astype(str).unique()[:5].tolist()),
            "converted_min": None if values.dropna().empty else float(values.min()),
            "converted_max": None if values.dropna().empty else float(values.max()),
        })
    # Metadata is kept outside the feature allowlist.
    if spec["domain"] == "source":
        times = pd.to_numeric(raw["time (s)"], errors="coerce")
        if times.isna().any() or times.isin([float("inf"), float("-inf")]).any():
            raise ValueError(f"Invalid source time: {spec['path']}")
        day = (times / 86400).apply(math.floor).astype(str)
        groups = "source:" + spec["season"] + ":simday_" + day
        timestamps = pd.Series("", index=raw.index)
        classes = pd.Series(spec["class_name"], index=raw.index)
        mechanisms = pd.Series(spec["mechanism"], index=raw.index)
        intensities = pd.Series(spec["intensity"], index=raw.index)
        fault_present = classes.ne("normal").astype(int)
        documented = pd.Series((spec["season"], spec["scenario"]) in DOCUMENTED_SOURCE_PAIRS, index=raw.index)
        label_origin = "G36-1wk/00_explanations.pdf: pp.3,7-8,10"
    else:
        parsed = pd.to_datetime(raw["Datetime"], format="%m/%d/%Y %H:%M", errors="coerce")
        if parsed.isna().any():
            raise ValueError(f"Unparseable target timestamps: {spec['path']}")
        day = parsed.dt.strftime("%Y-%m-%d")
        schedule = target_schedule()[spec["dataset"]]
        unknown = sorted(set(day) - set(schedule))
        if unknown:
            raise ValueError(f"Target dates absent from the documented schedule: {unknown}")
        times = parsed.astype("int64") / 1_000_000_000
        groups = "target:" + spec["dataset"] + ":" + day
        timestamps = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S")
        classes = day.map({date: item[0] for date, item in schedule.items()})
        mechanisms = day.map({date: item[1] for date, item in schedule.items()})
        intensities = day.map({date: item[2] for date, item in schedule.items()})
        fault_present = pd.to_numeric(raw["Fault Detection Ground Truth"], errors="coerce")
        if not fault_present.isin([0, 1]).all() or not fault_present.eq(classes.ne("normal").astype(int)).all():
            raise ValueError(f"Target schedule contradicts binary ground truth: {spec['path']}")
        documented = day.map(lambda date: (spec["dataset"], date) in DOCUMENTED_TARGET_PAIRS)
        label_origin = "LBNLDataSynthesisInventory.pdf: Table 3-4 p.19" if spec["dataset"] == "SZVAV" else "LBNLDataSynthesisInventory.pdf: Table 3-3 p.18"

    if times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError(f"Duplicate or unsorted timestamps require review: {spec['path']}")
    relative_path = spec["path"].relative_to(DATA_ROOT).as_posix()
    file_id = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
    row_ids = pd.Series([f"{file_id}:{i}" for i in range(len(raw))], index=raw.index)
    run_ids = pd.Series(f"source:{file_id}", index=raw.index) if spec["domain"] == "source" else groups
    # Each day/gap/invalid row breaks a segment. Future windows must stay in a segment.
    delta = times.diff()
    boundary = groups.ne(groups.shift()) | (delta.sub(60).abs() > 0.000001)
    boundary |= ~core_valid | ~core_valid.shift(1, fill_value=False)
    segment_number = boundary.cumsum()
    observations = pd.DataFrame({
        "row_id": row_ids, "domain": spec["domain"], "dataset": spec["dataset"],
        "season": spec["season"], "run_id": run_ids, "condition_group": groups,
        "segment_id": file_id + ":segment_" + segment_number.astype(str),
        "sample_time_s": times, "timestamp_local": timestamps,
        "core_valid": core_valid,
        "active_occupied": core_valid & converted["fan_status"].eq(1) & converted["occupied"].eq(1),
    })
    observations = pd.concat([observations, converted], axis=1)
    labels = pd.DataFrame({
        "row_id": row_ids, "run_id": run_ids, "condition_group": groups,
        "class_name": classes, "class_id": classes.map(CLASS_IDS),
        "fault_present": fault_present.astype(int), "mechanism": mechanisms,
        "intensity": intensities, "documented_domain_pair": documented,
        "label_origin": label_origin,
    })
    return observations, labels, audits


def data_contract() -> dict:
    return {
        "contract_version": "aeris-data-v1",
        "class_ids": CLASS_IDS, "core_features": CORE_FEATURES,
        "auxiliary_features_not_enabled_for_training": AUXILIARY_FEATURES,
        "source_columns": SOURCE_COLUMNS, "target_columns_after_whitespace_strip": TARGET_COLUMNS,
        "temperature_conversions": {"source": "C = K - 273.15", "target": "C = (F - 32) * 5/9"},
        "sample_interval_seconds": 60,
        "quality_policy": "Preserve rows and missing values; flag nonnumeric, nonfinite and out-of-range values. No imputation, scaling, clipping or statistical fitting.",
        "sanity_bounds": {"temperature_c": [-60, 100], "fraction_roundoff_tolerance": 0.000001, "binary": [0, 1]},
        "feature_policy": "Only core_features may enter either model branch. All other columns are metadata or disabled auxiliary signals.",
        "source_group_policy": "condition_group blocks the same season/weather day across every fault severity and its baseline; run_id and segment_id are also retained.",
        "target_group_policy": "Keep complete AHU-mode/calendar days together; split before windowing and keep windows within segment_id.",
        "target_label_policy": "target_reference_labels.csv.gz is evaluation/review ground truth; V1 must not use its labels for target adaptation.",
        "time_policy": "Source time is the publisher's simulation seconds. Target timestamps are recorded local naive times; timezone is not asserted or converted to UTC.",
        "known_limits": [
            "Only listed DomainPairs cases are exact publisher-listed pairs. Other included scenarios share parent classes but are not established equivalent faults or severities.",
            "Source valve leakage ratios are flow-coefficient ratios, not the target experiment's percent-of-maximum-flow intensities.",
            "Source heating-valve faults occur in HeatingSeason; cooling-valve faults occur in CoolingSeason and ShoulderSeason. Coverage is not a full class-by-season design.",
            "Source AHU is multi-zone; real target AHU has single-zone VAV and CAV operation. This mapping is a feature contract, not proof of matching dynamics.",
            "BAS requests and overridden control signals can have different semantics during faults. Command-based shortcuts and transfer errors must be evaluated.",
            "Source has one supply temperature setpoint; target has heating/cooling setpoints. They are excluded from the initial common feature set.",
            "Simulation-only true-position fields are excluded. Target labels, dates, IDs, modes and filenames are excluded as predictive features.",
            "Scenario labels indicate an imposed condition; a fault can be inactive or unobservable at some timestamps. active_occupied is available for a restricted first experiment.",
            "Metadata group counts are not counts of independent buildings, independent fault injections, or independent minute observations.",
        ],
    }


def main() -> int:
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Expected project data directory: {DATA_ROOT}")
    sources = source_specs()
    targets = [
        {"path": unique_file("SZVAV.csv"), "domain": "target", "dataset": "SZVAV", "season": "CoolingSeason"},
        {"path": unique_file("SZCAV.csv"), "domain": "target", "dataset": "SZCAV", "season": "HeatingSeason"},
    ]
    print(f"Project: {PROJECT_ROOT}")
    print(f"Selected source files: {len(sources)} | target files: {len(targets)}", flush=True)
    quality_rows, manifest_rows, coverage_rows = [], [], []
    totals = Counter()
    target_days = Counter()
    with TemporaryDirectory(prefix="aeris_prepare_", dir=DATA_ROOT) as tmp:
        stage = Path(tmp)
        for domain, specs in (("source", sources), ("target", targets)):
            label_name = "source_labels.csv.gz" if domain == "source" else "target_reference_labels.csv.gz"
            with gzip.open(stage / f"{domain}_observations.csv.gz", "wt", encoding="utf-8", newline="") as observation_out, gzip.open(stage / label_name, "wt", encoding="utf-8", newline="") as label_out:
                for index, spec in enumerate(specs, 1):
                    relative_path = spec["path"].relative_to(DATA_ROOT).as_posix()
                    print(f"[{domain} {index}/{len(specs)}] {spec['path'].parent.name}/{spec['path'].name}", flush=True)
                    observations, labels, quality = read_and_convert(spec)
                    observations.to_csv(observation_out, index=False, header=index == 1)
                    labels.to_csv(label_out, index=False, header=index == 1)
                    quality_rows.extend(quality)
                    totals[f"{domain}_rows"] += len(observations)
                    totals[f"{domain}_core_valid_rows"] += int(observations["core_valid"].sum())
                    totals[f"{domain}_active_occupied_rows"] += int(observations["active_occupied"].sum())
                    manifest_rows.append({
                        "domain": domain, "dataset": spec["dataset"], "season": spec["season"],
                        "relative_path": relative_path, "sha256": sha256_file(spec["path"]),
                        "rows": len(observations), "core_valid_rows": int(observations["core_valid"].sum()),
                        "condition_groups": observations["condition_group"].nunique(),
                        "class_name": spec.get("class_name", "mixed; see target reference labels"),
                    })
                    info = labels.assign(core_valid=observations["core_valid"], active_occupied=observations["active_occupied"])
                    for (run, condition, class_name), group in info.groupby(["run_id", "condition_group", "class_name"], sort=True):
                        coverage_rows.append({
                            "domain": domain, "dataset": spec["dataset"], "season": spec["season"],
                            "run_id": run, "condition_group": condition, "class_name": class_name,
                            "rows": len(group), "core_valid_rows": int(group["core_valid"].sum()),
                            "active_occupied_rows": int(group["active_occupied"].sum()),
                            "documented_domain_pair": bool(group["documented_domain_pair"].all()),
                        })
                        if domain == "target":
                            target_days[class_name] += 1

        pd.DataFrame(quality_rows).to_csv(stage / "quality_audit.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(manifest_rows).to_csv(stage / "input_manifest.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(coverage_rows).to_csv(stage / "coverage_by_group.csv", index=False, encoding="utf-8-sig")
        mapping_rows = [{"feature": feature, "role": "core" if feature in CORE_FEATURES else "auxiliary",
                         "source_column": SOURCE_COLUMNS[feature], "target_column": TARGET_COLUMNS[feature],
                         "canonical_unit": "degC" if feature in TEMPERATURES else "binary" if feature in BINARY_FEATURES else "fraction"}
                        for feature in SOURCE_COLUMNS]
        pd.DataFrame(mapping_rows).to_csv(stage / "feature_mapping.csv", index=False, encoding="utf-8-sig")
        contract = data_contract()
        (stage / "data_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = {"contract_version": contract["contract_version"], "prepared_utc": datetime.now(timezone.utc).isoformat(),
                   "source_files": len(sources), "target_files": len(targets), "core_feature_count": len(CORE_FEATURES),
                   "source_files_by_class": dict(Counter(spec["class_name"] for spec in sources)),
                   "target_days_by_class": dict(target_days), **dict(totals)}
        lines = ["AERIS - data preparation complete", f"Source files: {len(sources)}", f"Target files: {len(targets)}",
                 f"Core model features: {len(CORE_FEATURES)}", "", "Source files by class:"]
        lines += [f"  {name}: {count}" for name, count in summary["source_files_by_class"].items()]
        lines += ["", "Target documented days by class:"]
        lines += [f"  {name}: {count}" for name, count in target_days.items()]
        lines += ["", *[f"{key}: {value:,}" for key, value in totals.items()], "",
                  f"Output folder: {OUTPUT_ROOT}",
                  "Review attachment: preparation_review.zip (small reports only)",
                  "Target reference labels are separate from target observations.",
                  "Missing/invalid values are flagged; no imputation or scaling has been fitted.",
                  "Splits, time windows, and model training are the next implementation step."]
        text_summary = "\n".join(lines) + "\n"
        (stage / "preparation_summary.txt").write_text(text_summary, encoding="utf-8")
        (stage / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        review_names = ["preparation_summary.txt", "preparation_summary.json", "input_manifest.csv",
                        "quality_audit.csv", "coverage_by_group.csv", "feature_mapping.csv", "data_contract.json"]
        with zipfile.ZipFile(stage / "preparation_review.zip", "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name in review_names:
                bundle.write(stage / name, arcname=name)
        # Publish only after every selected source and target has passed validation.
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        ready_path = OUTPUT_ROOT / "READY.json"
        ready_path.unlink(missing_ok=True)
        for path in sorted(stage.iterdir()):
            os.replace(path, OUTPUT_ROOT / path.name)
        ready_path.write_text(json.dumps({"contract_version": contract["contract_version"], "prepared_utc": summary["prepared_utc"]}, indent=2), encoding="utf-8")
    print("\n" + text_summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"\nSTOP - data preparation needs review: {exc}", flush=True)
        raise SystemExit(1)
