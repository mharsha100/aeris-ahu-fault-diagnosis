"""AERIS M3 audit and deterministic preparation. No model is trained.

Run: python pipeline.py --cache PATH_TO_WARMUP_DATA --output PATH
Source groups remain intact. Target labels were already exposed by the warm-up;
all target records therefore have the role exploratory_target, never sealed_test.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

VERSION = "m3-v1"
TEMPS = ["supply_air_c", "outdoor_air_c", "mixed_air_c", "return_air_c"]
SOURCE_PREFIX = {
    "supply_air_c": "AHU Supply Air Temperature (",
    "outdoor_air_c": "AHU Outdoor Air Dry Bulb Temperature (",
    "mixed_air_c": "AHU Mixed Air Temperature (",
    "return_air_c": "AHU Return Air Temperature (",
    "fan_on": "AHU Supply Air Fan Status (",
    "fan_speed": "AHU Supply Air Fan Speed Control Signal (",
    "oa_command": "Control Signal for AHU Outdoor Air Damper from BAS (",
    "cooling_command": "Control Signal for AHU Cooling Coil Valve from BAS (",
    "heating_command": "Control Signal for AHU Heating Coil Valve (",
    "occupied": "Indicator if the System Operates in Occupied Mode (",
}
TARGET_NAMES = dict(zip(SOURCE_PREFIX, [
    "AHU: Supply Air Temperature", "AHU: Outdoor Air Temperature",
    "AHU: Mixed Air Temperature", "AHU: Return Air Temperature",
    "AHU: Supply Air Fan Status", "AHU: Supply Air Fan Speed Control Signal",
    "AHU: Outdoor Air Damper Control Signal", "AHU: Cooling Coil Valve Control Signal",
    "AHU: Heating Coil Valve Control Signal", "Occupancy Mode Indicator",
]))
SCENARIOS = {
    "SZVAV": {
        "09-11": "cooling_valve_leak", "09-12": "heating_valve_leak",
        "09-14": "heating_valve_stuck", "09-15": "heating_valve_stuck",
        "09-18": "oa_damper_stuck", "09-19": "oa_damper_stuck",
        "09-20": "normal", "09-21": "normal", "09-22": "cooling_valve_stuck",
        "09-23": "normal", "09-24": "normal",
    },
    "SZCAV": {
        "03-11": "cooling_valve_stuck", "03-12": "cooling_valve_stuck",
        "03-15": "cooling_valve_leak", "03-16": "cooling_valve_leak",
        "03-18": "oa_damper_stuck", "03-19": "oa_damper_stuck",
        "03-20": "oa_damper_leak", "03-21": "oa_damper_leak",
        "03-22": "heating_valve_leak", "03-23": "heating_valve_leak",
        "03-24": "heating_valve_stuck", "03-25": "heating_valve_stuck",
        "03-26": "heating_valve_stuck", "03-31": "cooling_valve_stuck", "04-01": "normal",
    },
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str, allow_nan=False), encoding="utf-8")


def review_gate(frame, setting, score, bounds=None, schema_ok=True):
    """60 consecutive 1-minute samples; hard scope checks override any score.

    bounds, when supplied, must come from training only; M3 deliberately fits none.
    This checks policy behavior, not trained-model accuracy or calibration.
    """
    reasons = []
    if not schema_ok or not set(SOURCE_PREFIX).issubset(frame.columns):
        return {"state": "Needs review", "reasons": ["invalid_schema"], "score": score}
    if setting not in {"G36_cooling", "G36_heating"}:
        reasons.append("unsupported_setting")
    if len(frame) != 60:
        reasons.append("invalid_interval_length")
    numeric = frame[list(SOURCE_PREFIX)]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        reasons.append("missing_or_nonfinite_required_input")
    if ((frame[TEMPS] < -40) | (frame[TEMPS] > 70)).any().any():
        reasons.append("temperature_outside_physical_screen")
    commands = ["fan_speed", "oa_command", "cooling_command", "heating_command"]
    if ((frame[commands] < 0) | (frame[commands] > 1)).any().any():
        reasons.append("command_outside_0_1")
    if not frame["fan_on"].eq(1).all() or not frame["occupied"].eq(1).all():
        reasons.append("unsupported_off_or_unoccupied_mode")
    if "time_s" not in frame or not frame["time_s"].diff().dropna().eq(60).all():
        reasons.append("nonconsecutive_time")
    if bounds is None:
        reasons.append("training_support_not_fitted")
    else:
        for c, (low, high) in bounds.items():
            # At least 6/60 readings outside frozen training min/max triggers review.
            if ((frame[c] < low) | (frame[c] > high)).sum() >= 6:
                reasons.append("outside_training_support:" + c)
    if score < 0.80:
        reasons.append("low_score_provisional_threshold")
    return {"state": "Needs review" if reasons else "Candidate allowed", "reasons": reasons, "score": score}


def control_tests():
    base = pd.DataFrame({c: [20.0 if c in TEMPS else 1.0 if c in {"fan_on", "occupied"} else 0.5] * 60 for c in SOURCE_PREFIX})
    base["time_s"] = np.arange(60) * 60
    bounds = {c: [0, 40] for c in TEMPS}  # Synthetic fixture, NOT learned support.
    trials = []
    def check(name, f=base, setting="G36_cooling", score=.99, limits=bounds, reason=None):
        result = review_gate(f, setting, score, limits)
        passed = (result["state"] == "Candidate allowed") if reason is None else (result["state"] == "Needs review" and reason in result["reasons"])
        trials.append({"test": name, "passed": passed, **result})
    check("valid_fixture")
    check("high_score_unsupported_target", setting="SZVAV_summer", reason="unsupported_setting")
    check("high_score_shoulder", setting="G36_shoulder", reason="unsupported_setting")
    f = base.copy(); f.loc[:5, "outdoor_air_c"] = 41
    check("high_score_6_of_60_out_of_support", f, reason="outside_training_support:outdoor_air_c")
    f = base.copy(); f.loc[:4, "outdoor_air_c"] = 41
    check("5_of_60_boundary_does_not_trigger", f)
    f = base.copy(); f.loc[0, "mixed_air_c"] = np.nan
    check("missing_required_reading", f, reason="missing_or_nonfinite_required_input")
    f = base.copy(); f.loc[0, "time_s"] = -60
    check("time_gap", f, reason="nonconsecutive_time")
    f = base.copy(); f.loc[0, "fan_on"] = 0
    check("fan_off", f, reason="unsupported_off_or_unoccupied_mode")
    check("no_fitted_support", limits=None, reason="training_support_not_fitted")
    check("low_score", score=.79, reason="low_score_provisional_threshold")
    assert all(x["passed"] for x in trials)
    return trials


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", type=Path, required=True); ap.add_argument("--output", type=Path, default=Path("run"))
    args = ap.parse_args(); start = time.perf_counter()
    out = args.output; ev = out / "evidence"; proc = out / "processed"
    ev.mkdir(parents=True, exist_ok=True); proc.mkdir(parents=True, exist_ok=True)
    source_zip = args.cache / "Public_ScientificData_AHUFaults.zip"
    target_zip = args.cache / "lbnl_2019_afdd.zip"
    inventory, manifest, quality, mappings, days = [], [], [], [], []
    archive_hashes = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in [source_zip, target_zip]}
    archives = [ZipFile(source_zip), ZipFile(target_zip)]
    work = []
    for name in archives[0].namelist():
        if "/04_G36-1wk/" in name and name.endswith(".csv"):
            p = Path(name); season = p.parent.name
            selected = p.stem.startswith(("BaselineSystem", "OADamStuck_", "CooCoiValStuck_", "HeaCoiValStuck_"))
            inventory.append({"member": name, "season": season, "bytes": archives[0].getinfo(name).file_size, "selected": selected})
            if selected: work.append(("source", name, season, archives[0]))
    for name in archives[1].namelist():
        if Path(name).name in {"SZVAV.csv", "SZCAV.csv"}:
            work.append(("target", name, "summer" if "SZVAV" in name else "winter", archives[1]))
    for domain, name, season, arc in work:
        raw = arc.read(name); df = pd.read_csv(__import__("io").BytesIO(raw), low_memory=False)
        headers_trimmed = sum(c != c.strip() for c in df.columns)
        df.columns = df.columns.str.strip()
        assert not df.columns.duplicated().any()
        stem = Path(name).stem; run_id = f"{season}__{stem}"
        role = "source_development" if domain == "source" else "exploratory_target"
        mapped = {}; selected_names = []
        for canonical, prefix in SOURCE_PREFIX.items():
            matches = [c for c in df if c.startswith(prefix)] if domain == "source" else [TARGET_NAMES[canonical]]
            assert len(matches) == 1 and matches[0] in df, (name, canonical, matches)
            original = matches[0]; selected_names.append(original)
            numeric = pd.to_numeric(df[original], errors="coerce")
            mapped[canonical] = numeric - 273.15 if domain == "source" and canonical in TEMPS else (numeric - 32) * 5/9 if canonical in TEMPS else numeric
            if run_id.endswith("BaselineSystem") or domain == "target":
                mappings.append({"domain":domain,"season":season,"canonical":canonical,"raw_column":original,"conversion":"K to C" if domain=="source" and canonical in TEMPS else "F to C" if canonical in TEMPS else "identity","new_parse_missing":int((numeric.isna() & df[original].notna()).sum())})
        prepared = pd.DataFrame(mapped)
        all_numeric = df.drop(columns=["Datetime"],errors="ignore").apply(pd.to_numeric,errors="coerce")
        new_missing = int(all_numeric.isna().sum().sum() - df.drop(columns=["Datetime"],errors="ignore").isna().sum().sum())
        if domain == "source":
            ts = pd.to_numeric(df["time (s)"], errors="raise")
            case = "normal" if stem == "BaselineSystem" else "oa_damper_stuck" if stem.startswith("OADam") else "cooling_valve_stuck" if stem.startswith("Coo") else "heating_valve_stuck"
            group_id = f"G36_{season}_shared_weather_week"
            prepared.insert(0,"time_s",ts)
            label_counts = {}
        else:
            dt = pd.to_datetime(df["Datetime"],format="%m/%d/%Y %H:%M",errors="raise")
            ts = dt.astype("datetime64[ns]").astype("int64") / 1e9
            prepared.insert(0,"time_s",ts)
            prepared.insert(1,"timestamp_local_unspecified",df["Datetime"])
            case = "mixed_daily_cases"; group_id = stem + "_FLEXLAB"
            label_counts = {str(k):int(v) for k,v in df["Fault Detection Ground Truth"].value_counts().items()}
            labels=[]
            for date, ix in df.groupby(dt.dt.strftime("%Y-%m-%d")).groups.items():
                c = SCENARIOS[stem].get(date[5:],"unmapped")
                assert c != "unmapped", date
                sub = df.loc[ix,"Fault Detection Ground Truth"]
                assert sub.eq(0 if c == "normal" else 1).all()
                days.append({"system":stem,"season":season,"date":date,"class":c,"rows":len(ix),"role":role})
                labels.extend([{"row_id":int(i),"date":date,"scenario":c,"binary_label":int(df.loc[i,"Fault Detection Ground Truth"])} for i in ix])
            pd.DataFrame(labels).sort_values("row_id").to_csv(proc/(run_id+"__labels_quarantined.csv"),index=False)
        required_missing = prepared[list(SOURCE_PREFIX)].isna().any(axis=1)
        range_bad = ((prepared[TEMPS] < -40) | (prepared[TEMPS] > 70)).any(axis=1)
        unit_cols = ["fan_speed","oa_command","cooling_command","heating_command"]
        range_bad |= ((prepared[unit_cols] < 0) | (prepared[unit_cols] > 1)).any(axis=1)
        flags = pd.DataFrame({"row_id":np.arange(len(df)),"required_missing":required_missing,"outside_physical_screen":range_bad,"unoccupied_or_fan_off":~(prepared.fan_on.eq(1)&prepared.occupied.eq(1))})
        flags.to_csv(proc/(run_id+"__quality_flags.csv"),index=False)
        prepared.to_csv(proc/(run_id+"__features.csv"),index=False,float_format="%.10g")
        assert not any("truth" in c.lower() or "stuck position" in c.lower() or "class" in c.lower() for c in prepared.columns)
        assert ts.diff().dropna().ge(60).all(), "Non-increasing or sub-minute timestamps"
        invalid_tokens={c:df.loc[pd.to_numeric(df[c],errors="coerce").isna() & df[c].notna(),c].value_counts().to_dict() for c in df if c!='Datetime'}
        quality.append({"domain":domain,"run_id":run_id,"rows":len(df),"raw_fields":len(df.columns),"headers_trimmed":headers_trimmed,"raw_missing_cells":int(df.isna().sum().sum()),"new_numeric_missing_cells":new_missing,"exact_duplicate_rows":int(df.duplicated().sum()),"duplicate_timestamps":int(ts.duplicated().sum()),"time_steps_not_60":int(ts.diff().dropna().ne(60).sum()),"required_missing_rows":int(required_missing.sum()),"range_flagged_rows":int(range_bad.sum()),"fan_on_occupied_rows":int((prepared.fan_on.eq(1)&prepared.occupied.eq(1)).sum()),"binary_labels":label_counts,"invalid_tokens":{k:v for k,v in invalid_tokens.items() if v}})
        manifest.append({"domain":domain,"member":name,"run_id":run_id,"season":season,"class":case,"role":role,"group_id":group_id,"raw_sha256":hashlib.sha256(raw).hexdigest(),"raw_bytes":len(raw),"rows":len(df),"processed_sha256":digest(proc/(run_id+"__features.csv"))})
    for arc in archives: arc.close()
    pd.DataFrame(inventory).to_csv(ev/"source_inventory.csv",index=False)
    pd.DataFrame(manifest).to_csv(ev/"manifest.csv",index=False)
    pd.DataFrame(days).to_csv(ev/"target_day_inventory.csv",index=False)
    pd.DataFrame(mappings).to_csv(ev/"channel_map.csv",index=False)
    write_json(ev/"quality.json",quality)
    tests=control_tests(); write_json(ev/"control_tests.json",tests)
    # No valid model split is released: one shared weather week per source season.
    # Case counts never treat minute rows, adjacent windows, or severities as replications.
    group_roles = {}
    for m in manifest:
        group_roles.setdefault(m['group_id'],set()).add(m['role'])
    overlap = sum(len(roles)>1 for roles in group_roles.values())
    assert overlap == 0
    split={"status":"blocked_insufficient_independent_cases", "source_development_runs":sum(m['domain']=='source' for m in manifest),"source_condition_groups":len({m['group_id'] for m in manifest if m['domain']=='source'}),"target_exploratory_days":len(days),"sealed_target_cases":0,"trained_models":0,"windows_created":0,"learned_transform_fits":0,"minimum_final_cases_per_class_per_season":5,"source_validation_minimum_per_class_per_season":5,"source_train_minimum_per_class_per_season":5,"group_overlap_between_roles":overlap,"target_labels_previously_exposed":True}
    assert len([m for m in manifest if m['domain']=='source'])==33
    assert len({m['raw_sha256'] for m in manifest})==len(manifest)
    write_json(ev/"split_status.json",split)
    run={"version":VERSION,"audit_utc":datetime.now(timezone.utc).isoformat(),"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,"platform":platform.platform(),"pipeline_sha256":digest(__file__),"archives":archive_hashes,"elapsed_seconds":round(time.perf_counter()-start,2),"checks_passed":len(tests),"split":split}
    write_json(ev/"run.json",run)
    print(json.dumps(run,indent=2))


if __name__ == "__main__":
    main()
