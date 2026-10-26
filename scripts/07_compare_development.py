"""Compare completed AERIS runs using saved source-validation reports only."""

import csv
import hashlib
import json
from pathlib import Path
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(root: Path) -> Path:
    models = root / "data/aeris_models"
    rows = []
    baseline_run = read_json(models / "v0/training_run.json")
    baseline_config = read_json(models / "v0/hybrid.json")
    for name in ["v0", "v1_attention", "v0_cross_adapted", "v1_attention_cross_adapted"]:
        folder = models / name
        if not (folder / "READY.json").is_file():
            if name in {"v0", "v1_attention"}:
                raise FileNotFoundError(f"Complete initial training first: {folder}")
            continue
        config = read_json(folder / "hybrid.json")
        if digest(folder / "hybrid.json") != read_json(folder / "READY.json")["hybrid_sha256"]:
            raise ValueError(f"Changed model configuration: {name}")
        run = read_json(folder / "training_run.json")
        if run["source_pack_sha256"] != baseline_run["source_pack_sha256"]:
            raise ValueError(f"Different source data: {name}")
        for contract in ["window_protocol.json", "source_scaler.json"]:
            if config["artifact_sha256"][contract] != baseline_config["artifact_sha256"][contract]:
                raise ValueError(f"Different preprocessing: {name}")
        adapted = name.endswith("_cross_adapted")
        if adapted:
            parent = models / config["parent_bundle"]
            if digest(parent / "hybrid.json") != config["parent_hybrid_sha256"]:
                raise ValueError(f"Adaptation parent changed: {name}")
            initial_run = read_json(parent / "training_run.json")
            history_folder = parent
        else:
            initial_run, history_folder = run, folder
        if name == "v1_attention" and config["comparison_baseline_hybrid_sha256"] != digest(models / "v0/hybrid.json"):
            raise ValueError("V0 changed since V1 initial training")
        with (history_folder / "training_history.csv").open(newline="", encoding="utf-8") as stream:
            completed_epochs = sum(1 for _ in csv.DictReader(stream))
        detailed = read_json(folder / "source_validation_metrics.json")
        component = "after_adaptation" if adapted else f"hybrid_{config['model_version']}"
        metric = detailed[component]
        row = {
            "bundle": name, "architecture_version": config["model_version"],
            "architecture_id": config.get("architecture_id", "rf_tcn_last_v0"),
            "stage": "cross_setting_adaptation" if adapted else "initial_source_training",
            "adaptation_status": run.get("adaptation_status", "not_attempted"),
            "initial_temporal_epochs_completed": completed_epochs,
            "initial_best_epoch": initial_run["best_temporal_epoch"],
            "adaptation_epochs_completed": run.get("completed_adaptation_epochs", 0),
            "macro_f1": metric["macro_f1"], "accuracy": metric["accuracy"],
            "log_loss": metric["log_loss"], "ece_10_bins": metric["ece_10_bins"],
        }
        for label in ["normal", "damper_fault", "cooling_valve_fault", "heating_valve_fault"]:
            row[f"{label}_recall"] = metric["per_class"][label]["recall"]
            row[f"{label}_f1"] = metric["per_class"][label]["f1-score"]
        rows.append(row)
    output = root / "data/aeris_comparison"
    output.mkdir(parents=True, exist_ok=True)
    table = output / "v0_v1_source_validation.csv"
    with table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    notes = (
        "AERIS source-validation development comparison\n"
        "Scores were used for model selection; these are not final target-domain results.\n"
        "Initial and adaptation epoch counts are shown explicitly. Budgets may differ.\n"
        "A skipped/rejected adaptation retains the parent predictor.\n"
        "Source-validation retention does not establish successful cross-setting transfer.\n"
        "No test packs or target reference labels were opened by this comparison command.\n"
    )
    (output / "comparison_notes.txt").write_text(notes, encoding="utf-8")
    with zipfile.ZipFile(output / "development_comparison.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(table, table.name)
        bundle.write(output / "comparison_notes.txt", "comparison_notes.txt")
    for row in rows:
        print(f"{row['bundle']}: source-validation macro F1={row['macro_f1']:.4f}; {row['adaptation_status']}")
    print(notes)
    print(f"Saved: {table}")
    return table


if __name__ == "__main__":
    try:
        compare(PROJECT_ROOT)
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"STOP - development comparison: {exc}")
