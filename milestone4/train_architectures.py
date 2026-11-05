"""Train controlled V0/V1/V2 hybrids fresh, with preserved historical bundles."""

import argparse
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from prepare_experiment import verify_readiness


def run(root=ROOT, device="cuda", check_only=False, resume=False):
    from aeris.modeling import load_hybrid, sha256_file
    from aeris.training import environment_check, train_source_hybrid, write_json
    root = Path(root)
    scope = verify_readiness(root)
    models = root / "data/aeris_models"
    kinds = {"v0": "causal_tcn_last", "v1": "causal_tcn_attention", "v2": "causal_tcn_combined"}
    for version, kind in kinds.items():
        environment = environment_check(device, kind)
        print(f"{version.upper()} forward pass: passed on {environment.get('gpu', device)}", flush=True)
    if check_only:
        print("Scope, data provenance and all three device checks passed. No fitting performed.")
        return None
    records = []
    for version in ["v0", "v1", "v2"]:
        name = f"m4_{version}"
        output = models / name
        print(f"\n=== {name}: 50 epochs, fresh RF and NN, fixed 0.50/0.50 fusion ===", flush=True)
        if (output / "READY.json").exists() and resume:
            restored = load_hybrid(output, device="cpu")
            config = restored.configuration
            record = json.loads((output / "training_run.json").read_text())
            if (config["model_version"] != version or config.get("temporal_kind") != kinds[version]
                    or record.get("completed_temporal_epochs") != 50
                    or record.get("early_stopping_enabled") is not False
                    or record.get("fixed_fusion_weight") != 0.5
                    or record.get("experiment_scope") != scope
                    or record.get("source_pack_sha256") != scope["source_pack_sha256"]):
                raise ValueError(f"Existing {name} does not match this controlled experiment")
            for filename, expected in record["source_code_sha256"].items():
                if sha256_file(root / "src/aeris" / filename) != expected:
                    raise ValueError(f"Training code changed since {name}; preserve the old run and version the experiment")
            del restored
            print(f"Verified completed {name}; resuming with the next architecture.", flush=True)
        else:
            output = train_source_hybrid(root, model_version=version, run_name=name, device=device,
                epochs=50, patience=0, seed=42, batch_size=256, trees=200, jobs=-1,
                fusion_weight=0.5, experiment_scope=scope)
            record = json.loads((output / "training_run.json").read_text())
        records.append({"run": name, "model_version": version,
                        "completed_epochs": record["completed_temporal_epochs"],
                        "best_epoch": record["best_temporal_epoch"],
                        "temporal_parameters": record["temporal_parameter_count"],
                        "hybrid_sha256": sha256_file(output / "hybrid.json")})
    output = root / "milestone4/evidence"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "architecture_training.json", {
        "status": "three_initial_architectures_trained", "scope": scope,
        "runs": records, "target_adaptation_completed": False, "feedback_loop_completed": False,
        "ui_implemented": False, "note": "Initial architecture training is one step in the integrated AERIS pipeline, not a completed project."})
    attachment = root / "data/m4_architectures_review.zip"
    with zipfile.ZipFile(attachment, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(output / "architecture_training.json", "architecture_training.json")
        for file in sorted((output / "readiness").iterdir()):
            if file.is_file():
                archive.write(file, f"readiness/{file.name}")
        for entry in records:
            folder = models / entry["run"]
            for file in sorted(folder.iterdir()):
                if file.suffix in {".json", ".csv", ".txt"} or file.name.endswith(".csv.gz"):
                    archive.write(file, f"{entry['run']}/{file.name}")
    print(f"\nAll three initial hybrid architectures completed. Review ZIP: {attachment}")
    return attachment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Verify and skip already complete matching runs")
    args = parser.parse_args()
    try:
        run(device=args.device, check_only=args.check_only, resume=args.resume)
    except (ImportError, ValueError, OSError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"STOP - M4 architecture training: {exc}")
