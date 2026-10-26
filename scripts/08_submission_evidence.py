"""Collect development errors and CPU inference timing for model exploration."""

from pathlib import Path
import argparse
import json
import os
import platform
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def collect(root, model_names, requests=100):
    import numpy as np
    import pandas as pd
    import torch
    from aeris.modeling import CLASS_NAMES, load_hybrid, metrics, sha256_file
    from aeris.training import load_source_partition, write_json

    if requests < 4:
        raise ValueError("Use at least four timed requests")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    windows = root / "data/aeris_windows_v1"
    output = root / "experiments/model_exploration"
    rows = []
    for name in model_names:
        if name not in {"v0", "v1_attention"}:
            raise ValueError("Choose initial architecture bundles v0 or v1_attention")
        bundle = root / "data/aeris_models" / name
        model = load_hybrid(bundle, device="cpu")
        run = json.loads((bundle / "training_run.json").read_text())
        if sha256_file(windows / "source_validation.npz") != run["source_pack_sha256"]["source_validation"]:
            raise ValueError("Validation pack differs from the trained model's provenance")
        pack = load_source_partition(windows, "source_validation", model.protocol)
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        # Explicit inference configuration: single CPU job in the forest.
        model.forest.set_params(n_jobs=1)
        prediction = model.predict(pack["X_seq"], pack["X_tab"])
        score = metrics(pack["y"], prediction["probabilities"])
        meta = pd.read_csv(windows / "window_metadata.csv.gz")
        meta = meta.loc[meta.partition.eq("source_validation")].set_index("window_id")
        if meta.index.duplicated().any() or set(meta.index) != set(pack["window_id"]):
            raise ValueError("Validation metadata does not align with the prepared pack")
        cases = meta.loc[pack["window_id"]].reset_index()
        cases["true_condition"] = [CLASS_NAMES[x] for x in pack["y"]]
        cases["predicted_condition"] = [CLASS_NAMES[x] for x in prediction["class_id"]]
        cases["confidence"] = prediction["confidence"]
        cases["needs_review"] = prediction["needs_review"]
        cases["branch_disagreement"] = prediction["branch_disagreement"]
        wrong = cases.loc[pack["y"] != prediction["class_id"]].copy()
        # First diversify by true/predicted condition, run and day; then take ten.
        examples = wrong.sort_values("confidence", ascending=False).drop_duplicates(
            ["true_condition", "predicted_condition", "run_id", "condition_group"]
        ).head(10)
        examples.to_csv(folder / "error_examples.csv", index=False)
        counts = wrong.groupby(["true_condition", "predicted_condition"]).size().reset_index(name="windows")
        counts.to_csv(folder / "error_counts.csv", index=False)
        write_json(folder / "source_validation_metrics.json", score)
        policy = {"windows": len(cases), "incorrect_windows": len(wrong),
                  "incorrect_at_or_above_review_threshold": int((wrong.confidence >= model.configuration["review_confidence_threshold"]).sum()),
                  "review_threshold": model.configuration["review_confidence_threshold"],
                  "review_fraction": float(prediction["needs_review"].mean()),
                  "examples_selected": len(examples),
                  "examples_are_independent_cases": False}
        write_json(folder / "error_summary.json", policy)
        # Same deterministic stratified development sample for each architecture.
        rng = np.random.default_rng(42)
        if requests > len(pack["y"]):
            raise ValueError("Requested more distinct timing windows than validation contains")
        selected = []
        for label in range(4):
            candidates = np.flatnonzero(pack["y"] == label)
            count = requests // 4 + (label < requests % 4)
            selected.extend(rng.choice(candidates, min(count, len(candidates)), replace=False).tolist())
        if len(selected) < requests:
            remaining = np.setdiff1d(np.arange(len(pack["y"])), selected)
            selected.extend(rng.choice(remaining, requests - len(selected), replace=False).tolist())
        rng.shuffle(selected)
        for ix in selected[:3]:
            model.predict(pack["X_seq"][ix:ix+1], pack["X_tab"][ix:ix+1])
        timings = []
        for ix in selected:
            started = time.perf_counter()
            model.predict(pack["X_seq"][ix:ix+1], pack["X_tab"][ix:ix+1])
            timings.append({"window_id": str(pack["window_id"][ix]),
                            "seconds": time.perf_counter() - started})
        pd.DataFrame(timings).to_csv(folder / "inference_timing_samples.csv", index=False)
        times = np.array([x["seconds"] for x in timings])
        timing = {"requests": len(times), "warmup_requests": min(3, len(selected)),
                  "p50_seconds": float(np.quantile(times, .5)), "p95_seconds": float(np.quantile(times, .95)),
                  "max_seconds": float(times.max()), "under_5_seconds": int((times < 5).sum()),
                  "device": "cpu", "torch_threads": torch.get_num_threads(), "forest_jobs": 1,
                  "python": platform.python_version(), "platform": platform.platform(),
                  "processor": platform.processor(), "torch": torch.__version__,
                  "included": "Input shape/finite validation, forest/TCN inference, calibration and review rule on prepared windows",
                  "excluded": "Disk loading, raw preprocessing, UI rendering, model loading and startup",
                  "section_2_1_target": "At least 95 of 100 full local UI requests render within 5 seconds",
                  "full_ui_target_verified": False,
                  "model_config_sha256": sha256_file(bundle / "hybrid.json")}
        write_json(folder / "inference_timing.json", timing)
        for filename in ["training_run.json", "training_history.csv", "training_environment.json", "hybrid.json"]:
            shutil.copyfile(bundle / filename, folder / filename)
        completed = len(pd.read_csv(bundle / "training_history.csv"))
        rows.append({"model": name, "validation_windows": len(pack["y"]),
                     "macro_f1": score["macro_f1"], "accuracy": score["accuracy"],
                     "completed_epochs": completed, "best_epoch": run["best_temporal_epoch"],
                     "cpu_inference_p95_seconds": timing["p95_seconds"], "timed_requests": len(times)})
        print(f"{name}: F1={score['macro_f1']:.4f}; {len(examples)} error examples; CPU p95={timing['p95_seconds']:.6f}s", flush=True)
    # Merge rows to preserve V0 evidence collected while V1 was still training.
    table = output / "experiment_results.csv"
    frame = pd.DataFrame(rows)
    if table.exists():
        frame = pd.concat([pd.read_csv(table), frame]).drop_duplicates("model", keep="last")
    frame.to_csv(table, index=False)
    notes = (
        "# Model experimentation evidence\n\n"
        "These are measured source-validation development results. Final test data and target labels are unused.\n\n"
        "The initial stage explored is AHU diagnosis with a Random Forest + TCN hybrid. "
        "V1 changes final-step pooling to learned attention pooling. Both components are trained fresh. "
        "V0 completed 37 epochs (best 29) in the original project run; V1 uses 50 by default. "
        "Actual budgets are in each training history; this difference prevents attributing any change solely to attention.\n\n"
        "Error examples are selected deterministically from saved-window predictions, diversified by "
        "true/predicted class, run and day. Overlapping windows are not independent fault events. "
        "The tables identify symptoms for review; they do not establish physical causes.\n\n"
        "Timing measures CPU inference on prepared windows with three warmups, four or fewer torch threads "
        "and one forest job. It excludes loading, raw preprocessing and the UI. "
        "It is a partial measurement against the five-second full-request target; the full UI target remains unverified.\n\n"
        "Scripts: ../../scripts/04_train_v0.py, ../../scripts/05_train_v1.py, ../../scripts/08_submission_evidence.py. "
        "Keep source datasets and model weights outside Git. No new dependencies are required.\n"
    )
    (output / "README.md").write_text(notes, encoding="utf-8")
    archive_path = root / "data/model_exploration_evidence.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output))
    print(f"Git-readable evidence: {output}\nReview attachment: {archive_path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["v0", "v1_attention"], default=["v0", "v1_attention"])
    parser.add_argument("--requests", type=int, default=100)
    args = parser.parse_args()
    try:
        collect(ROOT, args.models, args.requests)
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as exc:
        raise SystemExit(f"STOP - submission evidence: {exc}")
