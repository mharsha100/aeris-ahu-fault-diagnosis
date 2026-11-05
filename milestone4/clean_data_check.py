"""Bounded, fresh-environment data reproduction check; preserves m3-v1.

This checks the data dependencies and a complete uncached 7 MB target archive.
It does not claim a fresh CUDA training install or a fresh 929 MB source download.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TARGET = {
    "name": "lbnl_2019_afdd.zip",
    "url": "https://data.openei.org/files/910/Data%20Sets%20for%20AFDD%20Evauluation%20of%20Building%20FDD%20Algorithms.zip",
    "bytes": 7010416,
    "sha256": "68c58568e3f0e4e7d0f0de3194a2ce7ca80703c057325ee402a94d3eab5a76c1",
    "doi": "10.25984/1824861",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def data_pins():
    pins = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if line.startswith(("numpy==", "pandas==")):
            name, version = line.split("==", 1)
            pins[name] = version
    if set(pins) != {"numpy", "pandas"}:
        raise ValueError("Root requirements must pin both data dependencies")
    return pins


def original_audit_hashes(root):
    folder = root / "milestone3"
    if not (folder / "evidence/run.json").is_file():
        raise FileNotFoundError("Restore the original milestone3 folder first")
    version = json.loads((folder / "evidence/run.json").read_text())["version"]
    if version != "m3-v1":
        raise ValueError("Expected original m3-v1 audit")
    return {p.relative_to(folder).as_posix(): digest(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and (p.parent == folder or "evidence" in p.relative_to(folder).parts)
            and "__pycache__" not in p.parts}


def download_verified(spec, directory):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("Download directory already exists; use a new run folder for an uncached check")
    directory.mkdir(parents=True)
    partial = directory / (spec["name"] + ".partial")
    request = urllib.request.Request(spec["url"], headers={
        "User-Agent": "AERIS-M4-Reproduction/1.0", "Cache-Control": "no-cache"})
    received = 0
    started = time.perf_counter()
    # No local archive is reused. The exact expected size bounds this download.
    with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as stream:
        while True:
            block = response.read(min(1024 * 1024, spec["bytes"] + 1 - received))
            if not block:
                break
            received += len(block)
            if received > spec["bytes"]:
                raise ValueError("Response exceeds the pinned archive size")
            stream.write(block)
    actual = digest(partial)
    if received != spec["bytes"] or actual != spec["sha256"]:
        raise ValueError(f"Pinned snapshot mismatch: bytes={received}, SHA256={actual}")
    with zipfile.ZipFile(partial) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC check failed: {bad}")
        members = archive.namelist()
    final = directory / spec["name"]
    partial.rename(final)
    return final, {**spec, "action": "downloaded_uncached_and_verified",
                   "received_bytes": received, "archive_members": len(members),
                   "crc_check": "passed", "download_and_integrity_seconds": time.perf_counter() - started}


def check(run_dir):
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "reproduction_report.json"
    if report_path.exists():
        raise FileExistsError("A report already exists for this run; use a new run folder")
    report = {"version": "m4-repro-v1", "started_utc": datetime.now(timezone.utc).isoformat(),
              "status": "started", "python": platform.python_version(), "executable": sys.executable,
              "platform": platform.platform(), "script_sha256": digest(__file__),
              "root_requirements_sha256": digest(ROOT / "requirements.txt"),
              "scope": "Fresh NumPy/pandas environment and uncached complete target download/parse",
              "full_training_environment_reproduced": False, "uncached_large_source_download_tested": False}
    before = None
    try:
        if Path(sys.prefix).resolve() != run_dir / "env" or sys.prefix == sys.base_prefix:
            raise ValueError("Run with the fresh venv at <run-dir>/env, not the existing aeris-data environment")
        import numpy as np
        import pandas as pd
        pins = data_pins()
        installed = {"numpy": np.__version__, "pandas": pd.__version__}
        if installed != pins:
            raise ValueError(f"Data dependencies differ from root requirements: {installed} vs {pins}")
        report["installed_data_dependencies"] = installed
        before = original_audit_hashes(ROOT)
        report["m3_v1_original_hashes"] = before
        archive_path, acquisition = download_verified(TARGET, run_dir / "fresh_download")
        report["acquisition"] = acquisition
        parsed = []
        with zipfile.ZipFile(archive_path) as archive:
            for basename in ["SZVAV.csv", "SZCAV.csv"]:
                matches = [n for n in archive.namelist() if Path(n).name == basename]
                if len(matches) != 1:
                    raise ValueError(f"Expected exactly one {basename}")
                with archive.open(matches[0]) as stream:
                    frame = pd.read_csv(stream, low_memory=False)
                names = {str(c).strip(): c for c in frame.columns}
                if "Datetime" not in names or frame.empty:
                    raise ValueError(f"Missing time channel or empty input: {basename}")
                temperatures = {}
                for name in ["AHU: Supply Air Temperature", "AHU: Outdoor Air Temperature",
                             "AHU: Mixed Air Temperature", "AHU: Return Air Temperature"]:
                    if name not in names:
                        raise ValueError(f"Missing required temperature channel: {name}")
                    raw = frame[names[name]]
                    values = pd.to_numeric(raw, errors="coerce")
                    temperatures[name] = {"raw_missing": int(raw.isna().sum()),
                        "new_parse_failures": int((raw.notna() & values.isna()).sum()),
                        "converted_celsius_min": float(((values - 32) * 5 / 9).min()),
                        "converted_celsius_max": float(((values - 32) * 5 / 9).max())}
                parsed.append({"file": basename, "rows": len(frame), "columns": len(frame.columns),
                               "temperatures": temperatures, "imputation_applied": False})
        report["parsed_files"] = parsed
        report["new_temperature_parse_failures"] = sum(v["new_parse_failures"] for f in parsed for v in f["temperatures"].values())
        if report["new_temperature_parse_failures"] != 91:
            raise ValueError("Parsing behavior differs from the recorded 91-failure snapshot; investigate before proceeding")
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if before is not None:
            report["m3_v1_unchanged"] = original_audit_hashes(ROOT) == before
            if not report["m3_v1_unchanged"]:
                report["status"] = "failed_original_audit_changed"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Reproduction status: {report['status']}\nReport: {report_path}", flush=True)
    if report["status"] != "passed":
        raise RuntimeError("Reproduction check did not pass")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        check(args.run_dir)
    except Exception as exc:
        raise SystemExit(f"STOP - clean data reproduction check: {exc}")
