"""Inventory extracted AERIS datasets and their CSV headers.

Place this file in <project>/scripts/ and run it with the aeris-data Python.
Reports are written to <project>/data/aeris_inventory/. Raw files are only read.
This step does not assign fault labels, clean data, or train a model.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
REPORT_ROOT = DATA_ROOT / "aeris_inventory"


def dataset_name(path: Path) -> str:
    """Identify dataset scope without interpreting a fault filename."""
    parts = {part.casefold() for part in path.relative_to(DATA_ROOT).parts[:-1]}
    if "04_g36-1wk" in parts:
        return "G36-1wk"
    if path.name.casefold() in {"szvav.csv", "szcav.csv"}:
        return "LBNL-target-candidate"
    return "other"


def season_name(path: Path) -> str:
    for part in path.relative_to(DATA_ROOT).parts[:-1]:
        if part.casefold() in {"coolingseason", "shoulderseason", "heatingseason"}:
            return part
    return ""


def read_header(path: Path) -> tuple[list[str], str]:
    # Preserve the publisher's names exactly. Do not guess units or mappings.
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header:
        raise ValueError("CSV has no header")
    signature = hashlib.sha256(
        json.dumps(header, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return header, signature


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    print(f"Project: {PROJECT_ROOT}")
    print(f"Data:    {DATA_ROOT}")
    if not DATA_ROOT.is_dir():
        print("ERROR: data/ was not found. Put this script inside the project's scripts/ folder.")
        return 1

    # Exclude generated reports on every run, including when inventory is rerun.
    files = sorted(
        (
            path
            for path in DATA_ROOT.rglob("*")
            if path.is_file() and not path.is_relative_to(REPORT_ROOT)
        ),
        key=lambda path: path.relative_to(DATA_ROOT).as_posix().casefold(),
    )
    csv_files = [path for path in files if path.suffix.casefold() == ".csv"]
    pdf_files = [path for path in files if path.suffix.casefold() == ".pdf"]
    zip_files = [path for path in files if path.suffix.casefold() == ".zip"]

    inventory = []
    schemas = {}
    for path in csv_files:
        dataset = dataset_name(path)
        row = {
            "dataset": dataset,
            "season": season_name(path),
            "relative_path": path.relative_to(DATA_ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "column_count": "",
            "schema_id": "",
            "header_error": "",
        }
        # Read only headers of the selected source/target scopes, never whole CSVs.
        if dataset != "other":
            try:
                header, signature = read_header(path)
                row["column_count"] = len(header)
                row["schema_id"] = signature
                schemas[signature] = header
            except (OSError, UnicodeError, csv.Error, ValueError) as exc:
                row["header_error"] = f"{type(exc).__name__}: {exc}"
        inventory.append(row)

    schema_rows = [
        {"schema_id": signature, "column_index": index, "column_name": name}
        for signature, header in sorted(schemas.items())
        for index, name in enumerate(header, start=1)
    ]
    documentation = [
        {
            "filename": path.name,
            "relative_path": path.relative_to(DATA_ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
        }
        for path in pdf_files
    ]

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(
        REPORT_ROOT / "csv_inventory.csv",
        ["dataset", "season", "relative_path", "size_bytes", "column_count", "schema_id", "header_error"],
        inventory,
    )
    write_csv(
        REPORT_ROOT / "schema_inventory.csv",
        ["schema_id", "column_index", "column_name"],
        schema_rows,
    )
    write_csv(
        REPORT_ROOT / "documentation_inventory.csv",
        ["filename", "relative_path", "size_bytes"],
        documentation,
    )

    source = [row for row in inventory if row["dataset"] == "G36-1wk"]
    target = [row for row in inventory if row["dataset"] == "LBNL-target-candidate"]
    errors = [row for row in inventory if row["header_error"]]
    lines = [
        "AERIS - extracted data inventory",
        f"Extracted CSV files found: {len(csv_files)}",
        f"G36-1wk CSV files: {len(source)}",
        f"SZVAV/SZCAV candidate files: {len(target)}",
        f"PDF documentation files: {len(pdf_files)}",
        f"Cached ZIP files: {len(zip_files)}",
        f"Distinct inspected CSV headers: {len(schemas)}",
        f"Header read errors: {len(errors)}",
        "",
        "G36-1wk files by season:",
    ]
    for season, count in sorted(Counter(row["season"] or "unspecified" for row in source).items()):
        lines.append(f"  {season}: {count}")
    lines.extend(["", "Target candidate paths (relative to data/):"])
    for row in target:
        lines.append(f"  {row['relative_path']} | columns={row['column_count']}")

    lines.extend(["", "Relevant documentation paths (relative to data/):"])
    expected_docs = {"domainpairs.pdf", "conventions.pdf", "lbnldatasynthesisinventory.pdf"}
    for path in pdf_files:
        if path.name.casefold() in expected_docs or (
            dataset_name(path) == "G36-1wk" and path.name.casefold() == "00_explanations.pdf"
        ):
            lines.append(f"  {path.relative_to(DATA_ROOT).as_posix()}")
    for row in errors:
        lines.append(f"HEADER ERROR: {row['relative_path']} | {row['header_error']}")
    if not source:
        lines.append("CHECK: no extracted CSVs under a folder named 04_G36-1wk were found.")
    if not target:
        lines.append("CHECK: no extracted SZVAV.csv or SZCAV.csv files were found.")
    lines.extend([
        "",
        f"Reports: {REPORT_ROOT}",
        "Created: csv_inventory.csv, schema_inventory.csv, documentation_inventory.csv, summary.txt",
        "Inventory only: file counts are not verified fault-class counts or independent sample counts.",
    ])
    summary = "\n".join(lines) + "\n"
    (REPORT_ROOT / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    return 1 if errors or not source or not target else 0


if __name__ == "__main__":
    raise SystemExit(main())
