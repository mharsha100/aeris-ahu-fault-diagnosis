"""Train the first complete classical + temporal hybrid using source data only."""

from pathlib import Path
import argparse
import json
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    options = parser.parse_args()
    try:
        from aeris.training import environment_check, train_v0
        if options.check_only:
            print(json.dumps(environment_check(options.device), indent=2))
            return 0
        train_v0(PROJECT_ROOT, device=options.device, epochs=options.epochs,
                 batch_size=options.batch_size, patience=options.patience,
                 seed=options.seed, trees=options.trees, jobs=options.jobs,
                 overwrite=options.overwrite)
        return 0
    except ModuleNotFoundError as exc:
        print(f"STOP - missing training dependency: {exc}")
        print("Install requirements-training.txt and the official CUDA PyTorch wheel in aeris-data.")
        return 1
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, OSError, KeyError) as exc:
        print(f"STOP - V0 training needs review: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
