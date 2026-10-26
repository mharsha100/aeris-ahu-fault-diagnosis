"""Train the V1 Random Forest + attention-pooled temporal hybrid from scratch."""

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
    parser.add_argument("--patience", type=int, default=0, help="0 runs the full epoch budget; positive values enable early stopping")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        from aeris.training import environment_check, train_v1
        if args.check_only:
            print(json.dumps(environment_check(args.device, "causal_tcn_attention"), indent=2))
            return 0
        train_v1(PROJECT_ROOT, device=args.device, epochs=args.epochs,
                 batch_size=args.batch_size, patience=args.patience,
                 seed=args.seed, trees=args.trees, jobs=args.jobs)
        return 0
    except (ImportError, ValueError, FileNotFoundError, FileExistsError, RuntimeError, OSError, KeyError) as exc:
        print(f"STOP - V1 training: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
