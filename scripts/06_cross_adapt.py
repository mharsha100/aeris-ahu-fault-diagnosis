"""Run cross-setting adaptation for a trained AERIS architecture."""

from pathlib import Path
import argparse
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["v0", "v1_attention"], required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--epochs", type=int, default=5, help="Additional gated adaptation budget after initial architecture training")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    try:
        from aeris.adaptation import adapt_hybrid
        adapt_hybrid(PROJECT_ROOT, model_name=args.model, device=args.device,
                     epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
        return 0
    except (ImportError, ValueError, FileNotFoundError, FileExistsError, RuntimeError, OSError, KeyError, AssertionError) as exc:
        print(f"STOP - cross-setting adaptation: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
