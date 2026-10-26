"""Retired command: the original implementation used the wrong version mapping."""


def main() -> int:
    print("This legacy command used an incorrect version mapping and is retired.")
    print("Train architecture V1: python scripts/05_train_v1.py --device cuda --epochs 50")
    print("Then cross-adapt it: python scripts/06_cross_adapt.py --model v1_attention --device cuda")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
