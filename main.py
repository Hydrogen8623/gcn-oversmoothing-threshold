"""Entry point for GCN oversmoothing threshold experiments."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GCN oversmoothing threshold experiments."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/matrix/default.yaml",
        help="Path to the experiment config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved settings without running experiments.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("GCN oversmoothing threshold project.")
    print(f"Config: {args.config}")
    if args.dry_run:
        print("Dry run: no experiments executed.")
        return
    print("Current stage: project scaffold. Training pipeline not implemented yet.")
    print("Next step: implement data generation and GCN training loop.")


if __name__ == "__main__":
    main()