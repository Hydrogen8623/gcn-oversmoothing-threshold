
"""Summarize experiment result.json files.

Usage:
    python scripts/summarize.py --outputs outputs/ --out results/
"""

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


EXPECTED_SEEDS = {0, 1, 2, 3, 4}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize experiment results."
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("outputs"),
        help="Root directory containing result.json files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results"),
        help="Directory for summary.csv and summary.md.",
    )
    parser.add_argument(
        "--default-k",
        type=int,
        default=4,
        help="Fallback community count when k is absent in JSON.",
    )
    return parser.parse_args()


def first_value(data, keys):
    """Return the first non-None value among candidate keys."""
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def extract_int(data, keys, path, pattern):
    """Extract an integer from JSON fields or the result path."""
    value = first_value(data, keys)

    if value is not None:
        return int(value)

    match = re.search(pattern, str(path), re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None


def extract_result(path, outputs_root, default_k):
    """Load one result.json and normalize its fields."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    k = extract_int(
        data,
        ["k", "num_communities"],
        path,
        r"(?:^|[/\\_-])k(\d+)(?:[/\\_-]|$)",
    )

    if k is None:
        k = default_k

    layers = extract_int(
        data,
        ["num_layers", "layers", "n_layers"],
        path,
        r"(?:layers?|L)(\d+)",
    )

    seed = extract_int(
        data,
        ["seed"],
        path,
        r"seed[_-]?(\d+)",
    )

    if layers is None or seed is None:
        raise ValueError(
            f"Cannot determine layers or seed from {path}"
        )

    val_acc = first_value(
        data,
        ["best_val_acc", "val_acc", "best_val_accuracy"],
    )

    test_acc = first_value(
        data,
        ["test_acc", "test_accuracy"],
    )

    status = data.get("status", "unknown")

    noise_scale = data.get("noise_scale")

    relative_parent = path.parent.relative_to(outputs_root)

    run_id = f"k{k}_L{layers}_s{seed}"

    if noise_scale is not None:
        run_id += f"_noise{noise_scale}"
    else:
        run_id += f"_{str(relative_parent).replace('/', '_').replace(chr(92), '_')}"

    return {
        "run_id": run_id,
        "k": k,
        "layers": layers,
        "seed": seed,
        "noise_scale": noise_scale,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "status": status,
    }



def collect_results(outputs_root, default_k):
    """Collect result.json files from the target experiment directory.

    If --outputs points to the general outputs/ directory,
    prioritize outputs/diagnostics/snr/ to avoid mixing unrelated runs.
    """
    outputs_root = outputs_root.resolve()

    target_dir = outputs_root / "diagnostics" / "snr"

    if target_dir.is_dir():
        search_root = target_dir
        print(f"Using target experiment directory: {search_root}")
    else:
        search_root = outputs_root
        print(f"Using output directory: {search_root}")

    files = sorted(search_root.rglob("result.json"))

    if not files:
        raise FileNotFoundError(
            f"No result.json found under: {search_root}"
        )

    results = []
    errors = []

    for path in files:
        try:
            results.append(
                extract_result(path, search_root, default_k)
            )
        except (
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            errors.append(f"{path}: {exc}")

    print(f"Found result.json files: {len(files)}")
    print(f"Successfully parsed: {len(results)}")
    print(f"Failed to parse: {len(errors)}")

    # Show how many unrelated files were excluded.
    if target_dir.is_dir():
        all_files = list(outputs_root.rglob("result.json"))
        excluded = len(all_files) - len(files)
        print(f"Excluded unrelated result.json files: {excluded}")

    for error in errors:
        print(f"[WARNING] {error}")

    return results



def write_csv(results, output_path):
    """Write one row per experiment run."""
    fieldnames = [
        "run_id",
        "k",
        "layers",
        "seed",
        "val_acc",
        "test_acc",
        "status",
    ]

    with output_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in sorted(
            results,
            key=lambda x: (
                x["k"],
                x["layers"],
                x["seed"],
                str(x["noise_scale"]),
            ),
        ):
            writer.writerow({
                key: result[key] for key in fieldnames
            })


def format_mean_std(values):
    """Format mean and sample standard deviation."""
    mean = statistics.mean(values)

    if len(values) >= 2:
        std = statistics.stdev(values)
        return f"{mean:.3f} ± {std:.3f}"

    return f"{mean:.3f} ± N/A"


def write_markdown(results, output_path):
    """Aggregate by (k, layers) and write a Markdown report."""
    groups = defaultdict(list)

    for result in results:
        groups[(result["k"], result["layers"])].append(result)

    lines = [
        "# Experiment Summary",
        "",
        "Aggregation key: `(k, layers)`.",
        "",
        "| k | layers | test_acc (mean ± std) | n_seeds | missing_seeds | noise_scales | run_ids |",
        "|---:|---:|---:|---:|---|---|---|",
    ]

    for (k, layers), runs in sorted(groups.items()):
        valid_runs = [
            run for run in runs
            if run["status"] == "completed"
            and run["test_acc"] is not None
            and math.isfinite(float(run["test_acc"]))
        ]

        # Count distinct seeds, not the number of JSON files.
        actual_seeds = {run["seed"] for run in valid_runs}
        missing_seeds = sorted(EXPECTED_SEEDS - actual_seeds)

        test_values = [
            float(run["test_acc"]) for run in valid_runs
        ]

        if test_values:
            acc_summary = format_mean_std(test_values)
        else:
            acc_summary = "N/A"

        noise_scales = sorted({
            str(run["noise_scale"])
            for run in runs
            if run["noise_scale"] is not None
        })

        run_ids = [
            run["run_id"] for run in sorted(
                runs,
                key=lambda x: (
                    x["seed"],
                    str(x["noise_scale"]),
                ),
            )
        ]

        lines.append(
            f"| {k} | {layers} | {acc_summary} "
            f"| {len(actual_seeds)} | "
            f"{missing_seeds if missing_seeds else 'None'} | "
            f"{', '.join(noise_scales) or 'N/A'} | "
            f"{'; '.join(run_ids)} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- `n_seeds` counts distinct seeds among completed runs with valid test accuracy.",
        "- `missing_seeds` is checked against the expected seed set: 0, 1, 2, 3, 4.",
        "- Standard deviation uses the sample standard deviation (ddof=1).",
        "- Groups follow the requested `(k, layers)` key. If multiple noise scales exist, their runs are included in the same group; see `noise_scales` and `run_ids`.",
        "- Failed or incomplete runs are retained in `summary.csv`, but excluded from accuracy aggregation.",
    ])

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()

    outputs_root = args.outputs.resolve()
    output_dir = args.out

    if not outputs_root.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {outputs_root}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    results = collect_results(
        outputs_root,
        args.default_k,
    )

    if not results:
        raise RuntimeError("No valid experiment results were parsed.")

    csv_path = output_dir / "summary.csv"
    md_path = output_dir / "summary.md"

    write_csv(results, csv_path)
    write_markdown(results, md_path)

    print("\nSummary generated:")
    print(f"CSV: {csv_path.resolve()}")
    print(f"Markdown: {md_path.resolve()}")


if __name__ == "__main__":
    main()
