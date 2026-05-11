#!/usr/bin/env python3
"""Evaluate trained checkpoints on the independent data2 test set."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_command(
    root: Path,
    model: str,
    threshold: str,
    threshold_source: str,
    find_threshold: bool,
    force_reload: bool,
    predictions_csv: str,
    metrics_csv: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(root / "evaluate_test.py"),
        "--model",
        model,
        "--threshold",
        str(threshold),
        "--threshold-source",
        threshold_source,
    ]
    if find_threshold:
        cmd.append("--find-threshold")
    if force_reload:
        cmd.append("--force-reload")
    if predictions_csv:
        cmd.extend(["--predictions-csv", predictions_csv])
    if metrics_csv:
        cmd.extend(["--metrics-csv", metrics_csv])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hotspot model on data2")
    parser.add_argument("--model", default="gat", choices=["gat", "gat_v2", "ensemble", "mlp"])
    parser.add_argument("--threshold", default="0.5", help="Numeric threshold for independent test evaluation")
    parser.add_argument(
        "--threshold-source",
        default="fixed",
        choices=["fixed", "validation-mcc"],
        help="Use a fixed threshold or the mean validation MCC threshold from cross_validation_results.csv",
    )
    parser.add_argument("--find-threshold", action="store_true", help="Find threshold from ROC Youden index")
    parser.add_argument("--force-reload", action="store_true", help="Regenerate data2 feature cache")
    parser.add_argument("--predictions-csv", default="", help="Optional residue-level prediction CSV output")
    parser.add_argument("--metrics-csv", default="", help="Optional one-row metrics CSV output")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cmd = build_command(
        root,
        model=args.model,
        threshold=args.threshold,
        threshold_source=args.threshold_source,
        find_threshold=args.find_threshold,
        force_reload=args.force_reload,
        predictions_csv=args.predictions_csv,
        metrics_csv=args.metrics_csv,
    )
    subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
