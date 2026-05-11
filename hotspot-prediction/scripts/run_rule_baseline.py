#!/usr/bin/env python3
"""Run the lightweight rule-based baseline on the cached test dataset."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import RESULTS_DIR
from dataset import prepare_test_dataset
import importlib.util as _importlib_util
def _import_from_file(name, path):
    spec = _importlib_util.spec_from_file_location(name, path)
    mod = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
_eval_test = _import_from_file("evaluate_test", str(ROOT / "evaluate_test.py"))
calculate_metrics = _eval_test.calculate_metrics
calculate_balanced_metrics = _eval_test.calculate_balanced_metrics
from hotspot_prediction.baselines.rule_based import predict_rule_based_records
from hotspot_prediction.evaluation_exports import export_metrics_table
from hotspot_prediction.reporting import compute_topk_metrics, write_table


def write_prediction_records(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pdb_id", "residue_id", "aa", "label", "prob"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rule-based hotspot baseline")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--output-dir", default=str(Path(RESULTS_DIR) / "rule_baseline"))
    args = parser.parse_args()

    test_data = prepare_test_dataset(force_reload=args.force_reload)
    records = predict_rule_based_records(test_data)
    output_dir = Path(args.output_dir)

    prediction_csv = output_dir / "test_predictions.csv"
    metrics_csv = output_dir / "test_metrics.csv"
    topk_csv = output_dir / "topk_metrics.csv"

    write_prediction_records(prediction_csv, records)

    labels = np.array([row["label"] for row in records], dtype=int)
    probs = np.array([row["prob"] for row in records], dtype=float)
    metrics = calculate_metrics(labels, probs, args.threshold)
    balanced = calculate_balanced_metrics(labels, probs, args.threshold)
    export_metrics_table(metrics_csv, metrics, balanced, args.threshold)
    write_table(topk_csv, compute_topk_metrics(records, [3, 5, 10]))

    print(f"Rule baseline predictions: {prediction_csv}")
    print(f"Rule baseline metrics: {metrics_csv}")
    print(f"Rule baseline Top-K: {topk_csv}")


if __name__ == "__main__":
    main()
