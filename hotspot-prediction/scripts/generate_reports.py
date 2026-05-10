#!/usr/bin/env python3
"""Generate thesis-ready dataset and experiment summary tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hotspot_prediction.config import DEFAULT_CONFIG
from hotspot_prediction.data.io import read_excel_rows
from hotspot_prediction.data.normalize import (
    build_data1_train_samples,
    build_data2_test_samples,
    exclude_overlapping_train_samples,
    normalize_data1_rows,
    normalize_data2_rows,
)
from hotspot_prediction.reporting import (
    build_ablation_plan,
    build_dataset_summary,
    build_experiment_matrix,
    collect_ablation_results,
    compute_topk_metrics,
    write_table,
)


def read_prediction_records(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hotspot prediction report tables")
    parser.add_argument("--output-dir", default=str(DEFAULT_CONFIG.results_dir))
    parser.add_argument(
        "--predictions-csv",
        default="",
        help="Optional CSV with columns pdb_id,label,prob for Top-K metric generation",
    )
    parser.add_argument(
        "--ablation-results-root",
        default="",
        help="Optional directory containing per-profile result folders to aggregate",
    )
    parser.add_argument("--top-k", default="3,5,10", help="Comma-separated Top-K values")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    data1_mutations = normalize_data1_rows(read_excel_rows(DEFAULT_CONFIG.data1_file))
    data2_rows = normalize_data2_rows(read_excel_rows(DEFAULT_CONFIG.data2_file))
    train_samples = build_data1_train_samples(data1_mutations)
    test_samples = build_data2_test_samples(data2_rows)
    strict_train_samples, _ = exclude_overlapping_train_samples(
        train_samples,
        test_samples,
        policy="uniprot-or-pdb-chain",
    )

    dataset_summary = build_dataset_summary(
        data1_mutations=data1_mutations,
        train_samples=train_samples,
        data2_rows=data2_rows,
        test_samples=test_samples,
        strict_train_samples=strict_train_samples,
    )
    write_table(output_dir / "dataset_summary.csv", dataset_summary)
    write_table(output_dir / "experiment_matrix.csv", build_experiment_matrix())
    write_table(output_dir / "ablation_plan.csv", build_ablation_plan())

    print(f"Wrote dataset summary: {output_dir / 'dataset_summary.csv'}")
    print(f"Wrote experiment matrix: {output_dir / 'experiment_matrix.csv'}")
    print(f"Wrote ablation plan: {output_dir / 'ablation_plan.csv'}")

    if args.predictions_csv:
        prediction_path = Path(args.predictions_csv)
        records = read_prediction_records(prediction_path)
        k_values = [int(item.strip()) for item in args.top_k.split(",") if item.strip()]
        topk_rows = compute_topk_metrics(records, k_values)
        write_table(output_dir / "topk_metrics.csv", topk_rows)
        print(f"Wrote Top-K metrics: {output_dir / 'topk_metrics.csv'}")

    if args.ablation_results_root:
        ablation_rows = collect_ablation_results(args.ablation_results_root)
        write_table(output_dir / "ablation_results_summary.csv", ablation_rows)
        print(f"Wrote ablation result summary: {output_dir / 'ablation_results_summary.csv'}")


if __name__ == "__main__":
    main()
