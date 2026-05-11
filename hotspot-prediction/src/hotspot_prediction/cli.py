"""Command line entry points for hotspot prediction workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from hotspot_prediction.config import DEFAULT_CONFIG
from hotspot_prediction.data.io import read_excel_rows, write_csv_rows
from hotspot_prediction.data.normalize import (
    build_data1_train_samples,
    build_data2_test_samples,
    deduplicate_data2_rows,
    exclude_overlapping_train_samples,
    normalize_data1_rows,
    normalize_data2_rows,
)


def prepare_data(args: argparse.Namespace) -> None:
    data1_file = Path(args.data1)
    data2_file = Path(args.data2)
    output_dir = Path(args.output_dir)

    data1_rows = read_excel_rows(data1_file)
    data2_rows = read_excel_rows(data2_file)

    data1_normalized = normalize_data1_rows(data1_rows)
    data2_normalized_raw = normalize_data2_rows(data2_rows)
    data2_normalized, removed_data2_rows = deduplicate_data2_rows(data2_normalized_raw)
    train_samples = build_data1_train_samples(data1_normalized)
    test_samples = build_data2_test_samples(data2_normalized)
    original_train_samples = len(train_samples)
    train_samples, removed_train_samples = exclude_overlapping_train_samples(
        train_samples,
        test_samples,
        policy=args.overlap_policy,
    )

    write_csv_rows(output_dir / "data1_mutations.csv", data1_normalized)
    write_csv_rows(output_dir / "data2_test_samples.csv", data2_normalized)
    write_csv_rows(output_dir / "train_val_samples.csv", train_samples)
    write_csv_rows(output_dir / "legacy_data2_test_samples.csv", test_samples)

    print(f"Wrote {len(data1_normalized)} data1 mutation rows to {output_dir / 'data1_mutations.csv'}")
    print(
        f"Wrote {len(data2_normalized)} data2 test rows to {output_dir / 'data2_test_samples.csv'} "
        f"(removed {removed_data2_rows} exact duplicate rows)"
    )
    print(f"Wrote {len(train_samples)} train/validation samples to {output_dir / 'train_val_samples.csv'}")
    print(f"Wrote {len(test_samples)} legacy test samples to {output_dir / 'legacy_data2_test_samples.csv'}")
    print(
        f"Overlap policy {args.overlap_policy!r}: kept {len(train_samples)} of "
        f"{original_train_samples} train samples; removed {removed_train_samples}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hotspot prediction workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Normalize data1/data2 source spreadsheets")
    prepare.add_argument("--data1", default=str(DEFAULT_CONFIG.data1_file))
    prepare.add_argument("--data2", default=str(DEFAULT_CONFIG.data2_file))
    prepare.add_argument("--output-dir", default=str(DEFAULT_CONFIG.processed_dir))
    prepare.add_argument(
        "--overlap-policy",
        choices=["none", "pdb-chain", "uniprot", "uniprot-or-pdb-chain"],
        default="none",
        help=(
            "How to remove data1 train samples overlapping data2. Default keeps data1 usable "
            "and reports overlap; use uniprot-or-pdb-chain for strict independent evaluation."
        ),
    )
    prepare.set_defaults(func=prepare_data)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
