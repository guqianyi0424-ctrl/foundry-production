#!/usr/bin/env python3
"""Train the hotspot prediction model using data1-derived labels."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_command(root: Path, model: str, n_folds: int, force_reload: bool) -> list[str]:
    cmd = [
        sys.executable,
        str(root / "main.py"),
        "--mode",
        "train",
        "--model",
        model,
        "--n-folds",
        str(n_folds),
    ]
    if force_reload:
        cmd.append("--reprocess")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hotspot GAT model")
    parser.add_argument("--force-reload", action="store_true", help="Regenerate feature cache before training")
    parser.add_argument("--model", default="gat", choices=["gat", "gat_v2", "ensemble", "mlp"])
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cmd = build_command(root, model=args.model, n_folds=args.n_folds, force_reload=args.force_reload)
    subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
