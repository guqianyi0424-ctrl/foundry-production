#!/usr/bin/env python3
"""Train the hotspot prediction model using data1-derived labels."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hotspot GAT model")
    parser.add_argument("--force-reload", action="store_true", help="Regenerate feature cache before training")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root / "main.py"), "--mode", "train"]
    if args.force_reload:
        cmd.append("--reprocess")
    subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
