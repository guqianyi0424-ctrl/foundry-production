#!/usr/bin/env python3
"""Evaluate trained checkpoints on the independent data2 test set."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hotspot model on data2")
    parser.add_argument("--threshold", default="0.5", help="Numeric threshold for independent test evaluation")
    parser.add_argument("--force-reload", action="store_true", help="Regenerate data2 feature cache")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root / "evaluate_test.py"), "--threshold", str(args.threshold)]
    if args.force_reload:
        cmd.append("--force-reload")
    subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
