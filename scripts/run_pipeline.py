"""Phase 1 CLI entry point for the registration pipeline.

Usage:
    python scripts/run_pipeline.py --config configs/experiment_C1.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.pipeline import run_experiment  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="lunar image registration pipeline")
    ap.add_argument("--config", required=True,
                    help="experiment YAML config under configs/")
    args = ap.parse_args()
    row = run_experiment(args.config)
    return row


if __name__ == "__main__":
    main()