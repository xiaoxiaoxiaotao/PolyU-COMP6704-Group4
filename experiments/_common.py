"""Shared CLI bootstrap for experiment scripts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import run_experiment  # noqa: E402
from src.utils import deep_merge, load_yaml  # noqa: E402


def resolve_config_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    repo_candidate = REPO_ROOT / candidate
    if repo_candidate.exists():
        return repo_candidate
    raise FileNotFoundError(f"Config not found: {path}")


def build_config(path: str | Path) -> dict:
    base = load_yaml(REPO_ROOT / "configs" / "base.yaml")
    override = load_yaml(resolve_config_path(path))
    return deep_merge(base, override)


def main(description: str | None, default_config: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=default_config,
        help=f"Experiment YAML (default: {default_config})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config and planned jobs without fitting.",
    )
    args = parser.parse_args()

    config = build_config(args.config)
    print(f"Config: {args.config}")
    print(f"Experiment: {config.get('experiment')}")
    results = run_experiment(config, dry_run=args.dry_run)
    if args.dry_run:
        return
    if not results:
        print(
            "No runs executed. Solver modules are missing or no jobs matched; "
            "see the log for details."
        )
        raise SystemExit(2)
    print(f"Completed {len(results)} run(s).")
