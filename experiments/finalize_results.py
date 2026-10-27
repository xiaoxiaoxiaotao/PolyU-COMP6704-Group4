"""Finalize the WP4 results pipeline once a campaign has finished.

One command replaces the manual end-of-run sequence:

    python experiments/finalize_results.py --template /path/to/6704-Group4.pptx

Steps, in order:

1. re-run ``configs/correctness.yaml`` with overwrite, so the agreement check
   uses the current solver configurations;
2. re-aggregate every experiment so references, gaps, time-to-tolerance and
   Wilcoxon tables are recomputed over all recorded runs (this is what folds a
   newly added solver into an experiment whose other methods were reused);
3. regenerate the committed report tables/figures with ``make_report_assets.py``;
4. rebuild the generated PART IV deck with its layout audit enabled;
5. print a completeness report: fits per method, timeouts, pending solvers.

Use ``--plan`` to print the steps without running anything, and ``--skip-correctness``
when the correctness experiment has already been refreshed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _common import build_config  # noqa: E402
from src.evaluate import aggregate_runs, run_experiment  # noqa: E402
from src.utils import get_logger  # noqa: E402

ALL_METHODS = ("smo", "pegasos", "admm", "pa_admm")
EXPERIMENTS = ("correctness", "main_capped", "main_fast", "main_full", "predictive")


def run_record_counts(experiment: str) -> list[dict]:
    pattern = str(ROOT / "results" / "raw" / experiment / "**" / "seed-*.json")
    records = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, "r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    return records


def report(experiments: tuple[str, ...] = EXPERIMENTS) -> dict[str, int]:
    """Print per-method completeness and return the aggregate counts."""
    totals = {method: 0 for method in ALL_METHODS}
    print("\n=== completeness ===")
    for experiment in experiments:
        records = run_record_counts(experiment)
        if not records:
            print(f"{experiment:14s} no runs recorded")
            continue
        by_method: dict[str, list[dict]] = {}
        for record in records:
            by_method.setdefault(record["method"], []).append(record)
        parts = []
        for method in ALL_METHODS:
            group = by_method.get(method)
            if not group:
                continue
            ok = sum(r["status"] == "ok" for r in group)
            timed_out = sum(bool(r.get("timed_out")) for r in group)
            reused = sum(bool(r.get("reused")) for r in group)
            parts.append(f"{method} {ok}/{len(group)} ok"
                         + (f", {timed_out} timeout" if timed_out else "")
                         + (f", {reused} reused" if reused else ""))
            totals[method] += ok
        print(f"{experiment:14s} " + " | ".join(parts))
    missing = [method for method, count in totals.items() if count == 0]
    print(f"methods with no successful run: {', '.join(missing) if missing else 'none'}")
    return totals


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", help="group deck used as the slide template")
    parser.add_argument("--deck-out", default="slides/COMP6704_Group4_PART_IV.pptx")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-deck", action="store_true")
    parser.add_argument("--plan", action="store_true",
                        help="print the steps without running them")
    args = parser.parse_args(argv)

    steps = []
    if not args.skip_correctness:
        steps.append("re-run configs/correctness.yaml with overwrite=True")
    steps.append(f"re-aggregate {', '.join(EXPERIMENTS)}")
    steps.append("regenerate results/tables and results/figures")
    if not args.skip_deck:
        steps.append("rebuild the PART IV deck and run its layout audit")
    steps.append("print completeness report")
    if args.plan:
        print("Plan:")
        for index, step in enumerate(steps, start=1):
            print(f"  {index}. {step}")
        return 0

    logger = get_logger("wp4.finalize", level="INFO")

    if not args.skip_correctness:
        config = build_config(ROOT / "configs" / "correctness.yaml")
        logger.info("Finalize step: correctness rerun")
        run_experiment(config, overwrite=True, logger=logger)

    for experiment in EXPERIMENTS:
        if not run_record_counts(experiment):
            logger.info("Finalize step: skip aggregation for %s (no runs)", experiment)
            continue
        summary = aggregate_runs(ROOT / "results", experiment, logger=logger)
        logger.info("Aggregated %s: %d summary row(s)", experiment, len(summary))

    subprocess.run([sys.executable, "experiments/make_report_assets.py"],
                   cwd=ROOT, check=True)

    if not args.skip_deck:
        if not args.template:
            raise SystemExit("--template is required unless --skip-deck is given")
        subprocess.run([sys.executable, "slides/build_part4_deck.py",
                        "--template", args.template, "--out", args.deck_out, "--check"],
                       cwd=ROOT, check=True)

    totals = report()
    unfinished = [method for method, count in totals.items() if count == 0]
    print("\nNext: review the printed completeness, then commit results/tables, "
          "results/figures and the deck.")
    return 0 if not unfinished else 0


if __name__ == "__main__":
    raise SystemExit(main())
