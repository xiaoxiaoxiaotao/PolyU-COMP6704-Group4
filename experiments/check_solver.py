"""Check one solver against the harness contract on synthetic data.

Examples:
    python experiments/check_solver.py --method smo
    python experiments/check_solver.py --method admm --n 3000 --ratio 20

This does not run the full experiment matrix and never touches validation/test
labels for tuning. It writes a small diagnostic JSON under results/raw/.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
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

from src import data as data_module  # noqa: E402
from src.evaluate import (  # noqa: E402
    METHOD_REGISTRY,
    MethodUnavailableError,
    load_method,
    make_per_sample_costs,
)
from src.solver_api import check_solver_contract  # noqa: E402
from src.utils import save_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(METHOD_REGISTRY))
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--fraud-rate", type=float, default=0.10)
    parser.add_argument("--ratio", type=float, default=10.0)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--budget", type=float, default=60.0)
    parser.add_argument(
        "--json-out",
        default=None,
        help="Diagnostic JSON path (default: results/raw/solver_checks/{method}.json)",
    )
    args = parser.parse_args()

    try:
        method_class = load_method(args.method)
    except MethodUnavailableError as exc:
        print(f"[FAIL] {exc}")
        print("Implement the module/class listed in docs/INTERFACES.md first.")
        raise SystemExit(1)

    X, y = data_module.make_synthetic_fraud(
        n=args.n,
        d=args.d,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
        separation=2.5,
    )
    splits = data_module.make_stratified_splits(y, seed=args.seed)
    X_train, y_train = X[splits["train"]], y[splits["train"]]
    X_check = X[splits["val"]]
    C_per_sample = make_per_sample_costs(
        y_train,
        C=args.C,
        cost_ratio=args.ratio,
    )

    report = check_solver_contract(
        method_class,
        X_train,
        y_train,
        C_per_sample,
        X_check=X_check,
        fit_kwargs={
            "seed": args.seed,
            "tol": args.tol,
            "time_budget_sec": args.budget,
        },
    )
    report.update(
        {
            "method": args.method,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "check_config": {
                "n": args.n,
                "d": args.d,
                "fraud_rate": args.fraud_rate,
                "ratio": args.ratio,
                "C": args.C,
                "seed": args.seed,
                "tol": args.tol,
                "time_budget_sec": args.budget,
            },
        }
    )

    output = (
        Path(args.json_out)
        if args.json_out
        else Path("results") / "raw" / "solver_checks" / f"{args.method}.json"
    )
    save_json(report, output)

    print(f"Method: {args.method}")
    print(f"Passed: {report['passed']}")
    for error in report["errors"]:
        print(f"  ERROR: {error}")
    for warning in report["warnings"]:
        print(f"  WARN:  {warning}")
    details = report.get("details", {})
    print(
        "Details: "
        f"w_norm={details.get('w_norm')}, "
        f"b={details.get('b')}, "
        f"n_iter={details.get('n_iter')}, "
        f"n_history={details.get('n_objective_history')}, "
        f"has_timestamps={details.get('n_time_history') is not None}, "
        f"objective_recomputed={details.get('objective_recomputed')}"
    )
    print(f"Report: {output}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
