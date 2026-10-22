"""Harness smoke test using sklearn SGDClassifier(loss="hinge").

This is NOT a project result and NOT a substitute for any WP1-WP3 solver. It
only verifies that split -> fit -> validation threshold -> test metrics -> JSON
works end to end while the registered solver modules are still under development.
"""

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

import numpy as np  # noqa: E402
from sklearn.linear_model import SGDClassifier  # noqa: E402

from src import data as data_module  # noqa: E402
from src.evaluate import run_file_path, run_one  # noqa: E402


class SklearnSGDHingeSmoke:
    """Minimal BinarySVM-compatible wrapper used only by the harness smoke test."""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        C_per_sample: np.ndarray,
        seed: int = 0,
        tol: float = 1e-3,
        time_budget_sec: float | None = None,
        max_iter: int = 200,
        **kwargs: object,
    ) -> "SklearnSGDHingeSmoke":
        n_samples = int(X.shape[0])
        c_base = float(np.mean(C_per_sample))
        alpha = 1.0 / max(n_samples * c_base, 1e-12)
        self.model_ = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=alpha,
            max_iter=int(max_iter),
            tol=float(tol),
            random_state=int(seed),
            fit_intercept=True,
        )
        self.model_.fit(X, y, sample_weight=np.asarray(C_per_sample, dtype=np.float64))
        self.w = self.model_.coef_.ravel().copy()
        self.b = float(self.model_.intercept_[0])
        self.n_iter = int(getattr(self.model_, "n_iter_", 0))
        self.converged = bool(self.n_iter < int(max_iter))
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.model_.decision_function(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    X, y = data_module.make_synthetic_fraud(
        n=2000,
        d=8,
        fraud_rate=0.05,
        seed=0,
        separation=2.5,
    )
    splits = data_module.make_stratified_splits(y, seed=0)
    result = run_one(
        experiment="harness_smoke",
        dataset="synthetic",
        method="sklearn_sgd_smoke",
        seed=0,
        cost_ratio=10,
        X_train=X[splits["train"]],
        y_train=y[splits["train"]],
        X_val=X[splits["val"]],
        y_val=y[splits["val"]],
        X_test=X[splits["test"]],
        y_test=y[splits["test"]],
        C_grid=[0.1, 1.0],
        tolerance=1e-3,
        time_budget_sec=60.0,
        method_registry_extra={"sklearn_sgd_smoke": SklearnSGDHingeSmoke},
        results_dir=args.results_dir,
        data_meta={
            "purpose": "harness smoke test",
            "warning": (
                "sklearn SGDClassifier(loss='hinge') is a temporary stand-in. "
                "This JSON is not a final project result."
            ),
        },
    )
    path = run_file_path(
        args.results_dir,
        experiment="harness_smoke",
        dataset="synthetic",
        method="sklearn_sgd_smoke",
        cost_ratio=10,
        seed=0,
    )
    print(f"Smoke status: {result['status']}")
    print(f"Smoke JSON: {path}")
    if result["status"] != "ok":
        print(f"Error: {result.get('error')}")
        raise SystemExit(1)
    print(
        "Smoke metrics (temporary baseline, not for the report): "
        f"cost={result['cost_per_example']}, pr_auc={result['pr_auc']}, "
        f"balanced_accuracy={result['balanced_accuracy']}"
    )


if __name__ == "__main__":
    main()
