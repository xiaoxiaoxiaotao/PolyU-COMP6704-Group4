from __future__ import annotations

import json
import logging

import numpy as np
import pytest
from sklearn.linear_model import SGDClassifier

from src import data as data_module
from src.evaluate import (
    REQUIRED_RUN_FIELDS,
    MethodUnavailableError,
    aggregate_runs,
    load_method,
    run_experiment,
    run_file_path,
    run_one,
    time_to_tolerance,
)
from src.utils import get_logger


class SklearnSGDHingeSmoke:
    """Temporary BinarySVM-compatible stand-in; never used for final results."""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        C_per_sample: np.ndarray,
        seed: int = 0,
        tol: float = 1e-3,
        time_budget_sec: float | None = None,
        max_iter: int = 100,
        **kwargs: object,
    ) -> "SklearnSGDHingeSmoke":
        n_samples = int(X.shape[0])
        c_base = float(np.mean(C_per_sample))
        self.model_ = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=1.0 / max(n_samples * c_base, 1e-12),
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


class ToyHistorySVM:
    """Toy deterministic solver exposing a consistent objective history."""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        C_per_sample: np.ndarray,
        **kwargs: object,
    ) -> "ToyHistorySVM":
        self.w = np.zeros(X.shape[1], dtype=np.float64)
        self.w[0] = 1.0
        self.b = 0.0
        margins = y.astype(np.float64) * (X @ self.w + self.b)
        hinge = np.maximum(0.0, 1.0 - margins)
        self.objective = 0.5 * float(self.w @ self.w) + float(
            np.sum(C_per_sample * hinge)
        )
        self.objective_history = [
            self.objective * 4.0,
            self.objective * 2.0,
            self.objective,
        ]
        self.time_history = [0.0, 0.1, 0.2]
        self.n_iter = 3
        self.converged = True
        self.alphas = np.abs(np.asarray(C_per_sample, dtype=np.float64)).copy()
        self.rho_history = [1.0, 0.5]
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0.0, 1, -1)


class FailingSVM:
    def fit(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("intentional fit failure")

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones(X.shape[0])


def _synthetic_arrays(n: int = 600, d: int = 6, fraud_rate: float = 0.08):
    X, y = data_module.make_synthetic_fraud(
        n=n,
        d=d,
        fraud_rate=fraud_rate,
        seed=0,
        separation=2.5,
    )
    splits = data_module.make_stratified_splits(y, seed=0)
    return (
        X[splits["train"]],
        y[splits["train"]],
        X[splits["val"]],
        y[splits["val"]],
        X[splits["test"]],
        y[splits["test"]],
    )


def test_run_one_writes_valid_smoke_json(tmp_path) -> None:
    arrays = _synthetic_arrays()
    result = run_one(
        experiment="harness_smoke",
        dataset="synthetic",
        method="sklearn_sgd_smoke",
        seed=0,
        cost_ratio=10,
        X_train=arrays[0],
        y_train=arrays[1],
        X_val=arrays[2],
        y_val=arrays[3],
        X_test=arrays[4],
        y_test=arrays[5],
        C_grid=[0.1, 1.0],
        tolerance=1e-3,
        time_budget_sec=60,
        method_registry_extra={"sklearn_sgd_smoke": SklearnSGDHingeSmoke},
        results_dir=tmp_path,
        data_meta={"purpose": "unit smoke test"},
    )

    assert result["status"] == "ok"
    for field in REQUIRED_RUN_FIELDS:
        assert field in result
    assert 0.0 <= result["pr_auc"] <= 1.0
    assert 0.0 <= result["balanced_accuracy"] <= 1.0
    assert 0.0 <= result["cost_per_example"]
    assert result["threshold"] is not None
    assert result["objective"] is not None

    path = run_file_path(
        tmp_path,
        experiment="harness_smoke",
        dataset="synthetic",
        method="sklearn_sgd_smoke",
        cost_ratio=10,
        seed=0,
    )
    assert path.exists()
    text = path.read_text()
    assert "NaN" not in text
    assert "Infinity" not in text
    payload = json.loads(
        text,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant: {value}")
        ),
    )
    for field in REQUIRED_RUN_FIELDS:
        assert field in payload

    summary = aggregate_runs(tmp_path, "harness_smoke", logger=None)
    assert not summary.empty
    assert (tmp_path / "tables" / "harness_smoke_runs.csv").exists()
    assert (tmp_path / "tables" / "harness_smoke_summary.csv").exists()


def test_run_experiment_assigns_reference_and_time_to_tolerance(tmp_path) -> None:
    config = {
        "experiment": "toy_test",
        "dataset": "synthetic",
        "methods": ["toy"],
        "seeds": [0],
        "cost_ratios": [10],
        "C_grid": [1.0],
        "tolerance": 1e-6,
        "time_budget_sec": 30,
        "data": {
            "source": "synthetic",
            "synthetic_n_list": [400],
            "synthetic_d": 5,
            "synthetic_fraud_rate": 0.10,
            "synthetic_separation": 2.5,
            "standardize_amount": False,
        },
        "method_caps": {},
        "method_kwargs": {},
        "data_variants": [],
        "method_variants": [],
        "paths": {"results_dir": str(tmp_path)},
        "logging": {"level": "WARNING"},
        "aggregate": True,
        "history_max_points": 100,
        "overwrite": True,
        "reference_method": None,
    }
    logger = get_logger("test_harness", level=logging.WARNING)
    results = run_experiment(
        config,
        method_registry_extra={"toy": ToyHistorySVM},
        logger=logger,
    )

    assert len(results) == 1
    record = results[0]
    assert record["status"] == "ok"
    assert record["reference_objective"] == pytest.approx(record["objective"])
    assert record["relative_gap"] == pytest.approx(0.0)
    assert record["reached_tolerance"] is True
    assert record["time_to_tolerance_sec"] == pytest.approx(0.2)
    assert record["history_consistent"] is True
    assert record["alphas_summary"]["count"] == record["n_train"]
    assert record["rho_final"] == pytest.approx(0.5)
    assert record["rho_history"] == pytest.approx([1.0, 0.5])

    path = run_file_path(
        tmp_path,
        experiment="toy_test",
        dataset="synthetic-400",
        method="toy",
        cost_ratio=10,
        seed=0,
    )
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["reference_objective"] == pytest.approx(payload["objective"])
    assert (tmp_path / "tables" / "toy_test_summary.csv").exists()


def test_time_to_tolerance_function() -> None:
    time_sec, reached, final_gap = time_to_tolerance(
        [10.0, 5.0, 1.05, 1.0],
        [0.0, 1.0, 2.0, 3.0],
        reference_objective=1.0,
        tol=0.1,
    )
    assert time_sec == pytest.approx(2.0)
    assert reached is True
    assert final_gap == pytest.approx(0.0)

    time_sec, reached, final_gap = time_to_tolerance(
        [10.0, 5.0, 2.0],
        [0.0, 1.0, 2.0],
        reference_objective=1.0,
        tol=0.1,
    )
    assert time_sec is None
    assert reached is False
    assert final_gap == pytest.approx(1.0)

    time_sec, reached, _ = time_to_tolerance(
        [1.0], None, reference_objective=1.0, tol=0.1
    )
    assert time_sec is None
    assert reached is True


def test_missing_method_is_reported_not_faked() -> None:
    with pytest.raises(MethodUnavailableError):
        load_method("not_a_registered_method")
    with pytest.raises(MethodUnavailableError):
        load_method("smo_does_not_exist")


def test_failed_fit_still_writes_a_failure_json(tmp_path) -> None:
    arrays = _synthetic_arrays(n=300)
    result = run_one(
        experiment="failure_test",
        dataset="synthetic",
        method="failing",
        seed=0,
        cost_ratio=10,
        X_train=arrays[0],
        y_train=arrays[1],
        X_val=arrays[2],
        y_val=arrays[3],
        X_test=arrays[4],
        y_test=arrays[5],
        C_grid=[1.0],
        method_registry_extra={"failing": FailingSVM},
        results_dir=tmp_path,
    )
    assert result["status"] == "failed"
    assert "intentional fit failure" in result["error"]
    path = run_file_path(
        tmp_path,
        experiment="failure_test",
        dataset="synthetic",
        method="failing",
        cost_ratio=10,
        seed=0,
    )
    assert path.exists()
    payload = json.loads(path.read_text())
    for field in REQUIRED_RUN_FIELDS:
        assert field in payload
