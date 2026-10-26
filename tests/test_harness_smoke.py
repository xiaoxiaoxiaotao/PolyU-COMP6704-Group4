from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import SGDClassifier

from src import data as data_module
from src.evaluate import (
    REQUIRED_RUN_FIELDS,
    MethodUnavailableError,
    _assign_reference_objectives,
    aggregate_runs,
    load_method,
    pairwise_wilcoxon_tests,
    run_experiment,
    run_file_path,
    run_one,
    time_to_tolerance,
)
from src.metrics import evaluate_scores, select_threshold_min_cost
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

    scale = 1.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        C_per_sample: np.ndarray,
        **kwargs: object,
    ) -> "ToyHistorySVM":
        self.w = np.zeros(X.shape[1], dtype=np.float64)
        self.w[0] = self.scale
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


class ToyHistorySVMAlt(ToyHistorySVM):
    scale = 2.0


class CountingToySVM(ToyHistorySVM):
    fit_calls = 0

    def fit(self, *args: object, **kwargs: object) -> "CountingToySVM":
        type(self).fit_calls += 1
        return super().fit(*args, **kwargs)


class ToyRBFKernelSVM:
    """Nonlinear stand-in: no w, so the canonical linear objective is undefined."""

    def fit(self, X, y, C_per_sample, **kwargs: object) -> "ToyRBFKernelSVM":
        self.kernel = "rbf"
        self.gamma_ = 0.1
        self.b = 0.0
        self.n_iter = 3
        self.converged = True
        self.objective = 9.0
        self.objective_history = [18.0, 9.0]
        self.time_history = [0.1, 0.2]
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.linspace(-1.0, 1.0, int(np.asarray(X).shape[0]), dtype=np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0.0, 1, -1).astype(np.int8)


class FailingSVM:
    def fit(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("intentional fit failure")

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones(X.shape[0])


class SleepySVM(ToyHistorySVM):
    def fit(self, *args: object, **kwargs: object) -> "SleepySVM":
        time.sleep(2.0)
        return self


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
        "methods": ["toy", "toy_alt"],
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
        method_registry_extra={"toy": ToyHistorySVM, "toy_alt": ToyHistorySVMAlt},
        logger=logger,
    )

    assert len(results) == 2
    record = min(results, key=lambda item: item["objective"])
    assert record["status"] == "ok"
    assert record["config_hash"]
    assert record["reference_objective"] == pytest.approx(
        min(item["objective"] for item in results)
    )
    assert record["relative_gap"] == pytest.approx(0.0)
    assert record["reached_tolerance"] is True
    assert record["time_to_tolerance_sec"] == pytest.approx(0.2)
    assert record["history_consistent"] is True
    assert record["alphas_summary"]["count"] == record["n_train"]
    assert record["rho_final"] == pytest.approx(0.5)
    assert record["rho_history"] == pytest.approx([1.0, 0.5])
    assert record["evaluation_cost_ratio"] == 10

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


def test_evaluation_cost_ratio_is_used_for_threshold_and_cost(tmp_path) -> None:
    arrays = _synthetic_arrays()
    result = run_one(
        experiment="eval_cost_test",
        dataset="synthetic",
        method="toy",
        seed=0,
        cost_ratio=10,
        evaluation_cost_ratio=1,
        X_train=arrays[0],
        y_train=arrays[1],
        X_val=arrays[2],
        y_val=arrays[3],
        X_test=arrays[4],
        y_test=arrays[5],
        C_grid=[1.0],
        method_registry_extra={"toy": ToyHistorySVM},
        results_dir=tmp_path,
    )
    assert result["status"] == "ok"
    assert result["training_cost_ratio"] == 10
    assert result["evaluation_cost_ratio"] == 1
    val_scores = arrays[2] @ np.array([1.0] + [0.0] * (arrays[2].shape[1] - 1))
    expected_threshold = select_threshold_min_cost(
        arrays[3], val_scores, c_fn=1, c_fp=1
    )
    test_scores = arrays[4] @ np.array([1.0] + [0.0] * (arrays[4].shape[1] - 1))
    expected = evaluate_scores(
        arrays[5],
        test_scores,
        c_fn=1,
        c_fp=1,
        threshold=expected_threshold,
    )
    assert result["threshold"] == pytest.approx(expected_threshold)
    assert result["cost_per_example"] == pytest.approx(expected["cost_per_example"])


def test_hard_timeout_is_recorded_as_failure(tmp_path) -> None:
    arrays = _synthetic_arrays(n=300)
    result = run_one(
        experiment="timeout_test",
        dataset="synthetic",
        method="sleepy",
        seed=0,
        cost_ratio=10,
        X_train=arrays[0],
        y_train=arrays[1],
        X_val=arrays[2],
        y_val=arrays[3],
        X_test=arrays[4],
        y_test=arrays[5],
        C_grid=[1.0],
        hard_timeout_sec=0.1,
        method_registry_extra={"sleepy": SleepySVM},
        results_dir=tmp_path,
    )
    assert result["status"] == "failed"
    assert result["timed_out"] is True
    assert "hard_timeout" in result["error"]


def test_single_method_has_no_self_reference(tmp_path) -> None:
    record = {
        "experiment": "single_test",
        "dataset": "synthetic",
        "method": "toy",
        "variant": None,
        "seed": 0,
        "cost_ratio": 20,
        "evaluation_cost_ratio": 20,
        "n_train": 100,
        "tolerance": 1e-3,
        "objective": 10.0,
        "objective_history": [12.0, 10.0],
        "time_history": [0.1, 0.2],
        "history_points": 2,
        "history_consistent": True,
        "status": "ok",
    }
    _assign_reference_objectives(
        [record],
        {"reference_method": None},
        tmp_path,
        logger=None,
    )
    assert record["reference_objective"] is None
    assert record["time_to_tolerance_sec"] is None
    assert record["tolerance_note"] == "insufficient_comparable_runs"


def test_pairwise_wilcoxon_table() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["d"] * 6,
            "method": ["a"] * 3 + ["b"] * 3,
            "variant": ["base"] * 6,
            "seed": [0, 1, 2, 0, 1, 2],
            "cost_ratio": [10] * 6,
            "evaluation_cost_ratio": [10] * 6,
            "n_train": [100] * 6,
            "objective": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
        }
    )
    table = pairwise_wilcoxon_tests(frame, metrics=("objective",))
    assert len(table) == 1
    assert table.iloc[0]["method_a"] == "a"
    assert table.iloc[0]["method_b"] == "b"
    assert table.iloc[0]["n_pairs"] == 3


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


def _reference_record(*, method: str, C: float, objective: float) -> dict:
    return {
        "experiment": "C_reference_test",
        "dataset": "synthetic",
        "method": method,
        "variant": None,
        "seed": 0,
        "cost_ratio": 20,
        "evaluation_cost_ratio": 20,
        "n_train": 100,
        "C": C,
        "tolerance": 1e-3,
        "objective": objective,
        "objective_history": [objective + 1.0, objective],
        "time_history": [0.1, 0.2],
        "history_points": 2,
        "history_consistent": True,
        "status": "ok",
    }


def test_reference_requires_the_same_selected_C(tmp_path) -> None:
    records = [
        _reference_record(method="a", C=1.0, objective=10.0),
        _reference_record(method="b", C=10.0, objective=20.0),
    ]
    _assign_reference_objectives(
        records,
        {"reference_method": None},
        tmp_path,
        logger=None,
    )
    for record in records:
        assert record["reference_objective"] is None
        assert record["time_to_tolerance_sec"] is None
        assert record["tolerance_note"] == "insufficient_comparable_runs"

    same_c_records = [
        _reference_record(method="a", C=1.0, objective=10.0),
        _reference_record(method="b", C=1.0, objective=20.0),
    ]
    _assign_reference_objectives(
        same_c_records,
        {"reference_method": None},
        tmp_path,
        logger=None,
    )
    assert same_c_records[0]["reference_objective"] == pytest.approx(10.0)
    assert same_c_records[1]["reference_objective"] == pytest.approx(10.0)
    assert same_c_records[1]["relative_gap"] > 0


def test_pairwise_objective_requires_the_same_selected_C() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["d"] * 4,
            "method": ["a", "a", "b", "b"],
            "variant": ["base"] * 4,
            "seed": [0, 1, 0, 1],
            "cost_ratio": [10] * 4,
            "evaluation_cost_ratio": [10] * 4,
            "n_train": [100] * 4,
            "C": [1.0, 1.0, 10.0, 10.0],
            "objective": [1.0, 2.0, 3.0, 4.0],
            "cost_per_example": [0.1, 0.2, 0.3, 0.4],
        }
    )
    table = pairwise_wilcoxon_tests(
        frame,
        metrics=("objective", "cost_per_example"),
    )
    assert table[table["metric"] == "objective"].empty
    cost_rows = table[table["metric"] == "cost_per_example"]
    assert len(cost_rows) == 1
    assert cost_rows.iloc[0]["n_pairs"] == 2

    frame["C"] = 1.0
    table = pairwise_wilcoxon_tests(frame, metrics=("objective",))
    assert len(table) == 1
    assert table.iloc[0]["C"] == 1.0
    assert table.iloc[0]["n_pairs"] == 2


def test_aggregate_groups_objective_metrics_by_selected_C(tmp_path) -> None:
    raw_dir = tmp_path / "raw" / "C_aggregate_test" / "m"
    raw_dir.mkdir(parents=True)
    for seed, (C, objective, cost) in enumerate(
        [(1.0, 10.0, 0.1), (10.0, 20.0, 0.2)]
    ):
        record = {
            "experiment": "C_aggregate_test",
            "dataset": "synthetic",
            "method": "m",
            "variant": "base",
            "seed": seed,
            "cost_ratio": 10,
            "evaluation_cost_ratio": 10,
            "n_train": 100,
            "C": C,
            "status": "ok",
            "objective": objective,
            "cost_per_example": cost,
        }
        (raw_dir / f"seed-{seed}.json").write_text(json.dumps(record))

    summary = aggregate_runs(tmp_path, "C_aggregate_test", logger=None)
    objective_rows = summary[summary["metric"] == "objective"]
    assert len(objective_rows) == 2
    assert set(objective_rows["C"]) == {1.0, 10.0}
    cost_rows = summary[summary["metric"] == "cost_per_example"]
    assert len(cost_rows) == 1
    assert pd.isna(cost_rows.iloc[0]["C"])


def _reuse_test_config(tmp_path, method_kwargs=None) -> dict:
    return {
        "experiment": "reuse_test",
        "dataset": "synthetic",
        "methods": ["toy"],
        "seeds": [0],
        "cost_ratios": [10],
        "C_grid": [1.0],
        "tolerance": 1e-6,
        "time_budget_sec": 30,
        "reference_method": None,
        "reuse_existing": True,
        "data": {
            "source": "synthetic",
            "synthetic_n_list": [400],
            "synthetic_d": 5,
            "synthetic_fraud_rate": 0.10,
            "synthetic_separation": 2.5,
            "standardize_amount": False,
        },
        "method_kwargs": method_kwargs or {},
        "data_variants": [],
        "method_variants": [],
        "paths": {"results_dir": str(tmp_path)},
        "logging": {"level": "WARNING"},
        "aggregate": True,
        "history_max_points": 100,
        "overwrite": False,
    }


def test_run_experiment_reuses_matching_successful_runs(tmp_path) -> None:
    config = _reuse_test_config(tmp_path)
    registry = {"toy": CountingToySVM}
    logger = get_logger("test_reuse", level=logging.WARNING)
    CountingToySVM.fit_calls = 0

    first = run_experiment(config, method_registry_extra=registry, logger=logger)
    assert first[0]["status"] == "ok"
    assert CountingToySVM.fit_calls == 1

    second = run_experiment(config, method_registry_extra=registry, logger=logger)
    assert second[0]["status"] == "ok"
    assert second[0]["reused"] is True
    assert CountingToySVM.fit_calls == 1

    mismatch = json.loads(json.dumps(config))
    mismatch["method_kwargs"] = {"toy": {"unexpected": 1}}
    with pytest.raises(RuntimeError, match="cannot be reused"):
        run_experiment(
            mismatch,
            method_registry_extra=registry,
            logger=logger,
        )


def test_run_experiment_retries_existing_failed_runs(tmp_path) -> None:
    config = _reuse_test_config(tmp_path)
    logger = get_logger("test_reuse_failed", level=logging.WARNING)

    first = run_experiment(
        config,
        method_registry_extra={"toy": FailingSVM},
        logger=logger,
    )
    assert first[0]["status"] == "failed"

    CountingToySVM.fit_calls = 0
    second = run_experiment(
        config,
        method_registry_extra={"toy": CountingToySVM},
        logger=logger,
    )
    assert second[0]["status"] == "ok"
    assert not second[0].get("reused", False)
    assert CountingToySVM.fit_calls == 1


def test_nonlinear_runs_are_excluded_from_objective_comparison(tmp_path) -> None:
    config = _reuse_test_config(tmp_path)
    config["experiment"] = "family_test"
    config["methods"] = ["toy", "toy_rbf"]
    results = run_experiment(
        config,
        method_registry_extra={
            "toy": ToyHistorySVM,
            "toy_rbf": ToyRBFKernelSVM,
        },
        logger=get_logger("test_family", level=logging.WARNING),
    )
    by_method = {record["method"]: record for record in results}

    nonlinear = by_method["toy_rbf"]
    assert nonlinear["status"] == "ok"
    assert nonlinear["objective_recomputable"] is False
    assert nonlinear["kernel"] == "rbf"
    assert nonlinear["gamma"] == pytest.approx(0.1)
    assert nonlinear["reference_objective"] is None
    assert nonlinear["time_to_tolerance_sec"] is None
    assert (
        nonlinear["tolerance_note"] == "objective_not_recomputable_from_w_and_b"
    )
    assert nonlinear["pr_auc"] is not None

    linear = by_method["toy"]
    assert linear["objective_recomputable"] is True
    assert linear["reference_objective"] is None
    assert linear["tolerance_note"] == "insufficient_comparable_runs"
