from __future__ import annotations

import numpy as np
import pytest

from src.solver_api import (
    BaseBinarySVM,
    check_solver_contract,
    compute_primal_objective,
    extract_history,
    validate_training_data,
)


class ToyStatefulSVM(BaseBinarySVM):
    def fit(self, X, y, C_per_sample, **kwargs):
        self.w = np.zeros(X.shape[1], dtype=np.float64)
        self.w[0] = 1.0
        self.b = 0.0
        margins = y.astype(np.float64) * (X @ self.w + self.b)
        hinge = np.maximum(0.0, 1.0 - margins)
        self.objective = 0.5 * float(self.w @ self.w) + float(
            np.sum(C_per_sample * hinge)
        )
        self.objective_history = [self.objective * 3.0, self.objective]
        self.time_history = [0.0, 0.1]
        self.n_iter = 2
        self.converged = True
        self.alphas = np.abs(np.asarray(C_per_sample, dtype=np.float64)).copy()
        self.rho_history = [1.0, 0.5]
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=np.float64) @ self.w + self.b


class NoHistorySVM(BaseBinarySVM):
    def fit(self, X, y, C_per_sample, **kwargs):
        self.w = np.ones(X.shape[1], dtype=np.float64)
        self.b = 0.0
        self.n_iter = 1
        self.converged = False
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=np.float64) @ self.w


class ScaledObjectiveSVM(ToyStatefulSVM):
    def fit(self, X, y, C_per_sample, **kwargs):
        super().fit(X, y, C_per_sample, **kwargs)
        self.objective = self.objective * 2.0
        return self


class ScaledHistorySVM(ToyStatefulSVM):
    def fit(self, X, y, C_per_sample, **kwargs):
        super().fit(X, y, C_per_sample, **kwargs)
        self.objective_history = [value * 2.0 for value in self.objective_history]
        return self


def _data(n: int = 200, d: int = 4):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d))
    y = np.where(np.arange(n) % 5 == 0, 1, -1).astype(np.int8)
    C = np.abs(rng.random(n)) + 0.5
    return X, y, C


def test_base_predict_returns_pm_one() -> None:
    X, y, C = _data()
    model = ToyStatefulSVM().fit(X, y, C)
    predictions = model.predict(X[:20])
    assert predictions.dtype == np.int8
    assert set(np.unique(predictions)).issubset({-1, 1})


def test_validate_training_data_rejects_bad_inputs() -> None:
    X, y, C = _data()
    with pytest.raises(ValueError, match="only -1 and \\+1"):
        validate_training_data(X, np.zeros_like(y), C)
    with pytest.raises(ValueError, match="strictly positive"):
        validate_training_data(X, y, np.zeros_like(C))
    with pytest.raises(ValueError, match="y has"):
        validate_training_data(X[:100], y, C)
    X_valid, y_valid, C_valid = validate_training_data(X, y, C)
    assert X_valid.shape == X.shape
    assert y_valid.dtype == np.int8
    assert C_valid.shape == C.shape


def test_canonical_objective_formula() -> None:
    class FixedModel:
        w = np.array([1.0])
        b = 0.0

    X = np.array([[1.0], [1.0]])
    y = np.array([1, -1], dtype=np.int8)
    C = np.array([2.0, 3.0])
    # margins = [1, -1]; hinge = [0, 2]; 0.5*1 + 3*2 = 6.5
    assert compute_primal_objective(FixedModel(), X, y, C) == pytest.approx(6.5)


def test_extract_history_accepts_list_of_dicts() -> None:
    class Model:
        history = [
            {"objective": 3.0, "time": 0.0},
            {"objective": 2.0, "time": 0.2},
        ]

    objectives, times = extract_history(Model())
    np.testing.assert_allclose(objectives, [3.0, 2.0])
    np.testing.assert_allclose(times, [0.0, 0.2])


def test_contract_checker_passes_for_compliant_solver() -> None:
    X, y, C = _data()
    report = check_solver_contract(
        ToyStatefulSVM,
        X,
        y,
        C,
        X_check=X[:25],
        fit_kwargs={"seed": 0, "tol": 1e-4, "time_budget_sec": 10},
    )
    assert report["passed"] is True
    assert report["errors"] == []
    assert report["details"]["n_iter"] == 2
    assert report["details"]["n_objective_history"] == 2
    assert report["details"]["alphas_nonzero"] > 0
    assert report["details"]["rho_final"] == pytest.approx(0.5)


def test_contract_checker_rejects_missing_history() -> None:
    X, y, C = _data()
    report = check_solver_contract(NoHistorySVM, X, y, C, fit_kwargs={"seed": 0})
    assert report["passed"] is False
    assert any("objective_history" in error for error in report["errors"])


def test_contract_checker_rejects_inconsistent_objective_scale() -> None:
    X, y, C = _data()
    report = check_solver_contract(ScaledObjectiveSVM, X, y, C, fit_kwargs={"seed": 0})
    assert report["passed"] is False
    assert any("inconsistent" in error for error in report["errors"])


def test_contract_checker_rejects_inconsistent_history_scale() -> None:
    X, y, C = _data()
    report = check_solver_contract(ScaledHistorySVM, X, y, C, fit_kwargs={"seed": 0})
    assert report["passed"] is False
    assert any("objective_history" in error for error in report["errors"])
