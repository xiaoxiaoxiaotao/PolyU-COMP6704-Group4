"""Weighted Pegasos: contract, numerical behaviour and harness integration."""

import numpy as np
import pytest
from sklearn.svm import SVC

from src.solver_api import check_solver_contract, compute_primal_objective
from src.svm_pegasos import Pegasos


@pytest.fixture
def problem():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(160, 4))
    X[:, 0] += 2.5
    y = np.where(X @ [0.9, -0.7, 0.4, 0.2] - 2.0 + rng.normal(size=160) * 0.6 > 0, 1, -1)
    C = np.where(y > 0, 2.0, 0.4) * rng.uniform(0.8, 1.2, len(y))
    return X, y.astype(np.int8), C


def test_matches_libsvm_within_pegasos_accuracy(problem):
    X, y, C = problem
    reference = SVC(kernel="linear", C=1.0, tol=1e-10).fit(X, y, sample_weight=C)
    ref_objective = (
        0.5 * reference.coef_[0] @ reference.coef_[0]
        + C @ np.maximum(0.0, 1.0 - y * (X @ reference.coef_[0] + reference.intercept_[0]))
    )
    model = Pegasos().fit(X, y, C, epochs=400, time_budget_sec=60)
    assert model.objective == pytest.approx(compute_primal_objective(model, X, y, C), rel=1e-12)
    assert model.objective < ref_objective * 1.25
    assert np.mean(model.predict(X) == reference.predict(X)) >= 0.95
    assert np.linalg.norm(model.w) <= np.sqrt(2.0 * np.sum(C)) + 1e-9


def test_history_is_aligned_and_ends_at_the_returned_model(problem):
    X, y, C = problem
    model = Pegasos().fit(X, y, C, epochs=50, time_budget_sec=60)
    assert len(model.objective_history) == len(model.time_history) >= 2
    assert np.all(np.diff(model.time_history) >= 0)
    assert model.objective_history[-1] == pytest.approx(model.objective, rel=0.0, abs=0.0)
    assert model.n_iter > 0


def test_seeded_runs_are_reproducible_and_differ_across_seeds(problem):
    X, y, C = problem
    first = Pegasos().fit(X, y, C, seed=3, epochs=20)
    repeat = Pegasos().fit(X, y, C, seed=3, epochs=20)
    np.testing.assert_array_equal(first.w, repeat.w)
    assert first.b == repeat.b
    other = Pegasos().fit(X, y, C, seed=4, epochs=20)
    assert not np.allclose(first.w, other.w)


def test_batch_updates_reach_the_same_objective_scale(problem):
    X, y, C = problem
    single = Pegasos().fit(X, y, C, epochs=400, batch_size=1, seed=1)
    mini = Pegasos().fit(X, y, C, epochs=400, batch_size=32, seed=1)
    assert single.objective == pytest.approx(mini.objective, rel=0.1)


def test_time_budget_is_respected_and_reports_partial_work(problem):
    X, y, C = problem
    model = Pegasos().fit(X, y, C, epochs=100000, time_budget_sec=0.2)
    assert model.fit_time_sec <= 0.2
    assert model.stop_reason == "time_budget"
    assert not model.converged
    assert model.objective_history[-1] == model.objective
    assert np.all(np.isfinite(model.decision_function(X)))


def test_zero_budget_returns_the_initial_feasible_model(problem):
    X, y, C = problem
    model = Pegasos().fit(X, y, C, epochs=10, time_budget_sec=1e-9)
    assert model.n_iter == 0
    assert model.objective == pytest.approx(float(np.sum(C)), rel=1e-12)
    assert model.stop_reason in ("time_budget", "epochs")


def test_convergence_is_never_claimed(problem):
    X, y, C = problem
    model = Pegasos().fit(X, y, C, epochs=2000, tol=1e-9, time_budget_sec=60)
    assert model.stop_reason == "epochs"
    assert model.epochs_completed == 2000
    assert model.converged is False
    assert model.optimality_certificate.startswith("none")


@pytest.mark.parametrize("kwargs", [
    {"tol": 0}, {"tol": np.nan}, {"time_budget_sec": 0}, {"epochs": 0},
    {"batch_size": -1}, {"log_every": 0},
])
def test_invalid_options_fail_explicitly(problem, kwargs):
    X, y, C = problem
    options = {"epochs": 1}
    options.update(kwargs)
    with pytest.raises(ValueError):
        Pegasos().fit(X, y, C, **options)


def test_prediction_input_validation(problem):
    X, y, C = problem
    model = Pegasos().fit(X, y, C, epochs=5)
    with pytest.raises(ValueError, match="feature width"):
        model.decision_function(np.ones((3, 2)))
    with pytest.raises(ValueError, match="non-finite"):
        model.decision_function(np.full((3, 4), np.inf))


def test_solver_contract_and_harness(tmp_path, problem):
    from src.evaluate import run_one

    X, y, C = problem
    report = check_solver_contract(Pegasos, X, y, C, fit_kwargs={"time_budget_sec": 30})
    assert report["passed"], report
    assert not report["warnings"], report

    record = run_one(
        experiment="pegasos_test", dataset="synthetic", method="pegasos", seed=0,
        cost_ratio=10, X_train=X[:100], y_train=y[:100], X_val=X[100:130], y_val=y[100:130],
        X_test=X[130:], y_test=y[130:], C_grid=[1.0], tolerance=1e-3,
        time_budget_sec=60, method_kwargs={"epochs": 300},
        results_dir=tmp_path,
    )
    assert record["status"] == "ok", record.get("error")
    assert record["history_consistent"]
    assert record["history_points"] >= 2
