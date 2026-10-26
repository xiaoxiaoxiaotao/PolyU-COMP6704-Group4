"""WP3 numerical checks, independent references, and harness integration."""

import numpy as np
import pytest
from sklearn.svm import SVC

from src.solver_api import check_solver_contract, compute_primal_objective
from src.svm_admm import ADMMSVM, PAADMMSVM


@pytest.fixture
def problem():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(120, 4))
    X[:, 0] += 3.0  # Requires a nontrivial, unregularized intercept.
    y = np.where(X @ [0.8, -1.0, 0.3, 0.5] - 2.8 + rng.normal(size=120) * 0.7 > 0, 1, -1)
    C = np.where(y > 0, 1.7, 0.3) * rng.uniform(0.7, 1.3, len(y))
    return X, y.astype(np.int8), C


@pytest.mark.parametrize("solver", [ADMMSVM, PAADMMSVM])
def test_matches_libsvm_and_certifies_gap(solver, problem):
    X, y, C = problem
    reference = SVC(kernel="linear", C=1.0, tol=1e-10).fit(X, y, sample_weight=C)
    ref_w, ref_b = reference.coef_[0], reference.intercept_[0]
    ref_objective = 0.5 * ref_w @ ref_w + C @ np.maximum(0, 1 - y * (X @ ref_w + ref_b))
    model = solver().fit(X, y, C, tol=1e-5, time_budget_sec=20)
    assert model.converged, model.stop_reason
    assert model.objective == pytest.approx(ref_objective, rel=2e-5)
    assert model.dual_objective <= ref_objective + 1e-7
    assert model.relative_gap <= 1e-5
    np.testing.assert_allclose(model.decision_function(X), reference.decision_function(X), atol=0.01)
    assert np.mean(model.predict(X) == reference.predict(X)) >= 0.99
    assert model.objective == pytest.approx(compute_primal_objective(model, X, y, C), rel=1e-12)
    assert model.objective_history[-1] == model.objective
    assert len(model.objective_history) == len(model.time_history) == len(model.rho_history)
    assert np.all(np.diff(model.time_history) >= 0)
    assert model.residual_history[-1]["primal"] <= model.residual_history[-1]["eps_primal"]
    assert model.residual_history[-1]["dual"] <= model.residual_history[-1]["eps_dual"]
    if solver is ADMMSVM:
        assert model.factorization_count == 1
        assert set(model.rho_history) == {1.0}
    else:
        assert model.warm_start_epochs_completed == 10
        assert model.warm_start_updates == 10 * len(y)
        assert model.time_history[-1] >= model.warm_start_time_sec


@pytest.mark.parametrize("rho0", [0.001, 100.0])
def test_adaptive_rho_refactors_without_changing_optimum(problem, rho0):
    X, y, C = problem
    model = PAADMMSVM().fit(X, y, C, K=0, rho0=rho0, tol=1e-5)
    reference = ADMMSVM().fit(X, y, C, tol=1e-5)
    assert model.converged
    assert model.factorization_count > 1
    assert len(set(model.rho_history)) > 1
    assert model.objective == pytest.approx(reference.objective, rel=2e-5)


def test_k_zero_equals_cold_admm(problem):
    X, y, C = problem
    cold = ADMMSVM().fit(X, y, C)
    hybrid = PAADMMSVM().fit(X, y, C, K=0, adaptive_rho=False, alpha=1.0)
    np.testing.assert_array_equal(cold.w, hybrid.w)
    assert cold.b == hybrid.b
    assert cold.objective_history == hybrid.objective_history
    assert hybrid.warm_start_updates == 0


@pytest.mark.parametrize("solver", [ADMMSVM, PAADMMSVM])
def test_intercept_is_not_regularized_on_rank_deficient_data(solver):
    X = np.zeros((8, 3))
    y = np.array([1, 1, -1, -1, -1, -1, -1, -1], dtype=np.int8)
    # Small penalties distinguish the required free intercept (b=1) from
    # an incorrectly regularized intercept, whose optimum would be b=0.07.
    C = np.where(y > 0, 0.05, 0.005)
    model = solver().fit(X, y, C, tol=1e-6)
    assert model.converged
    np.testing.assert_array_equal(model.w, np.zeros(3))
    assert model.b == pytest.approx(1.0, abs=1e-5)
    assert model.objective == pytest.approx(0.06, rel=1e-6)


@pytest.mark.parametrize("sign", [-1, 1])
def test_single_class_has_zero_objective(sign):
    model = ADMMSVM().fit(np.zeros((6, 2)), np.full(6, sign), np.ones(6), tol=1e-6)
    assert model.converged
    assert model.objective == pytest.approx(0.0, abs=1e-10)
    assert sign * model.b >= 1.0 - 1e-10


def test_seeded_warm_start_is_reproducible_and_changes_initialization(problem):
    X, y, C = problem
    first = PAADMMSVM().fit(X, y, C, seed=7, max_iter=0, K=20)
    repeat = PAADMMSVM().fit(X, y, C, seed=7, max_iter=0, K=20)
    np.testing.assert_array_equal(first.w, repeat.w)
    assert first.objective_history == repeat.objective_history
    assert first.objective < np.sum(C)
    assert first.warm_start_epochs_completed == 20
    assert not first.converged


@pytest.mark.parametrize("solver", [ADMMSVM, PAADMMSVM])
def test_budget_exit_returns_a_finite_best_iterate(problem, solver):
    X, y, C = problem
    model = solver().fit(X, y, C, time_budget_sec=1e-12)
    assert model.stop_reason == "time_budget"
    assert not model.converged
    assert model.n_iter == 0
    assert model.objective == pytest.approx(np.sum(C), rel=1e-14)
    assert np.all(np.isfinite(model.decision_function(X)))
    assert model.objective_history[-1] == model.objective


def test_iteration_limit_and_refit_reset_state(problem):
    X, y, C = problem
    model = PAADMMSVM().fit(X, y, C)
    model.fit(X, y, C, K=0, max_iter=1, tol=1e-12)
    assert model.n_iter == 1
    assert not model.converged
    assert model.stop_reason == "max_iter"
    assert model.warm_start_epochs_completed == 0
    assert model.objective == min(model.objective_history)
    assert model.objective_history[-1] == model.objective


@pytest.mark.parametrize("kwargs", [
    {"rho0": 0}, {"rho0": np.inf}, {"alpha": 2}, {"alpha": np.nan},
    {"tol": 0}, {"time_budget_sec": -1}, {"K": -1}, {"K": 1.5},
    {"max_iter": -1}, {"warm_start_batch_size": 0}, {"adaptive_rho": "false"},
])
def test_invalid_options_fail_explicitly(problem, kwargs):
    with pytest.raises(ValueError):
        PAADMMSVM().fit(*problem, **kwargs)


def test_prediction_input_validation(problem):
    with pytest.raises(RuntimeError, match="fit"):
        ADMMSVM().decision_function(np.zeros((2, 4)))
    model = ADMMSVM().fit(*problem)
    with pytest.raises(ValueError, match="feature width"):
        model.decision_function(np.ones((3, 2)))
    with pytest.raises(ValueError, match="non-finite"):
        model.decision_function(np.full((3, 4), np.nan))


@pytest.mark.parametrize("batch_size", [8, 64])
def test_minibatch_warm_start_keeps_the_same_weighted_objective(problem, batch_size):
    X, y, C = problem
    model = PAADMMSVM().fit(X, y, C, K=5, warm_start_batch_size=batch_size)
    reference = ADMMSVM().fit(X, y, C)
    assert model.converged
    assert model.objective == pytest.approx(reference.objective, rel=2e-4)
    assert model.warm_start_updates == 5 * int(np.ceil(len(y) / batch_size))


def test_strict_harness_budget_accepts_a_timely_nonconverged_fit(tmp_path, problem):
    from src.evaluate import run_one

    X, y, C = problem
    record = run_one(
        experiment="wp3_budget_test", dataset="synthetic", method="admm",
        seed=0, cost_ratio=10, X_train=X[:80], y_train=y[:80],
        X_val=X[80:100], y_val=y[80:100], X_test=X[100:], y_test=y[100:],
        C_grid=[1.0], tolerance=1e-15, time_budget_sec=0.1,
        method_kwargs={"max_iter": 1000000}, results_dir=tmp_path,
    )
    assert record["status"] == "ok", record.get("error")
    assert record["fit_time_sec"] <= 0.1
    assert not record["converged"]
    assert record["history_consistent"]


@pytest.mark.parametrize("solver", [ADMMSVM, PAADMMSVM])
def test_solver_contract_and_real_harness(tmp_path, problem, solver):
    from src.evaluate import run_one

    X, y, C = problem
    report = check_solver_contract(solver, X, y, C, fit_kwargs={"time_budget_sec": 20})
    assert report["passed"], report
    assert not report["warnings"], report
    method = "admm" if solver is ADMMSVM else "pa_admm"
    record = run_one(
        experiment="wp3_test", dataset="synthetic", method=method,
        seed=0, cost_ratio=10,
        X_train=X[:80], y_train=y[:80], X_val=X[80:100], y_val=y[80:100],
        X_test=X[100:], y_test=y[100:], C_grid=[0.1],
        tolerance=1e-3, time_budget_sec=20, results_dir=tmp_path,
    )
    assert record["status"] == "ok", record.get("error")
    assert record["history_consistent"]
    assert record["rho_history"]
    assert record["history_points"] >= 2
