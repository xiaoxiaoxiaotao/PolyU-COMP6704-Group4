"""Independent correctness checks for the WP2 cost-sensitive SMO solver.

All datasets are small CPU fixtures. LIBSVM is a reference only: the project
implementation must still solve the dual itself. In particular, non-unique dual
solutions are compared through feasibility, objective values and decision scores,
not by requiring identical support-vector coefficients.
"""

import time
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.svm import SVC

from src.svm_smo import SMO


FIT_OPTIONS = dict(tol=1e-7, max_iter=50_000, time_budget_sec=20.0, log_every=7)


def _kernel(X, Y, kernel, gamma):
    if kernel == "linear":
        return X @ Y.T
    distances = np.maximum(
        np.sum(X * X, axis=1)[:, None]
        + np.sum(Y * Y, axis=1)[None, :]
        - 2.0 * X @ Y.T,
        0.0,
    )
    return np.exp(-gamma * distances)


def _primal(model, X, y, C, kernel="linear", gamma=0.7):
    """Recompute the stated objective independently of reported diagnostics."""
    if kernel == "linear":
        norm_squared = float(np.dot(model.w, model.w))
    else:
        signed_alpha = np.asarray(model.alphas) * y
        norm_squared = float(signed_alpha @ _kernel(X, X, kernel, gamma) @ signed_alpha)
    hinge = np.maximum(0.0, 1.0 - y * model.decision_function(X))
    return 0.5 * norm_squared + float(C @ hinge)


def _reference_primal(reference, X, y, C, kernel="linear", gamma=0.7):
    if kernel == "linear":
        norm_squared = float(np.sum(reference.coef_ ** 2))
    else:
        coefficients = reference.dual_coef_[0]
        support = reference.support_vectors_
        norm_squared = float(coefficients @ _kernel(support, support, kernel, gamma) @ coefficients)
    hinge = np.maximum(0.0, 1.0 - y * reference.decision_function(X))
    return 0.5 * norm_squared + float(C @ hinge)


def _assert_feasible(model, y, C):
    alpha = np.asarray(model.alphas)
    assert alpha.shape == y.shape
    assert np.all(np.isfinite(alpha))
    assert np.min(alpha) >= -1e-10
    assert np.max(alpha - C) <= 1e-10 * max(1.0, float(np.max(C)))
    assert abs(float(y @ alpha)) <= 1e-9 * max(1.0, float(np.sum(C)))


def _kkt_violation(model, X, y, C):
    alpha = np.asarray(model.alphas)
    margin = y * model.decision_function(X)
    lower = alpha <= 1e-8
    upper = alpha >= C - 1e-8
    violation = np.abs(margin - 1.0)
    violation[lower] = np.maximum(0.0, 1.0 - margin[lower])
    violation[upper] = np.maximum(0.0, margin[upper] - 1.0)
    return float(np.max(violation))


def _dataset(kind):
    rng = np.random.default_rng(410)
    if kind == "separable":
        y = np.tile(np.array([-1, 1]), 30)
        X = rng.normal(scale=0.3, size=(len(y), 5))
        X[:, 0] += 1.4 * y
    elif kind == "overlapping":
        X = rng.normal(size=(64, 4))
        y = np.where(X[:, 0] - 0.6 * X[:, 1] + rng.normal(size=64) > 0, 1, -1)
    elif kind == "xor":
        X = rng.uniform(-1.5, 1.5, size=(64, 2))
        y = np.where(X[:, 0] * X[:, 1] > 0, 1, -1)
    else:
        raise ValueError(kind)
    # Variation within each class also checks genuinely per-sample upper bounds.
    C = np.where(y > 0, 1.2, 0.35) * rng.uniform(0.7, 1.3, size=len(y))
    return X, y, C


@pytest.mark.parametrize("kind,kernel", [("separable", "linear"), ("overlapping", "linear"), ("xor", "rbf")])
def test_matches_libsvm_on_same_weighted_problem(kind, kernel):
    X, y, C = _dataset(kind)
    gamma = 0.7
    model = SMO().fit(X, y, C, kernel=kernel, gamma=gamma, seed=5, **FIT_OPTIONS)
    reference = SVC(C=1.0, kernel=kernel, gamma=gamma, tol=1e-10, shrinking=False)
    reference.fit(X, y, sample_weight=C)

    _assert_feasible(model, y, C)
    assert model.converged, model.stop_reason
    assert _kkt_violation(model, X, y, C) <= 2e-5
    objective = _primal(model, X, y, C, kernel, gamma)
    reference_objective = _reference_primal(reference, X, y, C, kernel, gamma)
    assert objective == pytest.approx(reference_objective, rel=2e-5, abs=2e-6)
    assert model.objective == pytest.approx(objective, rel=2e-10, abs=2e-9)
    signed_alpha = np.asarray(model.alphas) * y
    dual = float(np.sum(model.alphas) - 0.5 * signed_alpha @ _kernel(X, X, kernel, gamma) @ signed_alpha)
    assert objective >= dual - 1e-8 * max(1.0, abs(objective))
    assert (objective - dual) / max(1.0, abs(objective)) <= 2e-5

    query = np.random.default_rng(11).normal(size=(35, X.shape[1]))
    scores = model.decision_function(query)
    reference_scores = reference.decision_function(query)
    assert scores.shape == (len(query),)
    assert np.all(np.isfinite(scores))
    np.testing.assert_allclose(scores, reference_scores, atol=3e-3, rtol=3e-3)
    # A tiny shift at a decision boundary is not evidence of solver failure.
    away_from_boundary = np.abs(reference_scores) > 1e-2
    np.testing.assert_array_equal(
        model.predict(query)[away_from_boundary], reference.predict(query)[away_from_boundary]
    )
    if kernel == "rbf":
        assert model.w is None  # No fictitious finite-dimensional linear weight.


def test_unequal_bounds_have_an_analytic_solution_and_no_cost_renormalization():
    X = np.array([[0.0], [2.0]])
    y = np.array([-1, 1])
    C = np.array([0.1, 0.6])
    originals = (X.copy(), y.copy(), C.copy())
    model = SMO().fit(X, y, C, **FIT_OPTIONS)

    # Equality imposes alpha_1 = alpha_2. The unconstrained optimum is 0.5,
    # so the smaller bound gives both alpha=0.1. The positive point is free.
    np.testing.assert_allclose(model.alphas, [0.1, 0.1], atol=1e-9)
    np.testing.assert_allclose(model.w, [0.2], atol=1e-9)
    assert model.b == pytest.approx(0.6, abs=1e-8)
    assert model.objective == pytest.approx(0.18, abs=1e-8)
    assert model.converged
    for original, after in zip(originals, (X, y, C)):
        np.testing.assert_array_equal(after, original)


@pytest.mark.parametrize("kernel", ["linear", "rbf"])
def test_duplicate_opposite_labels_and_zero_curvature(kernel):
    X = np.repeat(np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]), 2, axis=0)
    y = np.array([-1, 1, -1, 1, -1, 1])
    C = np.array([0.2, 0.7, 0.3, 0.8, 0.4, 0.9])
    model = SMO().fit(X, y, C, kernel=kernel, gamma=0.7, **FIT_OPTIONS)
    reference = SVC(C=1.0, kernel=kernel, gamma=0.7, tol=1e-10, shrinking=False)
    reference.fit(X, y, sample_weight=C)
    _assert_feasible(model, y, C)
    assert model.converged
    assert _kkt_violation(model, X, y, C) <= 2e-5
    assert _primal(model, X, y, C, kernel) == pytest.approx(
        _reference_primal(reference, X, y, C, kernel), rel=2e-5, abs=2e-6
    )


def test_bias_is_valid_when_every_support_vector_is_at_its_bound():
    X = np.zeros((2, 3))
    y = np.array([-1, 1])
    C = np.full(2, 0.3)
    model = SMO().fit(X, y, C, **FIT_OPTIONS)
    np.testing.assert_allclose(model.alphas, C, atol=1e-10)
    assert -1.0 - 1e-9 <= model.b <= 1.0 + 1e-9
    assert model.objective == pytest.approx(0.6, abs=1e-9)
    assert model.converged
    _assert_feasible(model, y, C)
    assert _kkt_violation(model, X, y, C) <= 1e-8


def test_histories_are_aligned_and_use_the_canonical_objective():
    X, y, C = _dataset("overlapping")
    started = time.perf_counter()
    model = SMO().fit(X, y, C, **FIT_OPTIONS)
    elapsed = time.perf_counter() - started
    objectives = np.asarray(model.objective_history)
    timestamps = np.asarray(model.time_history)
    assert objectives.ndim == timestamps.ndim == 1
    assert len(objectives) == len(timestamps) >= 2
    assert np.all(np.isfinite(objectives))
    assert np.all(np.isfinite(timestamps))
    assert np.all(timestamps >= 0.0)
    assert np.all(np.diff(timestamps) >= 0.0)
    assert timestamps[-1] <= elapsed + 0.01
    assert objectives[-1] == pytest.approx(_primal(model, X, y, C), rel=1e-9, abs=1e-8)
    assert model.stop_reason == "converged"


@pytest.mark.parametrize("budget", ["max_iter", "time_budget_sec"])
def test_zero_budget_returns_a_feasible_initial_model_and_does_not_claim_convergence(budget):
    X, y, C = _dataset("overlapping")
    options = {**FIT_OPTIONS, budget: 0}
    model = SMO().fit(X, y, C, **options)
    _assert_feasible(model, y, C)
    assert not model.converged
    assert model.n_iter == 0
    assert model.stop_reason == ("time_budget" if budget == "time_budget_sec" else "max_iter")
    assert np.all(np.isfinite(model.decision_function(X)))
    assert model.objective == pytest.approx(_primal(model, X, y, C), rel=1e-10, abs=1e-9)
    assert len(model.objective_history) == len(model.time_history) >= 2


def test_iteration_limit_is_not_reported_as_convergence():
    X, y, C = _dataset("overlapping")
    model = SMO().fit(X, y, C, **{**FIT_OPTIONS, "max_iter": 1})
    assert model.n_iter <= 1
    assert not model.converged
    assert model.stop_reason == "max_iter"
    _assert_feasible(model, y, C)


def test_rbf_certifies_the_solution_reached_by_the_last_allowed_pair():
    X = np.array([[-1.0], [1.0]])
    y = np.array([-1, 1])
    C = np.ones(2)
    model = SMO().fit(X, y, C, kernel="rbf", gamma=1.0,
                      max_iter=1, time_budget_sec=2.0, tol=1e-7)

    # Both dual variables reach C in the single exact pair solve.
    assert model.n_iter == 1
    np.testing.assert_allclose(model.alphas, C, atol=1e-12)
    assert model.converged and model.stop_reason == "converged"
    assert not model.returned_best_checkpoint
    assert model.kkt_violation == pytest.approx(0.0, abs=1e-12)
    assert model.relative_duality_gap == pytest.approx(0.0, abs=1e-12)
    assert model.objective == pytest.approx(1.0 + np.exp(-4.0), abs=1e-12)
    assert model.objective_history[-1] == pytest.approx(_primal(model, X, y, C, "rbf", 1.0))
    _assert_feasible(model, y, C)


@pytest.mark.parametrize("budget", [1.0, 2.0])
def test_rbf_terminal_certificate_cannot_claim_convergence_after_deadline(budget, monkeypatch):
    from src import svm_smo

    X = np.array([[-1.0], [1.0]])
    y = np.array([-1, 1])
    C = np.ones(2)
    model = SMO()
    clock = SimpleNamespace(now=0.0)
    original_get = svm_smo._KernelColumns.get

    def timed_kernel_column(cache, index):
        column = original_get(cache, index)
        # Charge one second for each fresh-certificate kernel access, after
        # the single pair update. The two budgets expire during the refresh
        # and exactly when the refresh finishes, respectively.
        if model.n_iter == 1:
            clock.now += 1.0
        return column

    monkeypatch.setattr(svm_smo, "time", SimpleNamespace(perf_counter=lambda: clock.now))
    monkeypatch.setattr(svm_smo._KernelColumns, "get", timed_kernel_column)
    model.fit(X, y, C, kernel="rbf", gamma=1.0, max_iter=1,
              time_budget_sec=budget, tol=1e-7)

    assert model.n_iter == 1
    assert model.kkt_violation == pytest.approx(0.0, abs=1e-12)
    assert model.relative_duality_gap == pytest.approx(0.0, abs=1e-12)
    assert not model.converged
    assert model.stop_reason == "time_budget"
    assert model.returned_best_checkpoint
    assert model.fit_time_sec >= budget
    assert model.objective_history[-1] == pytest.approx(_primal(model, X, y, C, "rbf", 1.0))
    _assert_feasible(model, y, C)


@pytest.mark.parametrize("gap_tol", [None, 1e-7])
def test_high_costs_require_primal_dual_certificate_after_kkt_tolerance(gap_tol):
    rng = np.random.default_rng(19)
    y = np.tile(np.array([-1, 1]), 20)
    X = rng.normal(scale=0.35, size=(40, 3))
    X[:, 0] += 1.1 * y
    C = np.full(len(y), 100.0)
    tol = 1e-4
    target = tol if gap_tol is None else gap_tol
    model = SMO().fit(
        X, y, C, tol=tol, gap_tol=gap_tol,
        max_iter=10_000, time_budget_sec=2.0, log_every=20,
    )

    # This fixture reaches the KKT tolerance while weighted slack still makes
    # its relative primal-dual gap too large. It must then perform more updates.
    kkt_history = np.asarray(model.kkt_violation_history)
    gap_history = np.asarray(model.relative_duality_gap_history)
    needs_refinement = np.flatnonzero((kkt_history <= tol) & (gap_history > target))
    assert len(needs_refinement) > 0
    assert model.n_iter > model.history_iters[needs_refinement[0]]
    assert model.converged, model.stop_reason
    assert model.dual_feasible
    assert model.kkt_violation <= tol
    assert model.gap_tol == target
    assert not model.returned_best_checkpoint

    primal = _primal(model, X, y, C)
    signed_alpha = model.alphas * y
    dual = float(np.sum(model.alphas) - 0.5 * signed_alpha @ (X @ X.T) @ signed_alpha)
    relative_gap = (primal - dual) / max(1.0, abs(primal), abs(dual))
    assert relative_gap <= target + 1e-10
    assert model.relative_duality_gap == pytest.approx(relative_gap, abs=1e-10)


@pytest.mark.parametrize("kernel", ["linear", "rbf"])
@pytest.mark.parametrize("budget_kind", ["iterations", "time"])
def test_budget_return_restores_the_best_measured_primal_with_consistent_state(
    kernel, budget_kind, monkeypatch,
):
    X, y, C = _dataset("overlapping" if kernel == "linear" else "xor")
    model = SMO()
    options = dict(FIT_OPTIONS, log_every=1, kernel=kernel, gamma=0.7)
    if budget_kind == "iterations":
        options["max_iter"] = 12
    else:
        # A deterministic monotonic clock expires after 12 accepted updates.
        # Patch only this module's clock reference, not Python's global clock.
        monkeypatch.setattr(
            "src.svm_smo.time",
            SimpleNamespace(perf_counter=lambda: float(getattr(model, "n_iter", 0))),
        )
        options["time_budget_sec"] = 12.0
    model.fit(X, y, C, **options)

    assert not model.converged
    assert model.stop_reason == ("max_iter" if budget_kind == "iterations" else "time_budget")
    assert model.n_iter == 12  # Total work is retained even if an older state wins.
    assert model.returned_best_checkpoint
    assert model.returned_checkpoint_iteration <= model.n_iter
    if kernel == "linear":
        assert model.returned_checkpoint_iteration < model.n_iter
    assert 0 <= model.returned_checkpoint_index < len(model.objective_history) - 1
    assert model.objective == pytest.approx(min(model.objective_history), rel=1e-10, abs=1e-10)
    assert model.objective_history[-1] == pytest.approx(model.objective)
    assert model.time_history[-1] >= model.time_history[model.returned_checkpoint_index]
    assert np.all(np.diff(model.time_history) >= 0.0)
    _assert_feasible(model, y, C)

    signed_alpha = model.alphas * y
    expected_scores = _kernel(X, X, kernel, 0.7) @ signed_alpha + model.b
    np.testing.assert_allclose(model.decision_function(X), expected_scores, atol=1e-9)
    if kernel == "linear":
        np.testing.assert_allclose(model.w, X.T @ signed_alpha, atol=1e-10)
    assert model.objective == pytest.approx(_primal(model, X, y, C, kernel), rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("dimension", [1, 7, 29])
def test_feature_dimension_is_not_hard_coded_and_predictions_follow_score_sign(dimension):
    rng = np.random.default_rng(9)
    X = rng.normal(size=(26, dimension))
    y = np.where(X[:, 0] > 0.0, 1, -1)
    model = SMO().fit(X, y, np.full(len(y), 0.4), **FIT_OPTIONS)
    assert model.w.shape == (dimension,)
    scores = model.decision_function(X)
    np.testing.assert_array_equal(model.predict(X), np.where(scores >= 0.0, 1, -1))
    assert model.decision_function(np.empty((0, dimension))).shape == (0,)


@pytest.mark.parametrize("cache_size_mb", [0.0, 0.0001])
def test_tiny_or_disabled_kernel_cache_preserves_solution(cache_size_mb):
    X, y, C = _dataset("xor")
    limited = SMO().fit(
        X, y, C, kernel="rbf", gamma=0.7, cache_size_mb=cache_size_mb, **FIT_OPTIONS
    )
    normal = SMO().fit(X, y, C, kernel="rbf", gamma=0.7, cache_size_mb=1.0, **FIT_OPTIONS)
    assert limited.converged and normal.converged
    _assert_feasible(limited, y, C)
    assert limited.objective == pytest.approx(normal.objective, rel=2e-7, abs=2e-7)
    np.testing.assert_allclose(limited.decision_function(X), normal.decision_function(X), atol=1e-5)


def test_same_seed_is_reproducible():
    X, y, C = _dataset("overlapping")
    first = SMO().fit(X, y, C, seed=14, **FIT_OPTIONS)
    second = SMO().fit(X, y, C, seed=14, **FIT_OPTIONS)
    np.testing.assert_allclose(first.alphas, second.alphas, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(first.decision_function(X), second.decision_function(X), atol=1e-12)
    assert first.n_iter == second.n_iter
    # Wall-clock timestamps are intentionally not required to be identical.


@pytest.mark.parametrize(
    "X,y,C",
    [
        (np.zeros((3, 2)), [-1, 1], [1.0, 1.0]),
        (np.zeros((2, 2)), [-1, 1], [1.0]),
        (np.zeros((2, 2)), [0, 1], [1.0, 1.0]),
        (np.zeros((2, 2)), [1, 1], [1.0, 1.0]),
        (np.array([[np.nan, 0.0], [0.0, 1.0]]), [-1, 1], [1.0, 1.0]),
        (np.array([[np.inf, 0.0], [0.0, 1.0]]), [-1, 1], [1.0, 1.0]),
        (np.zeros((2, 2)), [-1, 1], [0.0, 1.0]),
        (np.zeros((2, 2)), [-1, 1], [-0.1, 1.0]),
        (np.zeros((2, 2)), [-1, 1], [np.inf, 1.0]),
        (np.empty((0, 2)), [], []),
        (np.empty((2, 0)), [-1, 1], [1.0, 1.0]),
        (np.zeros(2), [-1, 1], [1.0, 1.0]),
    ],
)
def test_invalid_training_data_fails_clearly(X, y, C):
    with pytest.raises(ValueError):
        SMO().fit(X, y, C)


@pytest.mark.parametrize(
    "options",
    [
        {"kernel": "not-a-kernel"},
        {"kernel": "rbf", "gamma": -0.1},
        {"tol": 0.0},
        {"tol": np.nan},
        {"gap_tol": 0.0},
        {"gap_tol": np.nan},
        {"cache_size_mb": -1.0},
        {"time_budget_sec": -1.0},
        {"max_iter": -1},
        {"unsupported_config_key": True},
    ],
)
def test_invalid_or_unknown_hyperparameters_are_not_silently_ignored(options):
    with pytest.raises((ValueError, TypeError)):
        SMO().fit(np.array([[0.0], [2.0]]), np.array([-1, 1]), np.ones(2), **options)


def test_prediction_before_fit_and_wrong_feature_count_fail_clearly():
    model = SMO()
    with pytest.raises(RuntimeError):
        model.decision_function(np.zeros((2, 1)))
    model.fit(np.array([[0.0], [2.0]]), np.array([-1, 1]), np.ones(2), **FIT_OPTIONS)
    with pytest.raises(ValueError):
        model.decision_function(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        model.decision_function(np.array([[np.nan]]))
