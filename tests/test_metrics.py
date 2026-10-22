from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src import metrics


def test_cost_per_example_hand_computed() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, -1, 0, 1])
    # one false negative (cost 10) and one false positive (cost 1) over 4 rows.
    assert metrics.cost_per_example(y_true, y_pred, c_fn=10, c_fp=1) == pytest.approx(2.75)
    assert metrics.cost_per_example(y_true, y_pred, c_fn=1, c_fp=0) == pytest.approx(0.25)


def test_pr_auc_matches_sklearn() -> None:
    y_true = np.array([1, 0, 1, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3])
    expected = average_precision_score((y_true > 0).astype(int), scores)
    assert metrics.pr_auc(y_true, scores) == pytest.approx(expected)


def test_recall_at_fpr() -> None:
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.4, 0.8, 0.1])
    assert metrics.recall_at_fpr(y_true, scores, 0.0) == pytest.approx(0.5)
    assert metrics.recall_at_fpr(y_true, scores, 0.5) == pytest.approx(1.0)


def test_threshold_min_cost_picks_clean_cut() -> None:
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    threshold, cost = metrics.select_threshold_min_cost(
        y_true, scores, c_fn=10, c_fp=1, return_cost=True
    )
    assert threshold == pytest.approx(0.8)
    assert cost == pytest.approx(0.0)
    predictions = np.where(scores >= threshold, 1, -1)
    np.testing.assert_array_equal(predictions, [1, 1, -1, -1])


def test_threshold_handles_tied_scores() -> None:
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.8, 0.1])
    threshold, cost = metrics.select_threshold_min_cost(
        y_true, scores, c_fn=5, c_fp=1, return_cost=True
    )
    assert threshold == pytest.approx(0.8)
    assert cost == pytest.approx(0.25)


def test_evaluate_scores_with_validation_threshold() -> None:
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    result = metrics.evaluate_scores(y_true, scores, c_fn=10, c_fp=1, threshold=0.8)
    assert result["threshold"] == pytest.approx(0.8)
    assert result["cost_per_example"] == pytest.approx(0.0)
    assert result["pr_auc"] == pytest.approx(1.0)
    assert result["recall_at_fpr_0.001"] == pytest.approx(1.0)
    assert result["recall_at_fpr_0.01"] == pytest.approx(1.0)
    assert result["balanced_accuracy"] == pytest.approx(1.0)

    selected = metrics.evaluate_scores(y_true, scores, c_fn=10, c_fp=1)
    assert selected["threshold"] == pytest.approx(0.8)


def test_single_class_metrics_are_nan() -> None:
    y_true = np.zeros(5, dtype=int)
    scores = np.linspace(0.1, 0.9, 5)
    assert math.isnan(metrics.pr_auc(y_true, scores))
    assert math.isnan(metrics.recall_at_fpr(y_true, scores, 0.01))
    assert math.isnan(metrics.balanced_accuracy(y_true, np.ones(5, dtype=int)))


def test_bootstrap_ci_is_deterministic_and_contains_mean() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    first = metrics.bootstrap_ci(values, n_boot=500, seed=7)
    second = metrics.bootstrap_ci(values, n_boot=500, seed=7)
    assert first == second
    assert first[0] <= values.mean() <= first[1]
    constant = metrics.bootstrap_ci(np.full(4, 2.5))
    assert constant[0] == pytest.approx(2.5)
    assert constant[1] == pytest.approx(2.5)


def test_wilcoxon_degenerate_and_different_samples() -> None:
    identical = metrics.wilcoxon_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert identical["p_value"] == pytest.approx(1.0)
    assert identical["statistic"] == pytest.approx(0.0)

    different = metrics.wilcoxon_test(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
    )
    assert 0.0 <= different["p_value"] <= 1.0
    assert different["n_pairs"] == 6
