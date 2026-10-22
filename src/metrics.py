"""Cost-sensitive and ranking metrics for extreme-imbalance evaluation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
    roc_curve,
)


def _as_labels_and_scores(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true).ravel()
    values = np.asarray(scores, dtype=np.float64).ravel()
    if labels.size == 0:
        raise ValueError("y_true is empty")
    if labels.shape != values.shape:
        raise ValueError("y_true and scores must have the same length")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores contain non-finite values")
    return labels, values


def _has_both_classes(labels: np.ndarray) -> bool:
    positive = labels > 0
    return bool(positive.any() and (~positive).any())


def cost_per_example(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    c_fn: float,
    c_fp: float,
) -> float:
    """Return ``(c_FN * FN + c_FP * FP) / n`` for labels in {0,1} or {-1,+1}."""
    truth = np.asarray(y_true).ravel()
    prediction = np.asarray(y_pred).ravel()
    if truth.size == 0:
        return float("nan")
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if c_fn < 0 or c_fp < 0:
        raise ValueError("costs must be non-negative")

    positive = truth > 0
    predicted_positive = prediction > 0
    false_negative = positive & ~predicted_positive
    false_positive = ~positive & predicted_positive
    total = c_fn * float(false_negative.sum()) + c_fp * float(false_positive.sum())
    return total / float(truth.size)


def pr_auc(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> float:
    """Average precision (PR-AUC). Returns NaN when only one class is present."""
    labels, values = _as_labels_and_scores(y_true, scores)
    if not _has_both_classes(labels):
        return float("nan")
    return float(average_precision_score((labels > 0).astype(np.int8), values))


def recall_at_fpr(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    max_fpr: float,
) -> float:
    """Maximum recall at a false-positive rate no larger than ``max_fpr``."""
    labels, values = _as_labels_and_scores(y_true, scores)
    if not (0.0 <= max_fpr <= 1.0):
        raise ValueError("max_fpr must lie in [0, 1]")
    if not _has_both_classes(labels):
        return float("nan")
    fpr, tpr, _ = roc_curve((labels > 0).astype(np.int8), values, drop_intermediate=False)
    mask = fpr <= max_fpr
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))


def balanced_accuracy(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> float:
    """Balanced accuracy for labels in {0,1} or {-1,+1}."""
    truth = np.asarray(y_true).ravel()
    prediction = np.asarray(y_pred).ravel()
    if truth.size == 0 or truth.shape != prediction.shape:
        return float("nan")
    if not _has_both_classes(truth):
        return float("nan")
    truth_pm = np.where(truth > 0, 1, -1)
    prediction_pm = np.where(prediction > 0, 1, -1)
    return float(balanced_accuracy_score(truth_pm, prediction_pm))


def select_threshold_min_cost(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    c_fn: float,
    c_fp: float,
    return_cost: bool = False,
) -> float | tuple[float, float]:
    """Choose the score threshold minimizing validation cost.

    Predictions are positive when ``score >= threshold``. Ties are handled by
    grouping equal scores; on equal cost the higher threshold (fewer alerts) is
    preferred. If requested, also returns the minimum cost per example.
    """
    labels, values = _as_labels_and_scores(y_true, scores)
    if c_fn < 0 or c_fp < 0:
        raise ValueError("costs must be non-negative")

    positive = labels > 0
    n_positive = int(positive.sum())
    n_negative = int(positive.size - n_positive)

    order = np.argsort(-values, kind="mergesort")
    sorted_values = values[order]
    sorted_positive = positive[order]
    tp = np.cumsum(sorted_positive)
    fp = np.cumsum(~sorted_positive)

    # k=0 means "predict everything negative"; highest threshold.
    best_cost = c_fn * n_positive
    best_threshold = float(np.nextafter(np.max(values), np.inf))

    # Candidate cutoffs only at boundaries between distinct score values.
    group_ends = np.flatnonzero(
        np.r_[sorted_values[1:] != sorted_values[:-1], True]
    ) + 1
    for k in group_ends:
        cost = c_fn * (n_positive - tp[k - 1]) + c_fp * fp[k - 1]
        if cost < best_cost:
            best_cost = float(cost)
            best_threshold = float(sorted_values[k - 1])

    if not np.isfinite(best_threshold):
        best_threshold = float(np.max(values))
    if not np.isfinite(best_threshold):
        best_threshold = 0.0

    if return_cost:
        return best_threshold, float(best_cost) / float(labels.size)
    return best_threshold


def evaluate_scores(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    c_fn: float,
    c_fp: float,
    *,
    threshold: float | None = None,
) -> dict[str, float]:
    """Evaluate scores; select a cost-minimizing threshold if none is supplied.

    In the experiment harness the threshold is always fitted on validation
    scores and passed here with the test scores, so test labels never tune it.
    """
    labels, values = _as_labels_and_scores(y_true, scores)
    if threshold is None:
        threshold = float(
            select_threshold_min_cost(labels, values, c_fn=c_fn, c_fp=c_fp)
        )
    predictions = np.where(values >= float(threshold), 1, -1).astype(np.int8)
    return {
        "threshold": float(threshold),
        "cost_per_example": float(
            cost_per_example(labels, predictions, c_fn=c_fn, c_fp=c_fp)
        ),
        "pr_auc": float(pr_auc(labels, values)),
        "recall_at_fpr_0.001": float(recall_at_fpr(labels, values, 0.001)),
        "recall_at_fpr_0.01": float(recall_at_fpr(labels, values, 0.01)),
        "balanced_accuracy": float(balanced_accuracy(labels, predictions)),
    }


def precision_recall_points(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    max_points: int = 200,
) -> dict[str, list[float]] | None:
    """Return a small precision/recall curve for report figures, or None."""
    labels, values = _as_labels_and_scores(y_true, scores)
    if not _has_both_classes(labels):
        return None
    precision, recall, _ = precision_recall_curve(
        (labels > 0).astype(np.int8),
        values,
    )
    if max_points > 1 and precision.size > max_points:
        indices = np.unique(
            np.round(np.linspace(0, precision.size - 1, max_points)).astype(np.int64)
        )
        precision = precision[indices]
        recall = recall[indices]
    return {
        "precision": [float(value) for value in precision],
        "recall": [float(value) for value in recall],
    }


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    sample = np.asarray(values, dtype=np.float64).ravel()
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return float("nan"), float("nan")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must lie in (0, 1)")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")

    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, sample.size, size=(int(n_boot), sample.size))
    means = sample[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.percentile(means, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return float(lower), float(upper)


def wilcoxon_test(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Paired two-sided Wilcoxon signed-rank test, tolerant of degenerate input."""
    first = np.asarray(a, dtype=np.float64).ravel()
    second = np.asarray(b, dtype=np.float64).ravel()
    if first.shape != second.shape:
        raise ValueError("Paired samples must have the same length")

    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid]
    second = second[valid]
    n_pairs = int(first.size)
    if n_pairs == 0:
        return {"statistic": float("nan"), "p_value": float("nan"), "n_pairs": 0}

    differences = first - second
    if np.allclose(differences, 0.0):
        return {"statistic": 0.0, "p_value": 1.0, "n_pairs": n_pairs}

    result = stats.wilcoxon(first, second, zero_method="wilcox", alternative="two-sided")
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_pairs": n_pairs,
    }
