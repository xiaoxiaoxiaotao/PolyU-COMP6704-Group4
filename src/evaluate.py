"""Unified evaluation harness for cost-sensitive linear SVM solvers.

Solver modules are imported lazily through :data:`METHOD_REGISTRY`; importing
this module never imports a solver. The harness owns split construction,
per-sample weights, validation-only threshold selection, run logging, and
aggregation, so every method sees the same protocol.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from . import data as data_module
from .metrics import (
    bootstrap_ci,
    evaluate_scores,
    precision_recall_points,
    select_threshold_min_cost,
)
from .solver_api import (
    BinarySVM,
    compute_primal_objective,
    extract_history,
    extract_vector,
    filtered_fit_kwargs,
    get_attr,
)
from .utils import configure_single_thread, get_logger, save_json, set_seed

METHOD_REGISTRY: dict[str, tuple[str, str]] = {
    "smo": ("src.svm_smo", "SMO"),
    "pegasos": ("src.svm_pegasos", "Pegasos"),
    "admm": ("src.svm_admm", "ADMMSVM"),
    "pa_admm": ("src.svm_admm", "PAADMMSVM"),
    "cvxpy": ("src.baseline_cvxpy", "CVXPYLinearSVM"),
}

REQUIRED_RUN_FIELDS = (
    "experiment",
    "dataset",
    "method",
    "cost_ratio",
    "seed",
    "n_train",
    "n_val",
    "n_test",
    "d",
    "C",
    "hyperparams",
    "fit_time_sec",
    "n_iter",
    "converged",
    "objective",
    "reference_objective",
    "relative_gap",
    "time_to_tolerance_sec",
    "reached_tolerance",
    "pr_auc",
    "recall_at_fpr_0.001",
    "recall_at_fpr_0.01",
    "cost_per_example",
    "balanced_accuracy",
    "peak_memory_mb",
)

METRICS_FOR_TABLES = (
    "time_to_tolerance_sec",
    "fit_time_sec",
    "n_iter",
    "objective",
    "relative_gap",
    "pr_auc",
    "recall_at_fpr_0.001",
    "recall_at_fpr_0.01",
    "cost_per_example",
    "balanced_accuracy",
    "peak_memory_mb",
)


class MethodUnavailableError(RuntimeError):
    """Raised when a registered solver module/class is not available."""


def load_method(
    name: str,
    extra_methods: Mapping[str, Any] | None = None,
) -> type[BinarySVM]:
    """Lazily import and return a solver class from the registry."""
    if extra_methods and name in extra_methods:
        return extra_methods[name]
    if name not in METHOD_REGISTRY:
        known = ", ".join(sorted(METHOD_REGISTRY))
        raise MethodUnavailableError(f"Unknown method '{name}'. Registered methods: {known}")

    module_name, class_name = METHOD_REGISTRY[name]
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise MethodUnavailableError(
            f"Method '{name}' is unavailable: could not import '{module_name}'. "
            f"Expected class '{class_name}'. Original error: {exc}"
        ) from exc

    method_class = getattr(module, class_name, None)
    if method_class is None:
        raise MethodUnavailableError(
            f"Method '{name}' is unavailable: '{module_name}' has no class "
            f"'{class_name}'."
        )
    missing = [
        method_name
        for method_name in ("fit", "decision_function", "predict")
        if not hasattr(method_class, method_name)
    ]
    if missing:
        raise MethodUnavailableError(
            f"Method '{name}' does not implement: {', '.join(missing)}. "
            "See docs/INTERFACES.md."
        )
    return method_class


def method_available(
    name: str,
    extra_methods: Mapping[str, Any] | None = None,
) -> tuple[bool, str | None]:
    """Return availability without raising, for dry-run/status reporting."""
    try:
        load_method(name, extra_methods)
    except MethodUnavailableError as exc:
        return False, str(exc)
    return True, None


def make_per_sample_costs(
    y: Sequence[int] | np.ndarray,
    C: float,
    cost_ratio: float,
    c_fp: float = 1.0,
) -> np.ndarray:
    """Build ``C_i = C * w_y`` with ``w+ / w- = cost_ratio`` and mean weight one.

    This is the harness-side construction of the contract in EXPERIMENTS.md.
    WP1's ``src/costs.py`` remains the owner of the canonical cost model; this
    helper keeps the harness runnable before that module lands.
    """
    labels = np.asarray(y).ravel()
    if labels.size == 0:
        raise ValueError("y is empty")
    ratio = float(cost_ratio)
    base_fp = float(c_fp)
    if ratio <= 0 or base_fp <= 0:
        raise ValueError("cost_ratio and c_fp must be positive")

    c_fn = ratio * base_fp
    weights = np.where(labels > 0, c_fn, base_fp).astype(np.float64)
    weights /= weights.mean()
    return float(C) * weights


def time_to_tolerance(
    objective_history: Sequence[float] | np.ndarray | None,
    times: Sequence[float] | np.ndarray | None,
    reference_objective: float | None,
    tol: float = 1e-4,
) -> tuple[float | None, bool, float | None]:
    """Return ``(time_sec, reached, final_gap)`` for a convergence history.

    ``time_sec`` is None when the tolerance is not reached or when no valid
    timing history is available. ``final_gap`` is the last finite relative gap.
    """
    if reference_objective is None or not math.isfinite(float(reference_objective)):
        return None, False, None
    if tol <= 0:
        raise ValueError("tol must be positive")
    if objective_history is None:
        return None, False, None

    objectives = np.asarray(objective_history, dtype=np.float64).ravel()
    if objectives.size == 0:
        return None, False, None

    time_values: np.ndarray | None = None
    if times is not None:
        candidate_times = np.asarray(times, dtype=np.float64).ravel()
        if candidate_times.size == objectives.size:
            time_values = candidate_times

    finite = np.isfinite(objectives)
    if not np.any(finite):
        return None, False, None

    reference = float(reference_objective)
    denominator = abs(reference) if abs(reference) > 1e-12 else 1.0
    gaps = (objectives - reference) / denominator
    finite_gaps = gaps[finite]
    final_gap = float(finite_gaps[-1])
    reached = bool(final_gap <= tol)
    if not reached:
        return None, False, final_gap

    finite_indices = np.flatnonzero(finite)
    crossing_local = np.flatnonzero(gaps[finite] <= tol)
    if crossing_local.size == 0:
        return None, True, final_gap

    crossing_index = int(finite_indices[int(crossing_local[0])])
    if time_values is None or not np.isfinite(time_values[crossing_index]):
        return None, True, final_gap
    return float(time_values[crossing_index]), True, final_gap


def _relative_gap(objective: float | None, reference: float | None) -> float | None:
    if objective is None or reference is None:
        return None
    denominator = abs(float(reference)) if abs(float(reference)) > 1e-12 else 1.0
    return float((float(objective) - float(reference)) / denominator)


def _process_peak_memory_mb() -> float | None:
    try:
        import resource

        max_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return max_rss / (1024.0 * 1024.0)
        return max_rss / 1024.0
    except Exception:  # pragma: no cover - platform dependent
        return None


def _environment_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "single_thread": True,
    }


def _downsample_history(
    history: np.ndarray | None,
    times: np.ndarray | None,
    max_points: int,
) -> tuple[list[float] | None, list[float] | None, bool]:
    if history is None:
        return None, None, False
    if max_points < 2 or history.size <= max_points:
        return history.tolist(), None if times is None else times.tolist(), False

    indices = np.unique(
        np.round(np.linspace(0, history.size - 1, max_points)).astype(np.int64)
    )
    sampled_history = history[indices].tolist()
    sampled_times = None if times is None else times[indices].tolist()
    return sampled_history, sampled_times, True


def _clean_ratio(ratio: float | int) -> float | int:
    value = float(ratio)
    return int(value) if value.is_integer() else value


def _format_ratio(ratio: float | int) -> str:
    value = float(ratio)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def _slug(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in ("-", "_", ".") else "-"
        for character in str(value).strip()
    )
    return cleaned.strip("-") or "unnamed"


def run_file_path(
    results_dir: str | Path,
    experiment: str,
    dataset: str,
    method: str,
    cost_ratio: float | int,
    seed: int,
    variant: str | None = None,
) -> Path:
    """Return ``results/raw/{experiment}/{dataset}/{method}/ratio-{r}/seed-{s}.json``.

    Ablation method variants add one directory level to avoid overwriting
    different hyperparameter settings of the same method.
    """
    base = Path(results_dir) / "raw" / _slug(experiment) / _slug(dataset) / _slug(method)
    if variant:
        base = base / f"variant-{_slug(variant)}"
    return base / f"ratio-{_format_ratio(cost_ratio)}" / f"seed-{int(seed)}.json"


def save_run(
    record: Mapping[str, Any],
    results_dir: str | Path,
    overwrite: bool = True,
) -> Path:
    """Write one run JSON and attach its path to the record copy."""
    path = run_file_path(
        results_dir=results_dir,
        experiment=record["experiment"],
        dataset=record["dataset"],
        method=record["method"],
        cost_ratio=record["cost_ratio"],
        seed=record["seed"],
        variant=record.get("variant"),
    )
    if path.exists() and not overwrite:
        raise FileExistsError(f"Run already exists (overwrite=False): {path}")
    payload = dict(record)
    payload["run_file"] = str(path)
    save_json(payload, path)
    return path


def _base_record(
    *,
    experiment: str,
    dataset: str,
    method: str,
    variant: str | None,
    seed: int,
    cost_ratio: float | int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C_grid: Sequence[float],
    method_kwargs: Mapping[str, Any],
    tolerance: float,
    time_budget_sec: float,
    data_meta: Mapping[str, Any] | None,
) -> dict[str, Any]:
    ratio = _clean_ratio(cost_ratio)
    record: dict[str, Any] = {
        "experiment": experiment,
        "dataset": dataset,
        "method": method,
        "variant": variant,
        "cost_ratio": ratio,
        "seed": int(seed),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "d": int(X_train.shape[1]),
        "C": None,
        "C_grid": [float(value) for value in C_grid],
        "hyperparams": dict(method_kwargs),
        "c_fn": float(ratio),
        "c_fp": 1.0,
        "tolerance": float(tolerance),
        "time_budget_sec": float(time_budget_sec),
        "fit_time_sec": None,
        "tuning_fit_time_sec": None,
        "n_iter": None,
        "converged": None,
        "objective": None,
        "objective_reported": None,
        "objective_recomputed": None,
        "objective_consistency_rel_error": None,
        "history_final_consistency_rel_error": None,
        "history_consistent": None,
        "reference_objective": None,
        "relative_gap": None,
        "time_to_tolerance_sec": None,
        "reached_tolerance": False,
        "tolerance_note": "reference_objective_not_available",
        "pr_auc": None,
        "recall_at_fpr_0.001": None,
        "recall_at_fpr_0.01": None,
        "cost_per_example": None,
        "balanced_accuracy": None,
        "threshold": None,
        "val_cost_per_example": None,
        "val_pr_auc": None,
        "peak_memory_mb": _process_peak_memory_mb(),
        "n_train_positives": int(np.sum(y_train > 0)),
        "n_val_positives": int(np.sum(y_val > 0)),
        "n_test_positives": int(np.sum(y_test > 0)),
        "history_available": False,
        "history_truncated": False,
        "objective_history": None,
        "time_history": None,
        "alphas_summary": None,
        "rho_history": None,
        "rho_final": None,
        "candidate_diagnostics": [],
        "candidate_failures": [],
        "data_meta": dict(data_meta or {}),
        "environment": _environment_info(),
        "status": "failed",
        "error": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return record


def _validate_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    if X_train.ndim != 2 or X_val.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X arrays must be two-dimensional")
    if X_train.shape[1] != X_val.shape[1] or X_train.shape[1] != X_test.shape[1]:
        raise ValueError("X arrays have inconsistent feature dimensions")
    for name, labels, matrix in (
        ("train", y_train, X_train),
        ("val", y_val, X_val),
        ("test", y_test, X_test),
    ):
        if labels.shape[0] != matrix.shape[0]:
            raise ValueError(f"{name} X/y length mismatch")
        if not np.all(np.isin(labels, (-1, 1))):
            raise ValueError(f"{name} labels must be in {{-1, +1}}")
        if matrix.shape[0] == 0:
            raise ValueError(f"{name} partition is empty")


def run_one(
    *,
    experiment: str,
    dataset: str,
    method: str,
    seed: int,
    cost_ratio: float | int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C_grid: Sequence[float] = (1.0,),
    tolerance: float = 1e-4,
    time_budget_sec: float = 300.0,
    method_kwargs: Mapping[str, Any] | None = None,
    method_registry_extra: Mapping[str, Any] | None = None,
    variant: str | None = None,
    results_dir: str | Path = "results",
    reference_objective: float | None = None,
    overwrite: bool = True,
    history_max_points: int = 10000,
    save: bool = True,
    raise_errors: bool = False,
    logger: Any = None,
    data_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit one method/seed/ratio run and write a run JSON.

    Thresholds and base ``C`` are chosen on validation data only. The test
    partition is touched exactly once, after the selected model is fixed.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.int8)
    y_val = np.asarray(y_val, dtype=np.int8)
    y_test = np.asarray(y_test, dtype=np.int8)
    _validate_arrays(X_train, y_train, X_val, y_val, X_test, y_test)

    grid = [float(value) for value in C_grid]
    if not grid or any(value <= 0 for value in grid):
        raise ValueError("C_grid must contain positive values")
    if tolerance <= 0 or time_budget_sec <= 0:
        raise ValueError("tolerance and time_budget_sec must be positive")

    kwargs = dict(method_kwargs or {})
    record = _base_record(
        experiment=experiment,
        dataset=dataset,
        method=method,
        variant=variant,
        seed=seed,
        cost_ratio=cost_ratio,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        C_grid=grid,
        method_kwargs=kwargs,
        tolerance=tolerance,
        time_budget_sec=time_budget_sec,
        data_meta=data_meta,
    )

    try:
        method_class = load_method(method, method_registry_extra)
    except MethodUnavailableError as exc:
        record["error"] = str(exc)
        if save:
            save_run(record, results_dir, overwrite=overwrite)
        if logger is not None:
            logger.error(str(exc))
        if raise_errors:
            raise
        return record

    record["method_module"] = getattr(method_class, "__module__", None)
    record["method_class"] = getattr(method_class, "__name__", None)
    c_fn = float(cost_ratio)
    c_fp = 1.0

    candidates: list[dict[str, Any]] = []
    candidate_failures: list[dict[str, Any]] = []
    for C in grid:
        C_per_sample = make_per_sample_costs(
            y_train,
            C=C,
            cost_ratio=cost_ratio,
            c_fp=c_fp,
        )
        fit_kwargs = {
            "seed": int(seed),
            "tol": float(tolerance),
            "time_budget_sec": float(time_budget_sec),
        }
        fit_kwargs.update(kwargs)

        try:
            model = method_class()
            filtered_kwargs = filtered_fit_kwargs(model.fit, fit_kwargs)
            set_seed(seed)
            start = datetime.now(timezone.utc)
            timer = _WallTimer()
            with timer:
                model.fit(X_train, y_train, C_per_sample, **filtered_kwargs)
            fit_time = timer.elapsed
        except Exception as exc:
            candidate_failures.append(
                {
                    "C": C,
                    "reason": "fit_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=6),
                    "fit_time_sec": None,
                }
            )
            continue

        if fit_time > float(time_budget_sec):
            candidate_failures.append(
                {
                    "C": C,
                    "reason": "time_budget_exceeded",
                    "error": (
                        f"fit_time_sec={fit_time:.3f} exceeded budget "
                        f"{float(time_budget_sec):.3f}"
                    ),
                    "fit_time_sec": float(fit_time),
                }
            )
            continue

        try:
            val_scores = np.asarray(model.decision_function(X_val), dtype=np.float64).ravel()
            if val_scores.size != y_val.size or not np.all(np.isfinite(val_scores)):
                raise ValueError("decision_function returned invalid validation scores")
            threshold = float(
                select_threshold_min_cost(y_val, val_scores, c_fn=c_fn, c_fp=c_fp)
            )
            val_metrics = evaluate_scores(
                y_val,
                val_scores,
                c_fn=c_fn,
                c_fp=c_fp,
                threshold=threshold,
            )
        except Exception as exc:
            candidate_failures.append(
                {
                    "C": C,
                    "reason": "validation_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=6),
                    "fit_time_sec": float(fit_time),
                }
            )
            continue

        candidates.append(
            {
                "C": float(C),
                "model": model,
                "fit_time_sec": float(fit_time),
                "threshold": threshold,
                "val_metrics": val_metrics,
                "started_utc": start.isoformat(),
            }
        )

    record["candidate_failures"] = candidate_failures
    if not candidates:
        record["error"] = (
            "All C candidates failed. "
            + "; ".join(
                f"C={failure['C']}: {failure['reason']} ({failure.get('error')})"
                for failure in candidate_failures
            )
        )
        if save:
            save_run(record, results_dir, overwrite=overwrite)
        if logger is not None:
            logger.error("%s/%s seed=%s ratio=%s: %s", experiment, method, seed, cost_ratio, record["error"])
        if raise_errors:
            raise RuntimeError(record["error"])
        return record

    selected = min(
        candidates,
        key=lambda candidate: (
            candidate["val_metrics"]["cost_per_example"],
            candidate["C"],
        ),
    )
    model = selected["model"]
    selected_threshold = float(selected["threshold"])
    record["C"] = float(selected["C"])
    record["fit_time_sec"] = float(selected["fit_time_sec"])
    record["tuning_fit_time_sec"] = float(
        sum(candidate["fit_time_sec"] for candidate in candidates)
        + sum(failure.get("fit_time_sec") or 0.0 for failure in candidate_failures)
    )
    record["threshold"] = selected_threshold
    record["val_cost_per_example"] = float(selected["val_metrics"]["cost_per_example"])
    record["val_pr_auc"] = float(selected["val_metrics"]["pr_auc"])
    record["candidate_diagnostics"] = [
        {
            "C": candidate["C"],
            "fit_time_sec": candidate["fit_time_sec"],
            "val_cost_per_example": candidate["val_metrics"]["cost_per_example"],
            "val_pr_auc": candidate["val_metrics"]["pr_auc"],
            "threshold": candidate["threshold"],
        }
        for candidate in candidates
    ]

    try:
        test_scores = np.asarray(model.decision_function(X_test), dtype=np.float64).ravel()
        if test_scores.size != y_test.size or not np.all(np.isfinite(test_scores)):
            raise ValueError("decision_function returned invalid test scores")
        test_metrics = evaluate_scores(
            y_test,
            test_scores,
            c_fn=c_fn,
            c_fp=c_fp,
            threshold=selected_threshold,
        )
    except Exception as exc:
        record["error"] = f"test evaluation failed: {type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=6)
        if save:
            save_run(record, results_dir, overwrite=overwrite)
        if logger is not None:
            logger.error("%s/%s seed=%s ratio=%s: %s", experiment, method, seed, cost_ratio, record["error"])
        if raise_errors:
            raise
        return record

    C_per_sample = make_per_sample_costs(
        y_train,
        C=record["C"],
        cost_ratio=cost_ratio,
        c_fp=c_fp,
    )
    objective_reported_value = get_attr(
        model,
        ("objective", "final_objective", "primal_objective", "objective_value"),
    )
    objective_reported: float | None
    try:
        objective_reported = (
            float(objective_reported_value)
            if objective_reported_value is not None
            else None
        )
    except (TypeError, ValueError):
        objective_reported = None
    if objective_reported is not None and not math.isfinite(objective_reported):
        objective_reported = None

    objective_recomputed = compute_primal_objective(model, X_train, y_train, C_per_sample)
    consistency_error: float | None = None
    history_consistent: bool | None = None
    if objective_reported is not None and objective_recomputed is not None:
        denominator = max(abs(objective_recomputed), 1e-12)
        consistency_error = abs(objective_reported - objective_recomputed) / denominator
        history_consistent = bool(consistency_error <= 1e-3)
    objective = (
        objective_recomputed if objective_recomputed is not None else objective_reported
    )

    objective_history, time_history = extract_history(model)
    history_final_error: float | None = None
    if objective_history is not None and objective is not None:
        finite_history = objective_history[np.isfinite(objective_history)]
        if finite_history.size:
            history_final = float(finite_history[-1])
            history_final_error = abs(history_final - objective) / max(
                abs(objective), 1e-12
            )
            if history_final_error > 1e-3:
                history_consistent = False
    n_iter = get_attr(model, ("n_iter", "n_iter_", "iterations", "nit", "num_iter"))
    if n_iter is None and objective_history is not None:
        n_iter = int(objective_history.size)
    try:
        n_iter = int(n_iter) if n_iter is not None else None
    except (TypeError, ValueError):
        n_iter = None
    converged_value = get_attr(model, ("converged", "is_converged", "success"))
    converged = bool(converged_value) if converged_value is not None else None

    sampled_history, sampled_times, truncated = _downsample_history(
        objective_history,
        time_history,
        history_max_points,
    )

    alphas = extract_vector(model, ("alphas", "alpha"))
    alphas_summary: dict[str, Any] | None = None
    if alphas is not None:
        alphas_summary = {
            "count": int(alphas.size),
            "nonzero": int(np.sum(np.abs(alphas) > 1e-12)),
            "sum": float(np.sum(alphas)),
            "min": float(np.min(alphas)),
            "max": float(np.max(alphas)),
        }
    rho_raw = get_attr(model, ("rho_history", "rho_hist"))
    rho_history: list[float] | None = None
    rho_final: float | None = None
    if rho_raw is not None:
        rho_values = np.asarray(rho_raw, dtype=np.float64).ravel()
        if rho_values.size and np.all(np.isfinite(rho_values)):
            rho_history, _, _ = _downsample_history(rho_values, None, 1000)
            rho_final = float(rho_values[-1])

    record.update(
        {
            "n_iter": n_iter,
            "converged": converged,
            "objective": objective,
            "objective_reported": objective_reported,
            "objective_recomputed": objective_recomputed,
            "objective_consistency_rel_error": consistency_error,
            "history_final_consistency_rel_error": history_final_error,
            "history_consistent": history_consistent,
            "pr_auc": float(test_metrics["pr_auc"]),
            "recall_at_fpr_0.001": float(test_metrics["recall_at_fpr_0.001"]),
            "recall_at_fpr_0.01": float(test_metrics["recall_at_fpr_0.01"]),
            "cost_per_example": float(test_metrics["cost_per_example"]),
            "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "threshold": float(test_metrics["threshold"]),
            "pr_curve": precision_recall_points(y_test, test_scores, max_points=200),
            "peak_memory_mb": _process_peak_memory_mb(),
            "history_available": objective_history is not None,
            "history_truncated": bool(truncated),
            "objective_history": sampled_history,
            "time_history": sampled_times,
            "alphas_summary": alphas_summary,
            "rho_history": rho_history,
            "rho_final": rho_final,
            "status": "ok",
            "error": None,
        }
    )

    if reference_objective is not None and objective is not None:
        record["reference_objective"] = float(reference_objective)
        relative_gap = _relative_gap(objective, reference_objective)
        record["relative_gap"] = relative_gap
        record["reached_tolerance"] = bool(
            relative_gap is not None and relative_gap <= float(tolerance)
        )
        if objective_history is None:
            record["tolerance_note"] = "no_objective_history"
        elif history_consistent is False:
            record["tolerance_note"] = "solver_objective_inconsistent_with_recomputed_objective"
        else:
            reach_time, _, _ = time_to_tolerance(
                objective_history,
                time_history,
                float(reference_objective),
                float(tolerance),
            )
            record["time_to_tolerance_sec"] = reach_time
            record["tolerance_note"] = (
                None
                if reach_time is not None
                else "tolerance_not_reached_or_no_timing_history"
            )
    elif reference_objective is None:
        record["tolerance_note"] = "reference_objective_not_available"
    else:
        record["tolerance_note"] = "objective_unavailable"

    if save:
        save_run(record, results_dir, overwrite=overwrite)
    if logger is not None:
        logger.info(
            "%-40s method=%-8s ratio=%-3s seed=%s C=%-6g cost=%.6g "
            "pr_auc=%s t_tol=%s",
            dataset,
            method,
            _clean_ratio(cost_ratio),
            seed,
            record["C"],
            record["cost_per_example"],
            _format_optional(record["pr_auc"]),
            _format_optional(record["time_to_tolerance_sec"]),
        )
    return record


def _format_optional(value: float | None) -> str:
    return "pending" if value is None else f"{value:.4g}"


class _WallTimer:
    def __init__(self) -> None:
        import time

        self._time = time
        self.start: float | None = None
        self.end: float | None = None

    def __enter__(self) -> "_WallTimer":
        self.start = self._time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.end = self._time.perf_counter()

    @property
    def elapsed(self) -> float:
        if self.start is None:
            return 0.0
        stop = self.end if self.end is not None else self._time.perf_counter()
        return float(stop - self.start)


def _stratified_cap_indices(y: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    n_samples = y.size
    max_n = int(max_n)
    if max_n >= n_samples:
        return np.arange(n_samples)
    if max_n <= 0:
        raise ValueError("max_n must be positive")
    try:
        from sklearn.model_selection import train_test_split

        selected, _ = train_test_split(
            np.arange(n_samples),
            train_size=max_n,
            stratify=y,
            random_state=int(seed),
        )
        return np.sort(selected).astype(np.int64)
    except ValueError:
        rng = np.random.default_rng(int(seed))
        return np.sort(rng.choice(n_samples, size=max_n, replace=False)).astype(np.int64)


def _cap_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cap: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    meta: dict[str, Any] = {
        "train_cap": None if cap is None else int(cap),
        "cap_applied": False,
        "n_train_before_cap": int(X_train.shape[0]),
    }
    if cap is None or int(cap) >= X_train.shape[0]:
        return X_train, y_train, meta
    indices = _stratified_cap_indices(y_train, int(cap), seed)
    meta["cap_applied"] = True
    meta["n_train_after_cap"] = int(indices.size)
    return X_train[indices], y_train[indices], meta


def _apply_data_variant(
    X_train: np.ndarray,
    y_train: np.ndarray,
    variant: Mapping[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    variant_type = variant.get("type")
    meta: dict[str, Any] = {"data_variant_type": variant_type}
    if variant_type is None:
        return X_train, y_train, meta
    if variant_type == "positive_subsample":
        fraud_rate = float(variant["fraud_rate"])
        kept_positive = data_module.subsample_positive_indices(
            y_train,
            fraud_rate=fraud_rate,
            seed=seed,
        )
        kept_negative = np.flatnonzero(y_train <= 0)
        keep = np.sort(np.concatenate([kept_negative, kept_positive]))
        reduced_X = X_train[keep]
        reduced_y = y_train[keep]
        meta.update(
            {
                "target_fraud_rate": fraud_rate,
                "train_positives_before_variant": int(np.sum(y_train > 0)),
                "train_positives_after_variant": int(np.sum(reduced_y > 0)),
                "realized_fraud_rate": float(np.mean(reduced_y > 0)),
            }
        )
        return reduced_X, reduced_y, meta
    if variant_type == "class_rate":
        fraud_rate = float(variant["fraud_rate"])
        keep, mechanism = data_module.subsample_training_to_rate(
            y_train,
            fraud_rate=fraud_rate,
            seed=seed,
        )
        reduced_X = X_train[keep]
        reduced_y = y_train[keep]
        meta.update(
            {
                "target_fraud_rate": fraud_rate,
                "prevalence_mechanism": mechanism,
                "train_positives_before_variant": int(np.sum(y_train > 0)),
                "train_positives_after_variant": int(np.sum(reduced_y > 0)),
                "train_negatives_before_variant": int(np.sum(y_train <= 0)),
                "train_negatives_after_variant": int(np.sum(reduced_y <= 0)),
                "realized_fraud_rate": float(np.mean(reduced_y > 0)),
            }
        )
        return reduced_X, reduced_y, meta
    if variant_type == "label_noise":
        rate = float(variant["rate"])
        flipped = data_module.flip_labels(y_train, rate=rate, seed=seed)
        meta.update(
            {
                "label_noise_rate": rate,
                "n_labels_flipped": int(np.sum(flipped != y_train)),
            }
        )
        return X_train, flipped, meta
    raise ValueError(f"Unknown data variant type: {variant_type}")


def _build_ulb_data(
    config: Mapping[str, Any],
    cleaned: pd.DataFrame,
    seed: int,
    data_variant: Mapping[str, Any],
    logger: Any,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    str,
]:
    data_cfg = dict(config.get("data", {}))
    protocol = data_variant.get("split_protocol", data_cfg.get("split_protocol", "stratified"))
    if protocol not in ("stratified", "chronological"):
        raise ValueError(f"Unknown split_protocol: {protocol}")

    subset_n = data_cfg.get("subset_n")
    if subset_n is not None:
        subset_seed_setting = data_cfg.get("subset_seed", "per_seed")
        subset_seed = seed if subset_seed_setting == "per_seed" else int(subset_seed_setting)
        frame = data_module.stratified_subsample(cleaned, int(subset_n), subset_seed)
        split_source = f"generated-subset-{int(subset_n)}"
    else:
        frame = cleaned
        split_source = "generated"

    splits: dict[str, np.ndarray] | None = None
    use_saved = bool(data_cfg.get("use_saved_splits", True)) and subset_n is None
    if use_saved:
        splits_dir = Path(data_cfg.get("splits_dir", "data/splits"))
        if protocol == "stratified":
            split_path = splits_dir / f"stratified_seed{int(seed)}.json"
        else:
            split_path = splits_dir / f"chronological_seed{int(seed)}.json"
        if split_path.exists():
            splits = data_module.load_splits(split_path)
            split_source = f"saved:{split_path}"

    if splits is None:
        if protocol == "stratified":
            splits = data_module.make_stratified_splits(
                frame[data_module.LABEL_COLUMN].to_numpy(),
                seed=seed,
            )
        else:
            splits = data_module.make_chronological_split(frame)

    if logger is not None:
        logger.info(
            "Split protocol=%s source=%s seed=%s sizes=(%d,%d,%d) data_variant=%s",
            protocol,
            split_source,
            seed,
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
            data_variant.get("name", "base"),
        )

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = data_module.prepare_arrays(
        frame,
        splits,
        standardize_amount=bool(data_cfg.get("standardize_amount", True)),
    )

    meta: dict[str, Any] = {
        "source": "ulb",
        "split_protocol": protocol,
        "split_source": split_source,
        "subset_n": None if subset_n is None else int(subset_n),
        "clean_rows": int(len(cleaned)),
        "clean_positives": int(np.sum(cleaned[data_module.LABEL_COLUMN] == 1)),
        "n_train_before_variant": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "data_variant": data_variant.get("name", "base"),
    }
    for name, indices in splits.items():
        subset = frame.iloc[indices]
        meta[f"{name}_positives"] = int(np.sum(subset[data_module.LABEL_COLUMN] == 1))
        meta[f"{name}_fraud_rate"] = float(subset[data_module.LABEL_COLUMN].mean())
    if data_module.TIME_COLUMN in frame.columns:
        train_times = frame.iloc[splits["train"]][data_module.TIME_COLUMN]
        val_times = frame.iloc[splits["val"]][data_module.TIME_COLUMN]
        test_times = frame.iloc[splits["test"]][data_module.TIME_COLUMN]
        meta.update(
            {
                "train_time_min": float(train_times.min()),
                "train_time_max": float(train_times.max()),
                "val_time_min": float(val_times.min()),
                "val_time_max": float(val_times.max()),
                "test_time_min": float(test_times.min()),
                "test_time_max": float(test_times.max()),
            }
        )

    X_train, y_train, variant_meta = _apply_data_variant(
        X_train,
        y_train,
        data_variant,
        seed=seed,
    )
    meta.update(variant_meta)
    meta["n_train_after_variant"] = int(X_train.shape[0])
    meta["train_fraud_rate_after_variant"] = float(np.mean(y_train > 0))

    base_dataset = str(config.get("dataset", "ulb"))
    variant_name = data_variant.get("name")
    dataset_id = (
        base_dataset
        if variant_name in (None, "", "base")
        else f"{base_dataset}__{_slug(str(variant_name))}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, meta, dataset_id


def _synthetic_dataset_id(n_samples: int) -> str:
    if n_samples % 1000 == 0:
        return f"synthetic-{n_samples // 1000}k"
    return f"synthetic-{n_samples}"


def _build_synthetic_data(
    config: Mapping[str, Any],
    n_samples: int,
    seed: int,
    data_variant: Mapping[str, Any],
    logger: Any,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    str,
]:
    data_cfg = dict(config.get("data", {}))
    X, y = data_module.make_synthetic_fraud(
        n=int(n_samples),
        d=int(data_cfg.get("synthetic_d", 20)),
        fraud_rate=float(data_cfg.get("synthetic_fraud_rate", 0.01)),
        seed=seed,
        separation=float(data_cfg.get("synthetic_separation", 3.0)),
    )
    splits = data_module.make_stratified_splits(y, seed=seed)
    X_train, X_val, X_test = X[splits["train"]], X[splits["val"]], X[splits["test"]]
    y_train, y_val, y_test = y[splits["train"]], y[splits["val"]], y[splits["test"]]

    X_train, y_train, variant_meta = _apply_data_variant(
        X_train,
        y_train,
        data_variant,
        seed=seed,
    )
    meta: dict[str, Any] = {
        "source": "synthetic",
        "split_protocol": "stratified",
        "split_source": "generated",
        "synthetic_n": int(n_samples),
        "synthetic_d": int(X.shape[1]),
        "synthetic_fraud_rate": float(data_cfg.get("synthetic_fraud_rate", 0.01)),
        "synthetic_separation": float(data_cfg.get("synthetic_separation", 3.0)),
        "n_train_before_variant": int(np.sum(splits["train"].size)),
        "n_train_after_variant": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "data_variant": data_variant.get("name", "base"),
        **variant_meta,
    }
    if logger is not None:
        logger.info(
            "Synthetic n=%d d=%d seed=%d sizes=(%d,%d,%d)",
            n_samples,
            X.shape[1],
            seed,
            X_train.shape[0],
            X_val.shape[0],
            X_test.shape[0],
        )
    dataset_id = _synthetic_dataset_id(n_samples)
    variant_name = data_variant.get("name")
    if variant_name not in (None, "", "base"):
        dataset_id = f"{dataset_id}__{_slug(str(variant_name))}"
    return X_train, y_train, X_val, y_val, X_test, y_test, meta, dataset_id


def _method_job_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    base_kwargs = dict(config.get("method_kwargs", {}))
    method_variants = config.get("method_variants") or []
    if method_variants:
        specs: list[dict[str, Any]] = []
        for variant in method_variants:
            method = str(variant["method"])
            kwargs = dict(base_kwargs.get(method, {}))
            kwargs.update(variant.get("method_kwargs", {}))
            specs.append(
                {
                    "method": method,
                    "variant": variant.get("name") or variant.get("variant"),
                    "method_kwargs": kwargs,
                }
            )
        return specs

    specs = []
    for method in config.get("methods", []):
        specs.append(
            {
                "method": str(method),
                "variant": None,
                "method_kwargs": dict(base_kwargs.get(method, {})),
            }
        )
    return specs


def _data_variant_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    variants = config.get("data_variants") or []
    return [dict(variant) for variant in variants] if variants else [{"name": "base"}]


def _assign_reference_objectives(
    results: list[dict[str, Any]],
    config: Mapping[str, Any],
    results_dir: str | Path,
    logger: Any,
) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in results:
        if record.get("status") != "ok" or record.get("objective") is None:
            continue
        key = (
            record["dataset"],
            float(record["cost_ratio"]),
            int(record["seed"]),
            int(record["n_train"]),
        )
        groups.setdefault(key, []).append(record)

    reference_method = config.get("reference_method")
    for records in groups.values():
        objectives = [
            float(record["objective"])
            for record in records
            if record.get("objective") is not None
        ]
        if not objectives:
            continue
        if reference_method:
            method_objectives = [
                float(record["objective"])
                for record in records
                if record.get("method") == reference_method
                and record.get("objective") is not None
            ]
            reference = min(method_objectives) if method_objectives else min(objectives)
        else:
            reference = min(objectives)

        for record in records:
            objective = record.get("objective")
            record["reference_objective"] = float(reference)
            relative_gap = _relative_gap(objective, reference)
            record["relative_gap"] = relative_gap
            tolerance = float(record.get("tolerance", config.get("tolerance", 1e-4)))
            reached = bool(
                relative_gap is not None and relative_gap <= tolerance
            )
            record["reached_tolerance"] = reached

            history = record.get("objective_history")
            times = record.get("time_history")
            if history is None:
                record["time_to_tolerance_sec"] = None
                record["tolerance_note"] = "no_objective_history"
            elif record.get("history_consistent") is False:
                record["time_to_tolerance_sec"] = None
                record["tolerance_note"] = (
                    "solver_objective_inconsistent_with_recomputed_objective"
                )
            else:
                reach_time, _, _ = time_to_tolerance(
                    history,
                    times,
                    reference,
                    tolerance,
                )
                record["time_to_tolerance_sec"] = reach_time
                record["tolerance_note"] = (
                    None
                    if reach_time is not None
                    else "tolerance_not_reached_or_no_timing_history"
                )
            save_run(record, results_dir, overwrite=True)

        if logger is not None:
            logger.info(
                "Reference objective %.6g for dataset=%s ratio=%s seed=%s n_train=%s",
                reference,
                records[0]["dataset"],
                records[0]["cost_ratio"],
                records[0]["seed"],
                records[0]["n_train"],
            )


def run_experiment(
    config: Mapping[str, Any],
    *,
    dry_run: bool = False,
    logger: Any = None,
    method_registry_extra: Mapping[str, Any] | None = None,
    overwrite: bool | None = None,
) -> list[dict[str, Any]]:
    """Execute (or describe) the experiment matrix in ``config``."""
    experiment = str(config.get("experiment", "unnamed"))
    methods = _method_job_specs(config)
    if not methods:
        raise ValueError("Config must define 'methods' or 'method_variants'")
    seeds = [int(seed) for seed in config.get("seeds", [0])]
    ratios = [_clean_ratio(ratio) for ratio in config.get("cost_ratios", [10])]
    data_variants = _data_variant_specs(config)
    data_cfg = dict(config.get("data", {}))
    source = data_cfg.get("source", "ulb")
    results_dir = Path(config.get("paths", {}).get("results_dir", "results"))
    overwrite_setting = (
        bool(config.get("overwrite", True)) if overwrite is None else bool(overwrite)
    )

    if dry_run:
        print(f"# Resolved config: {experiment}")
        print(yaml.safe_dump(dict(config), sort_keys=False))
        job_count = 0
        print("# Planned jobs (solver availability is not checked in dry-run)")
        scale_values: list[int | None]
        if source == "synthetic":
            scale_values = [int(n) for n in data_cfg.get("synthetic_n_list", [10000])]
        else:
            scale_values = [None]
        for data_variant in data_variants:
            allowed_methods = data_variant.get("methods")
            for scale in scale_values:
                for seed in seeds:
                    for ratio in ratios:
                        for spec in methods:
                            if allowed_methods and spec["method"] not in allowed_methods:
                                continue
                            cap = config.get("method_caps", {}).get(spec["method"])
                            job_count += 1
                            subset_label = data_cfg.get("subset_n")
                            if subset_label is None:
                                subset_label = "full"
                            print(
                                "  "
                                f"data={data_variant.get('name', 'base')} "
                                f"scale={scale if scale is not None else subset_label} "
                                f"seed={seed} ratio={ratio} method={spec['method']} "
                                f"variant={spec['variant']} cap={cap} "
                                f"C_grid={config.get('C_grid', [1.0])} "
                                f"tol={config.get('tolerance', 1e-4)} "
                                f"budget={config.get('time_budget_sec', 300)}s"
                            )
        print(f"# Total planned jobs: {job_count}")
        return []

    configure_single_thread()
    if logger is None:
        log_dir = config.get("paths", {}).get("log_dir")
        log_file = None
        if log_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = Path(log_dir) / f"{_slug(experiment)}_{timestamp}.log"
        logger = get_logger(
            f"wp4.{_slug(experiment)}",
            log_file=log_file,
            level=config.get("logging", {}).get("level", "INFO"),
        )

    logger.info("Starting experiment=%s methods=%s seeds=%s ratios=%s", experiment, [spec["method"] for spec in methods], seeds, ratios)

    availability: dict[str, tuple[bool, str | None]] = {}
    for spec in methods:
        name = spec["method"]
        if name not in availability:
            availability[name] = method_available(name, method_registry_extra)
        available, reason = availability[name]
        if not available:
            logger.error("Skipping method '%s': %s", name, reason)

    runnable_specs = [
        spec for spec in methods if availability.get(spec["method"], (False, None))[0]
    ]
    if not runnable_specs:
        logger.error(
            "No solver modules are available. Expected src/svm_smo.py, "
            "src/svm_pegasos.py, src/svm_admm.py, or src/baseline_cvxpy.py. "
            "No fake results were produced."
        )
        return []

    if source == "ulb":
        raw_path = data_cfg.get("raw_path", "data/raw/creditcard.csv")
        raw_frame = data_module.load_raw(raw_path)
        cleaned = data_module.clean_data(
            raw_frame,
            drop_duplicates=bool(data_cfg.get("drop_duplicates", True)),
        )
        logger.info(
            "ULB data: raw=%d rows, cleaned=%d rows, positives=%d, duplicates_removed=%d",
            len(raw_frame),
            len(cleaned),
            int(np.sum(cleaned[data_module.LABEL_COLUMN] == 1)),
            len(raw_frame) - len(cleaned),
        )
        scale_values = [None]
    elif source == "synthetic":
        cleaned = None
        scale_values = [int(n) for n in data_cfg.get("synthetic_n_list", [10000])]
    else:
        raise ValueError(f"Unknown data source: {source}")

    results: list[dict[str, Any]] = []
    for data_variant in data_variants:
        allowed_methods = data_variant.get("methods")
        specs_for_variant = [
            spec
            for spec in runnable_specs
            if not allowed_methods or spec["method"] in allowed_methods
        ]
        if not specs_for_variant:
            continue
        for scale in scale_values:
            for seed in seeds:
                set_seed(seed)
                if source == "ulb":
                    (
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        X_test,
                        y_test,
                        data_meta,
                        dataset_id,
                    ) = _build_ulb_data(
                        config,
                        cleaned,
                        seed,
                        data_variant,
                        logger,
                    )
                else:
                    (
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        X_test,
                        y_test,
                        data_meta,
                        dataset_id,
                    ) = _build_synthetic_data(
                        config,
                        int(scale),
                        seed,
                        data_variant,
                        logger,
                    )

                for ratio in ratios:
                    for spec in specs_for_variant:
                        cap = config.get("method_caps", {}).get(spec["method"])
                        X_fit, y_fit, cap_meta = _cap_training_data(
                            X_train,
                            y_train,
                            cap=cap,
                            seed=seed,
                        )
                        job_meta = dict(data_meta)
                        job_meta.update(cap_meta)
                        job_meta["method_variant"] = spec["variant"]
                        logger.info(
                            "Running experiment=%s dataset=%s method=%s variant=%s "
                            "ratio=%s seed=%s n_train=%d",
                            experiment,
                            dataset_id,
                            spec["method"],
                            spec["variant"],
                            ratio,
                            seed,
                            X_fit.shape[0],
                        )
                        record = run_one(
                            experiment=experiment,
                            dataset=dataset_id,
                            method=spec["method"],
                            seed=seed,
                            cost_ratio=ratio,
                            X_train=X_fit,
                            y_train=y_fit,
                            X_val=X_val,
                            y_val=y_val,
                            X_test=X_test,
                            y_test=y_test,
                            C_grid=config.get("C_grid", [1.0]),
                            tolerance=float(config.get("tolerance", 1e-4)),
                            time_budget_sec=float(
                                config.get("time_budget_sec", 300.0)
                            ),
                            method_kwargs=spec["method_kwargs"],
                            method_registry_extra=method_registry_extra,
                            variant=spec["variant"],
                            results_dir=results_dir,
                            reference_objective=None,
                            overwrite=overwrite_setting,
                            history_max_points=int(
                                config.get("history_max_points", 10000)
                            ),
                            logger=logger,
                            data_meta=job_meta,
                        )
                        results.append(record)

    _assign_reference_objectives(results, config, results_dir, logger)

    if config.get("aggregate", True) and results:
        summary = aggregate_runs(results_dir, experiment, logger=logger)
        logger.info("Wrote aggregated tables for %s (%d rows)", experiment, len(summary))

    return results


def _record_for_csv(record: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in record.items():
        if key in ("objective_history", "time_history", "pr_curve", "rho_history"):
            continue
        if isinstance(value, (dict, list, tuple)):
            row[key] = json.dumps(value, default=str)
        else:
            row[key] = value
    return row


def aggregate_runs(
    results_dir: str | Path = "results",
    experiment: str = "main_full",
    *,
    logger: Any = None,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """Aggregate run JSONs into per-run and per-metric CSV tables."""
    raw_root = Path(results_dir) / "raw" / _slug(experiment)
    files = sorted(raw_root.glob("**/seed-*.json"))
    if not files:
        if logger is not None:
            logger.warning("No run JSONs found under %s", raw_root)
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        record.setdefault("run_file", str(path))
        records.append(record)

    runs_frame = pd.DataFrame([_record_for_csv(record) for record in records])
    runs_frame = runs_frame.fillna({"variant": "base"})
    runs_frame["variant"] = runs_frame["variant"].fillna("base")

    tables_dir = Path(results_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    runs_frame.to_csv(tables_dir / f"{_slug(experiment)}_runs.csv", index=False)

    group_keys = ["dataset", "method", "variant", "cost_ratio", "n_train"]
    for key in group_keys:
        if key not in runs_frame.columns:
            runs_frame[key] = None

    if "status" in runs_frame.columns:
        successful_frame = runs_frame[runs_frame["status"] == "ok"].copy()
    else:
        successful_frame = runs_frame.copy()

    if successful_frame.empty:
        if logger is not None:
            logger.warning("No successful runs to aggregate for %s", experiment)
        return pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    for metric in METRICS_FOR_TABLES:
        if metric not in successful_frame.columns:
            continue
        metric_rows: list[dict[str, Any]] = []
        for keys, group in successful_frame.groupby(group_keys, dropna=False):
            key_values = dict(zip(group_keys, keys))
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            n_values = int(values.size)
            if n_values:
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if n_values > 1 else 0.0
                ci_low, ci_high = bootstrap_ci(values, n_boot=n_boot, seed=0)
            else:
                mean = std = ci_low = ci_high = float("nan")
            metric_rows.append(
                {
                    **key_values,
                    "n": n_values,
                    "mean": mean,
                    "std": std,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
            summary_rows.append(
                {
                    **key_values,
                    "metric": metric,
                    "n": n_values,
                    "mean": mean,
                    "std": std,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
        metric_frame = pd.DataFrame(metric_rows)
        metric_frame.to_csv(
            tables_dir / f"{_slug(experiment)}_{metric}.csv",
            index=False,
        )

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(
        tables_dir / f"{_slug(experiment)}_summary.csv",
        index=False,
    )
    if logger is not None:
        logger.info(
            "Aggregated %d run(s) into %s",
            len(records),
            tables_dir,
        )
    return summary_frame
