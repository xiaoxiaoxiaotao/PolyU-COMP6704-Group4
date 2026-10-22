"""Solver-facing interface for WP1-WP3, shared with the evaluation harness.

The harness calls ``SolverClass().fit(X, y, C_per_sample, **kwargs)`` and then
reads a small set of diagnostics. This module defines that contract, common
validation helpers, and a contract checker that each solver owner can run
before integrating with the experiment matrix.

Importing this module does not import any solver.
"""

from __future__ import annotations

import inspect
import math
import time
from abc import ABC, abstractmethod
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class BinarySVM(Protocol):
    """Structural protocol for every solver in the comparison."""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        C_per_sample: np.ndarray,
        **kwargs: Any,
    ) -> Any: ...

    def decision_function(self, X: np.ndarray) -> np.ndarray: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


class BaseBinarySVM(ABC):
    """Optional convenience base class.

    Inheritance is not required; the harness uses structural typing. The base
    class provides input validation helpers and a default zero-threshold
    ``predict``. A solver still owns ``fit`` and ``decision_function``.
    """

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        C_per_sample: np.ndarray,
        **kwargs: Any,
    ) -> Any: ...

    @abstractmethod
    def decision_function(self, X: np.ndarray) -> np.ndarray: ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_array = np.asarray(X, dtype=np.float64)
        scores = validate_decision_scores(
            self.decision_function(X_array),
            n_samples=X_array.shape[0],
        )
        return np.where(scores >= 0.0, 1, -1).astype(np.int8)


def validate_training_data(
    X: np.ndarray,
    y: np.ndarray,
    C_per_sample: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the training contract and return normalized arrays."""
    X_array = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y).ravel()
    costs = np.asarray(C_per_sample, dtype=np.float64).ravel()

    if X_array.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X_array.shape}")
    if X_array.shape[0] == 0 or X_array.shape[1] == 0:
        raise ValueError("X is empty")
    if labels.size != X_array.shape[0]:
        raise ValueError(
            f"y has {labels.size} entries but X has {X_array.shape[0]} rows"
        )
    if costs.size != X_array.shape[0]:
        raise ValueError(
            f"C_per_sample has {costs.size} entries but X has "
            f"{X_array.shape[0]} rows"
        )
    if not np.all(np.isin(labels, (-1, 1))):
        raise ValueError("y must contain only -1 and +1")
    if not np.all(np.isfinite(costs)):
        raise ValueError("C_per_sample contains non-finite values")
    if np.any(costs <= 0):
        raise ValueError("C_per_sample must be strictly positive")
    if not np.all(np.isfinite(X_array)):
        raise ValueError("X contains non-finite values")
    return X_array, labels.astype(np.int8), costs


def validate_decision_scores(
    scores: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Return a finite 1-D score vector of the expected length."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim == 2 and 1 in values.shape:
        values = values.ravel()
    if values.ndim != 1:
        raise ValueError(
            f"decision_function must return a 1-D array, got shape {values.shape}"
        )
    if values.size != int(n_samples):
        raise ValueError(
            f"decision_function returned {values.size} scores for "
            f"{int(n_samples)} rows"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("decision_function returned non-finite scores")
    return values


def validate_predictions(
    predictions: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Return a 1-D {-1, +1} prediction vector of the expected length."""
    values = np.asarray(predictions).ravel()
    if values.size != int(n_samples):
        raise ValueError(
            f"predict returned {values.size} labels for {int(n_samples)} rows"
        )
    if not np.all(np.isin(values, (-1, 1))):
        raise ValueError("predict must return only -1 and +1")
    return values.astype(np.int8)


def filtered_fit_kwargs(fit_callable: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only kwargs accepted by ``fit``; pass all if it has ``**kwargs``."""
    try:
        signature = inspect.signature(fit_callable)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(kwargs)
    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in kwargs.items() if key in accepted}


def get_attr(model: Any, names: Sequence[str]) -> Any:
    """Read a diagnostic from attributes or a ``history`` mapping."""
    for name in names:
        if hasattr(model, name):
            value = getattr(model, name)
            if value is not None:
                return value
    history = getattr(model, "history", None)
    if isinstance(history, Mapping):
        for name in names:
            if name in history and history[name] is not None:
                return history[name]
    return None


def extract_history(model: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract ``(objective_history, time_history)`` from a fitted solver."""
    objective_history = get_attr(
        model,
        (
            "objective_history",
            "objective_hist",
            "objectives",
            "primal_history",
            "obj_history",
        ),
    )
    time_history = get_attr(
        model,
        (
            "time_history",
            "times",
            "timestamps",
            "iteration_times",
            "elapsed_times",
        ),
    )

    history = getattr(model, "history", None)
    if (
        objective_history is None
        and isinstance(history, (list, tuple))
        and history
        and isinstance(history[0], Mapping)
    ):
        objective_history = [
            entry.get("objective", entry.get("obj", entry.get("primal_objective")))
            for entry in history
        ]
        if all(value is None for value in objective_history):
            objective_history = None
        if time_history is None:
            time_history = [
                entry.get("time", entry.get("elapsed_sec", entry.get("time_sec")))
                for entry in history
            ]
            if all(value is None for value in time_history):
                time_history = None

    if objective_history is None:
        return None, None

    objectives = np.asarray(objective_history, dtype=np.float64).ravel()
    if objectives.size == 0:
        return None, None
    if time_history is None:
        return objectives, None
    times = np.asarray(time_history, dtype=np.float64).ravel()
    if times.size != objectives.size:
        return objectives, None
    return objectives, times


def extract_vector(model: Any, names: Sequence[str]) -> np.ndarray | None:
    """Extract a finite 1-D vector diagnostic (e.g. weights or alphas)."""
    value = get_attr(model, names)
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if array.size == 0 or not np.all(np.isfinite(array)):
        return None
    return array


def extract_bias(model: Any) -> float | None:
    """Extract a scalar bias/intercept diagnostic."""
    value = get_attr(model, ("b", "bias", "intercept_"))
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if array.size != 1 or not np.isfinite(array[0]):
        return None
    return float(array[0])


def compute_primal_objective(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    C_per_sample: np.ndarray,
) -> float | None:
    """Canonical weighted objective from the final linear model.

    Convention (matching the proposal and EXPERIMENTS.md):

        0.5 * ||w||^2 + sum_i C_i * max(0, 1 - y_i (x_i^T w + b))

    The unnormalized sum is intentional; do not divide by n.
    """
    weights = extract_vector(model, ("w", "weights", "coef_"))
    bias = extract_bias(model)
    if weights is None or bias is None:
        return None
    X_array = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y).ravel()
    if weights.size != X_array.shape[1] or labels.size != X_array.shape[0]:
        return None
    margins = labels.astype(np.float64) * (X_array @ weights + bias)
    hinge = np.maximum(0.0, 1.0 - margins)
    objective = 0.5 * float(weights @ weights) + float(
        np.sum(np.asarray(C_per_sample, dtype=np.float64).ravel() * hinge)
    )
    if not math.isfinite(objective):
        return None
    return objective


def check_solver_contract(
    method_class: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    C_per_sample: np.ndarray,
    *,
    X_check: np.ndarray | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
    objective_rtol: float = 1e-3,
) -> dict[str, Any]:
    """Fit a solver once and verify the interface and diagnostic contract.

    Returns a JSON-serializable report. ``passed`` is true only when there are
    no errors; warnings identify optional diagnostics that weaken the primary
    time-to-tolerance metric.
    """
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {
        "method_class": getattr(method_class, "__name__", str(method_class)),
        "method_module": getattr(method_class, "__module__", None),
    }

    try:
        X, y, costs = validate_training_data(X_train, y_train, C_per_sample)
    except Exception as exc:
        return {
            "passed": False,
            "errors": [f"training data contract: {exc}"],
            "warnings": warnings,
            "details": details,
        }

    check_X = X if X_check is None else np.asarray(X_check, dtype=np.float64)
    if check_X.ndim != 2 or check_X.shape[1] != X.shape[1]:
        return {
            "passed": False,
            "errors": ["X_check must be 2-D with the same feature width as X"],
            "warnings": warnings,
            "details": details,
        }

    try:
        model = method_class()
    except Exception as exc:
        return {
            "passed": False,
            "errors": [f"constructor failed: {type(exc).__name__}: {exc}"],
            "warnings": warnings,
            "details": details,
        }

    kwargs = {"seed": 0, "tol": 1e-4, "time_budget_sec": 300.0}
    kwargs.update(dict(fit_kwargs or {}))
    try:
        start = time.perf_counter()
        model.fit(
            X,
            y,
            costs,
            **filtered_fit_kwargs(model.fit, kwargs),
        )
        fit_time = time.perf_counter() - start
    except Exception as exc:
        return {
            "passed": False,
            "errors": [f"fit failed: {type(exc).__name__}: {exc}"],
            "warnings": warnings,
            "details": details,
        }
    details["fit_time_sec"] = float(fit_time)

    try:
        scores = validate_decision_scores(
            model.decision_function(check_X),
            n_samples=check_X.shape[0],
        )
        details["n_scores"] = int(scores.size)
        details["score_min"] = float(np.min(scores))
        details["score_max"] = float(np.max(scores))
    except Exception as exc:
        errors.append(f"decision_function contract: {exc}")

    try:
        predictions = validate_predictions(
            model.predict(check_X),
            n_samples=check_X.shape[0],
        )
        details["n_predictions"] = int(predictions.size)
        details["predicted_positive_rate"] = float(np.mean(predictions > 0))
    except Exception as exc:
        errors.append(f"predict contract: {exc}")

    weights = extract_vector(model, ("w", "weights", "coef_"))
    bias = extract_bias(model)
    if weights is None:
        errors.append("missing finite attribute 'w' (or weights/coef_)")
    else:
        details["w_norm"] = float(np.linalg.norm(weights))
        if weights.size != X.shape[1]:
            errors.append(
                f"w has {weights.size} entries but X has {X.shape[1]} features"
            )
    if bias is None:
        errors.append("missing finite scalar attribute 'b' (or bias/intercept_)")
    else:
        details["b"] = float(bias)

    n_iter = get_attr(model, ("n_iter", "n_iter_", "iterations", "nit", "num_iter"))
    if n_iter is None:
        warnings.append("missing 'n_iter'; the run log will leave it null")
    else:
        try:
            details["n_iter"] = int(n_iter)
        except (TypeError, ValueError):
            warnings.append("n_iter is not integer-castable")

    converged = get_attr(model, ("converged", "is_converged", "success"))
    if converged is None:
        warnings.append("missing 'converged'; the run log will leave it null")
    else:
        details["converged"] = bool(converged)

    objective_history, time_history = extract_history(model)
    if objective_history is None:
        errors.append(
            "missing 'objective_history'; time-to-tolerance cannot be computed"
        )
    else:
        details["n_objective_history"] = int(objective_history.size)
        details["objective_final"] = float(objective_history[-1])
        if objective_history.size < 2:
            warnings.append(
                "objective_history has fewer than two points; "
                "time_to_tolerance will be null"
            )
        if time_history is None:
            warnings.append(
                "missing aligned 'time_history'; time_to_tolerance will be null"
            )
        else:
            details["n_time_history"] = int(time_history.size)
            details["time_final_sec"] = float(time_history[-1])

    reported_value = get_attr(
        model,
        ("objective", "final_objective", "primal_objective", "objective_value"),
    )
    reported: float | None = None
    if reported_value is not None:
        try:
            reported = float(reported_value)
        except (TypeError, ValueError):
            warnings.append("reported objective is not a scalar")
        if reported is not None and not math.isfinite(reported):
            reported = None
    recomputed = compute_primal_objective(model, X, y, costs)
    if recomputed is not None:
        details["objective_recomputed"] = float(recomputed)
    if reported is None:
        warnings.append(
            "no reported scalar 'objective'; the harness can recompute it from w,b"
        )
    elif recomputed is None:
        warnings.append("objective consistency could not be checked without w,b")
    else:
        relative_error = abs(reported - recomputed) / max(abs(recomputed), 1e-12)
        details["objective_consistency_rel_error"] = float(relative_error)
        if relative_error > objective_rtol:
            errors.append(
                "reported objective is inconsistent with the canonical objective "
                f"from w,b (relative error {relative_error:.3e} > {objective_rtol:.1e})"
            )

    if objective_history is not None and recomputed is not None:
        finite_history = objective_history[np.isfinite(objective_history)]
        if finite_history.size:
            history_final = float(finite_history[-1])
            history_error = abs(history_final - recomputed) / max(
                abs(recomputed), 1e-12
            )
            details["history_final_consistency_rel_error"] = float(history_error)
            if history_error > objective_rtol:
                errors.append(
                    "objective_history final value is inconsistent with the "
                    f"canonical objective from w,b (relative error "
                    f"{history_error:.3e} > {objective_rtol:.1e})"
                )

    alphas = extract_vector(model, ("alphas", "alpha"))
    if alphas is not None:
        details["alphas_nonzero"] = int(np.sum(np.abs(alphas) > 1e-12))
        details["alphas_sum"] = float(np.sum(alphas))
    rho_history = get_attr(model, ("rho_history", "rho_hist"))
    if rho_history is not None:
        rho = np.asarray(rho_history, dtype=np.float64).ravel()
        if rho.size and np.all(np.isfinite(rho)):
            details["n_rho_history"] = int(rho.size)
            details["rho_final"] = float(rho[-1])

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "details": details,
    }
