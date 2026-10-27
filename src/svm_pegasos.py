"""Weighted Pegasos: primal stochastic subgradient training of the team's
cost-sensitive linear SVM.

The objective is the harness convention, an unnormalized sum with the
per-sample penalties supplied by the evaluator:

    P(w, b) = 0.5 * ||w||^2 + sum_i C_i * max(0, 1 - y_i * (x_i^T w + b))

``C_per_sample`` is used as given: class costs are already folded into it, so
nothing here reweights them. Dividing the objective by ``S = sum_i C_i`` gives

    P / S = (lambda / 2) * ||w||^2 + (1 / n) * sum_i a_i * hinge_i,
    lambda = 1 / S,   a_i = C_i / mean(C)

which is the form Pegasos minimizes, so ``eta_t = 1 / (lambda * t)`` with ``t``
counting parameter updates and ``a_i`` entering each sampled subgradient. The
radius ``sqrt(2 * S)`` bounds ``||w||`` because ``P(0, 0) = S``. The schedule is
started from ``t0 = 1 / lambda`` instead of ``t = 1``: the textbook first step
``1 / lambda = S`` is of order ``n`` and destroys the unregularized intercept,
and ``1 / (lambda * (t0 + t))`` still converges to the textbook rate.

Convergence is never claimed: Pegasos is a first-order method and exposes no
primal-dual gap, so ``converged`` stays ``False`` and ``stop_reason`` records
whether the epoch budget or the wall-clock budget ended the run. Epoch
stagnation is deliberately not treated as convergence (the averaged iterate
keeps improving well past any fixed relative change), and the returned model is
the lowest measured averaged checkpoint, with the final history entry re-timed
to match it.
"""

from __future__ import annotations

import time

import numpy as np

from src.solver_api import BaseBinarySVM, validate_training_data


def _positive_finite(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive")


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class Pegasos(BaseBinarySVM):
    """Weighted Pegasos with seeded shuffling and epoch-level checkpoints."""

    def fit(
        self, X, y, C_per_sample, *, seed=0, tol=1e-4, time_budget_sec=5400.0,
        epochs=100, batch_size=1, log_every=1,
    ) -> "Pegasos":
        start = time.perf_counter()
        X, y, C = validate_training_data(X, y, C_per_sample)
        _positive_finite("tol", tol)
        _positive_finite("time_budget_sec", time_budget_sec)
        for name, value in (("epochs", epochs), ("batch_size", batch_size),
                            ("log_every", log_every)):
            _positive_integer(name, value)

        n, d = X.shape
        # The harness rejects any measured overrun, so keep a margin for the
        # in-flight vectorized update and the final checkpoint pass.
        deadline = start + time_budget_sec - min(0.25, 0.05 * time_budget_sec)
        total_cost = float(np.sum(C))
        lam = 1.0 / total_cost
        weights = C / float(np.mean(C))
        radius = np.sqrt(2.0 * total_cost)
        # eta_1 = 1/lambda = sum(C) is of order n, which blows the unregularized
        # intercept up on the first update, so the schedule starts from
        # tau = t0 + 1 (t0 = 1/lambda) and only approaches 1/(lambda * t) later.
        t0 = 1.0 / lam

        w = np.zeros(d, dtype=np.float64)
        b = 0.0
        w_sum = np.zeros(d, dtype=np.float64)
        b_sum = 0.0
        steps = 0
        self.w = w.copy()
        self.b = 0.0
        self.n_iter = 0
        self.epochs_completed = 0
        self.converged = False
        self.stop_reason = "epochs"
        self.optimality_certificate = "none (first-order method, no duality gap)"
        self.objective_history = []
        self.time_history = []
        self.checkpoint_epochs = []
        self.lambda_ = lam
        self.radius_ = radius

        def averaged():
            if steps == 0:
                return w.copy(), b
            return w_sum / steps, b_sum / steps

        def objective_of(w_values, b_value):
            margins = y * (X @ w_values + b_value)
            return 0.5 * float(w_values @ w_values) + float(C @ np.maximum(0.0, 1.0 - margins))

        def checkpoint(epoch: int) -> float:
            average_w, average_b = averaged()
            objective = objective_of(average_w, average_b)
            self.objective_history.append(objective)
            self.time_history.append(time.perf_counter() - start)
            self.checkpoint_epochs.append(epoch)
            return objective

        objective = checkpoint(0)
        best = (objective, np.zeros(d), 0.0)
        rng = np.random.default_rng(seed)

        for epoch in range(1, int(epochs) + 1):
            if time.perf_counter() >= deadline:
                self.stop_reason = "time_budget"
                break
            order = rng.permutation(n)
            completed = True
            for offset in range(0, n, batch_size):
                if time.perf_counter() >= deadline:
                    completed = False
                    break
                index = order[offset:offset + batch_size]
                xb, yb = X[index], y[index]
                active = (yb * (xb @ w + b) < 1.0).astype(np.float64)
                coefficients = (weights[index] * yb * active) / len(index)
                eta = 1.0 / (lam * (t0 + steps + 1))
                w *= 1.0 - eta * lam
                w += eta * (xb.T @ coefficients)
                b += eta * float(np.sum(coefficients))
                norm = float(np.linalg.norm(w))
                if norm > radius:
                    w *= radius / norm
                steps += 1
                w_sum += w
                b_sum += b
            self.n_iter = steps
            if epoch % log_every == 0 or not completed:
                objective = checkpoint(epoch)
                average_w, average_b = averaged()
                if objective < best[0]:
                    best = (objective, average_w.copy(), average_b)
            if epoch > self.epochs_completed:
                self.epochs_completed = epoch
            if not completed:
                self.stop_reason = "time_budget"
                break

        self.w = best[1].copy()
        self.b = float(best[2])
        self.objective = float(best[0])
        if self.objective_history[-1] != self.objective:
            # The returned model is an earlier checkpoint; record it at its
            # actual return time so history[-1] matches w, b.
            self.objective_history.append(self.objective)
            self.time_history.append(time.perf_counter() - start)
            self.checkpoint_epochs.append(self.epochs_completed)
        self.fit_time_sec = time.perf_counter() - start
        return self

    def decision_function(self, X):
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.w.size:
            raise ValueError("X must be a 2-D array with the fitted feature width")
        if not np.all(np.isfinite(values)):
            raise ValueError("X contains non-finite values")
        return values @ self.w + self.b
