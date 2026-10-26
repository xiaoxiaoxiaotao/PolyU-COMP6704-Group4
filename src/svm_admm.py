"""WP3: cost-sensitive linear SVMs with ADMM and Pegasos-warmed ADMM.

The objective is ``||w||²/2 + sum(C_i * hinge(y_i * (X_i w + b)))``.
The intercept is unregularized. See docs/WP3/WP3_ADMM.md for the derivation,
scaled-dual sign convention, stopping certificate, and warm-start scaling.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from src.solver_api import BaseBinarySVM, validate_training_data


def _hinge_prox(a: np.ndarray, penalties: np.ndarray) -> np.ndarray:
    """Proximal map of sum(penalties_i * max(0, 1-z_i))."""
    return np.where(a >= 1.0, a, np.minimum(1.0, a + penalties))


def _balanced_dual(raw: np.ndarray, y: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Feasible SVM dual variables, with exact box and bias constraints.

    Shrink the class with greater total mass; shrinking preserves the box
    constraints. This is a feasible certificate, not an exact projection.
    """
    dual = np.clip(raw, 0.0, C)
    positive = y > 0
    pos_mass = float(np.sum(dual[positive]))
    neg_mass = float(np.sum(dual[~positive]))
    if pos_mass > neg_mass:
        dual[positive] *= neg_mass / pos_mass
    elif neg_mass > pos_mass:
        dual[~positive] *= pos_mass / neg_mass
    return dual


def _positive_finite(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive")


def _nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


class ADMMSVM(BaseBinarySVM):
    """Cold-start fixed-rho ADMM by default; options support WP3 ablations."""

    def fit(
        self, X, y, C_per_sample, *, seed=0, tol=1e-4,
        time_budget_sec=5400.0, rho0=1.0, adaptive_rho=False,
        alpha=1.0, max_iter=10000,
    ) -> "ADMMSVM":
        return self._fit(
            X, y, C_per_sample, seed=seed, tol=tol,
            time_budget_sec=time_budget_sec, rho0=rho0,
            adaptive_rho=adaptive_rho, alpha=alpha, max_iter=max_iter,
            K=0, warm_start_batch_size=256,
        )

    def _fit(
        self, X, y, C_per_sample, *, seed, tol, time_budget_sec,
        rho0, adaptive_rho, alpha, max_iter, K, warm_start_batch_size,
    ):
        start = time.perf_counter()
        X, y, C = validate_training_data(X, y, C_per_sample)
        for name, value in (("tol", tol), ("time_budget_sec", time_budget_sec), ("rho0", rho0)):
            _positive_finite(name, value)
        if not np.isfinite(alpha) or not 0.0 < alpha < 2.0:
            raise ValueError("alpha must lie strictly between 0 and 2")
        for name, value in (("max_iter", max_iter), ("K", K), ("warm_start_batch_size", warm_start_batch_size)):
            _nonnegative_integer(name, value)
        if warm_start_batch_size == 0:
            raise ValueError("warm_start_batch_size must be positive")
        if not isinstance(adaptive_rho, (bool, np.bool_)):
            raise ValueError("adaptive_rho must be a boolean")

        n, d = X.shape
        # The harness rejects even a small budget overrun. Leave time for the
        # last atomic BLAS operation and copying the selected iterate out.
        deadline = start + time_budget_sec - min(0.25, 0.05 * time_budget_sec)
        rho = float(rho0)
        theta = np.zeros(d + 1)
        self.n_iter = 0
        self.converged = False
        self.stop_reason = "max_iter"
        self.warm_start_epochs = int(K)
        self.warm_start_epochs_completed = 0
        self.warm_start_updates = 0
        self.factorization_count = 0
        self.z = None
        self.scaled_dual = None
        self.rho = rho
        self.objective_history = []
        self.time_history = []
        self.stage_history = []
        self.rho_history = []
        self.residual_history = []
        self.dual_objective_history = []
        self.relative_gap_history = []
        best_theta = theta.copy()
        best_objective = float(np.sum(C))
        self.dual_objective = 0.0

        def checkpoint(margins, stage):
            nonlocal best_theta, best_objective
            objective = 0.5 * float(theta[:-1] @ theta[:-1]) + float(C @ np.maximum(0.0, 1.0 - margins))
            if objective < best_objective:
                best_objective, best_theta = objective, theta.copy()
            self.objective_history.append(objective)
            self.time_history.append(time.perf_counter() - start)
            self.stage_history.append(stage)
            self.rho_history.append(rho)
            return objective

        checkpoint(np.zeros(n), "initial")

        # This private warm start keeps WP3 usable before WP1's public solver
        # arrives. It uses the same supplied C_i without applying costs twice.
        warm_start_begin = time.perf_counter()
        if K:
            rng = np.random.default_rng(seed)
            total_cost = float(np.sum(C))
            lam = 1.0 / total_cost
            weights = C / float(np.mean(C))
            radius = np.sqrt(2.0 * total_cost)
            for _ in range(K):
                if time.perf_counter() >= deadline:
                    break
                order = rng.permutation(n)
                completed = True
                if warm_start_batch_size == 1:
                    # Avoid allocating a matrix, mask, and index array for
                    # every individual stochastic update on the full dataset.
                    for index in order:
                        if time.perf_counter() >= deadline:
                            completed = False
                            break
                        active = y[index] * (X[index] @ theta[:-1] + theta[-1]) < 1.0
                        self.warm_start_updates += 1
                        step = self.warm_start_updates
                        theta[:-1] *= 1.0 - 1.0 / step
                        if active:
                            coefficient = n * C[index] * y[index] / step
                            theta[:-1] += coefficient * X[index]
                            theta[-1] += coefficient
                        norm = np.linalg.norm(theta[:-1])
                        if norm > radius:
                            theta[:-1] *= radius / norm
                for offset in range(0, n, warm_start_batch_size):
                    if warm_start_batch_size == 1:
                        break
                    if time.perf_counter() >= deadline:
                        completed = False
                        break
                    indices = order[offset:offset + warm_start_batch_size]
                    xb, yb = X[indices], y[indices]
                    active = yb * (xb @ theta[:-1] + theta[-1]) < 1.0
                    self.warm_start_updates += 1
                    eta = 1.0 / (lam * self.warm_start_updates)
                    coefficients = weights[indices] * yb * active / len(indices)
                    theta[:-1] *= 1.0 - eta * lam
                    theta[:-1] += eta * (xb.T @ coefficients)
                    theta[-1] += eta * float(np.sum(coefficients))
                    norm = np.linalg.norm(theta[:-1])
                    if norm > radius:
                        theta[:-1] *= radius / norm
                self.warm_start_epochs_completed += int(completed)
                checkpoint(y * (X @ theta[:-1] + theta[-1]), "pegasos")
                if not completed:
                    break
            # Retain the best observed feasible primal warm iterate, including
            # zero: stochastic epochs can overshoot severely at large n*C.
            theta = best_theta.copy()
        self.warm_start_time_sec = time.perf_counter() - warm_start_begin

        # Signed augmented design: A theta = y * (Xw + b). A.T A is
        # [X,1].T [X,1]; only a (d+1)-square matrix is factorized.
        A = None
        if max_iter and time.perf_counter() < deadline:
            A = np.empty((n, d + 1), dtype=np.float64)
            A[:, :-1] = X * y[:, None]
            A[:, -1] = y
            gram = A.T @ A
            regularizer = np.ones(d + 1)
            regularizer[-1] = 0.0
            margins = A @ theta
            z = margins.copy()
            q = np.zeros(n)  # scaled dual for z - A theta = 0
            factor = None
            factor_rho = None
            best_dual = 0.0

            for iteration in range(1, max_iter + 1):
                if time.perf_counter() >= deadline:
                    self.stop_reason = "time_budget"
                    break
                if factor_rho != rho:
                    system = gram.copy()
                    system.flat[::d + 2] += regularizer / rho
                    factor = cho_factor(system, lower=True, check_finite=False)
                    factor_rho = rho
                    self.factorization_count += 1
                theta = cho_solve(factor, A.T @ (z + q), check_finite=False)
                margins = A @ theta
                z_old = z
                relaxed = alpha * margins + (1.0 - alpha) * z_old
                z = _hinge_prox(relaxed - q, C / rho)
                q += z - relaxed

                primal_residual = float(np.linalg.norm(z - margins))
                dual_residual = float(np.linalg.norm(rho * (A.T @ (z - z_old))))
                eps_primal = np.sqrt(n) * tol + tol * max(np.linalg.norm(margins), np.linalg.norm(z))
                eps_dual = np.sqrt(d + 1) * tol + tol * np.linalg.norm(rho * (A.T @ q))
                objective = checkpoint(margins, "admm")
                dual = _balanced_dual(rho * q, y, C)
                dual_w = A[:, :-1].T @ dual
                dual_objective = float(np.sum(dual) - 0.5 * (dual_w @ dual_w))
                best_dual = max(best_dual, dual_objective)
                gap = max(0.0, best_objective - best_dual) / max(abs(best_dual), 1e-12)
                self.dual_objective_history.append(best_dual)
                self.relative_gap_history.append(gap)
                self.residual_history.append({
                    "iteration": iteration, "primal": primal_residual,
                    "dual": dual_residual, "eps_primal": float(eps_primal),
                    "eps_dual": float(eps_dual), "rho": rho,
                })
                self.n_iter = iteration
                self.dual_objective = best_dual
                # Stop on both feasibility residuals AND a certified gap;
                # successive objective stagnation alone is insufficient.
                if (primal_residual <= eps_primal and dual_residual <= eps_dual
                        and gap <= tol and time.perf_counter() < deadline):
                    self.converged = True
                    self.stop_reason = "converged"
                    break
                if adaptive_rho:
                    new_rho = rho
                    if primal_residual > 10.0 * dual_residual:
                        new_rho = min(2.0 * rho, 1e8)
                    elif dual_residual > 10.0 * primal_residual:
                        new_rho = max(rho / 2.0, 1e-8)
                    if new_rho != rho:
                        q *= rho / new_rho  # preserve unscaled dual rho*q
                        rho = new_rho
            self.z = z
            self.scaled_dual = q
            self.rho = rho

        if time.perf_counter() >= deadline and not self.converged:
            self.stop_reason = "time_budget"
        self.w = best_theta[:-1].copy()
        self.b = float(best_theta[-1])
        self.objective = best_objective
        self.relative_gap = max(0.0, best_objective - self.dual_objective) / max(abs(self.dual_objective), 1e-12)
        # Selecting an earlier best iterate is a real final checkpoint. Keep
        # history[-1] consistent with the returned w,b, including budget exits.
        if self.objective_history[-1] != best_objective:
            self.objective_history.append(best_objective)
            self.time_history.append(time.perf_counter() - start)
            self.stage_history.append("best_iterate")
            self.rho_history.append(rho)
        self.fit_time_sec = time.perf_counter() - start
        return self

    def decision_function(self, X):
        if not hasattr(self, "w"):
            raise RuntimeError("fit must be called before decision_function")
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.w.size:
            raise ValueError("X must be a 2-D array with the fitted feature width")
        if not np.all(np.isfinite(values)):
            raise ValueError("X contains non-finite values")
        return values @ self.w + self.b


class PAADMMSVM(ADMMSVM):
    """Weighted Pegasos warm start followed by adaptive, relaxed ADMM."""

    def fit(
        self, X, y, C_per_sample, *, seed=0, tol=1e-4,
        time_budget_sec=5400.0, K=10, rho0=1.0, adaptive_rho=True,
        alpha=1.6, max_iter=10000, warm_start_batch_size=1,
    ) -> "PAADMMSVM":
        return self._fit(
            X, y, C_per_sample, seed=seed, tol=tol,
            time_budget_sec=time_budget_sec, rho0=rho0,
            adaptive_rho=adaptive_rho, alpha=alpha, max_iter=max_iter,
            K=K, warm_start_batch_size=warm_start_batch_size,
        )
