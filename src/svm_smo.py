"""Cost-sensitive binary SVM trained by sequential minimal optimization.

The objective is the *unnormalized* primal
    0.5 * ||w||^2 + sum(C_i * max(0, 1 - y_i * f(x_i))).
C_per_sample already contains the team's cost weights. This module never
renormalizes it. The RBF version uses the corresponding RKHS norm.

The implementation solves the dual using two-coordinate feasible steps and a
second-order working-set rule. It does not call a third-party SVM solver.
References: Platt (1998), MSR-TR-98-14; Fan, Chen and Lin (2005), JMLR 6.
"""

from collections import OrderedDict
from numbers import Integral, Real
import time

import numpy as np


class _KernelColumns:
    """An LRU cache of kernel columns, bounded independently of sample count.

    cache_size_mb bounds retained columns, not all process memory. Two active
    columns and vector temporaries may coexist outside the retained cache.
    No n-by-n Gram matrix is constructed.
    """

    def __init__(self, X, kernel, gamma, cache_size_mb):
        self.X = X
        self.kernel = kernel
        self.gamma = gamma
        self.diagonal = (
            np.einsum("ij,ij->i", X, X)
            if kernel == "linear" else np.ones(len(X), dtype=np.float64)
        )
        self.squared_norms = np.einsum("ij,ij->i", X, X)
        self.capacity = int(cache_size_mb * 1024 ** 2 // (8 * len(X)))
        self.columns = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, index):
        if index in self.columns:
            self.hits += 1
            self.columns.move_to_end(index)
            return self.columns[index]
        self.misses += 1
        column = self.X @ self.X[index]
        if self.kernel == "rbf":
            distance = self.squared_norms + self.squared_norms[index] - 2.0 * column
            np.maximum(distance, 0.0, out=distance)
            column = np.exp(-self.gamma * distance)
            column[index] = 1.0
        if not np.all(np.isfinite(column)):
            raise FloatingPointError("Kernel computation produced NaN or Inf; rescale X.")
        if self.capacity:
            while len(self.columns) >= self.capacity:
                self.columns.popitem(last=False)
            self.columns[index] = column
        return column


class SMO:
    """Binary linear/RBF SVM with unequal per-sample dual box constraints.

    All configuration belongs to fit so SMO() satisfies the WP4 registry.
    Working-set selection and tie-breaking are deterministic. seed is accepted
    and recorded for the common protocol, but no randomness is required.
    """

    def fit(
        self, X, y, C_per_sample, *, seed=0, tol=1e-4, gap_tol=None,
        time_budget_sec=5400.0, max_iter=100000, kernel="linear",
        gamma="scale", cache_size_mb=64.0, log_every=100,
        working_set="second_order", **kwargs,
    ):
        """Fit on training data only and return self.

        tol bounds the dual KKT working-set gap. Convergence also requires
        dual feasibility and (P-D)/max(1, abs(P), abs(D)) <= gap_tol, where
        gap_tol defaults to tol. This certificate is distinct from the
        evaluator's post-hoc relative objective gap to a reference method.
        The additional gap requirement can require more updates at large C.

        max_iter counts accepted pair updates. A zero iteration/time budget is
        valid and returns the initial feasible iterate with converged=False.
        On time/iteration exhaustion, return the lowest-primal feasible
        checkpoint actually measured. n_iter still counts all work performed;
        returned_checkpoint_iteration identifies the returned state. A real
        terminal checkpoint records that selection at its actual return time.
        The dual objective is nondecreasing during pair updates, but restoring
        a better primal checkpoint can lower the terminal dual objective.
        """
        started = time.perf_counter()
        self._is_fitted = False
        if kwargs:
            raise TypeError("Unsupported SMO fit parameter(s): " + ", ".join(sorted(kwargs)))
        X, y, C = self._validate_training(X, y, C_per_sample)
        if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if not isinstance(max_iter, Integral) or isinstance(max_iter, bool) or max_iter < 0:
            raise ValueError("max_iter must be a nonnegative integer.")
        if not isinstance(log_every, Integral) or isinstance(log_every, bool) or log_every < 1:
            raise ValueError("log_every must be a positive integer.")
        self._positive_scalar(tol, "tol")
        if gap_tol is None:
            gap_tol = tol
        self._positive_scalar(gap_tol, "gap_tol")
        self._nonnegative_scalar(cache_size_mb, "cache_size_mb")
        if time_budget_sec is None:
            budget = np.inf
        else:
            self._nonnegative_scalar(time_budget_sec, "time_budget_sec")
            budget = float(time_budget_sec)
        if kernel not in ("linear", "rbf"):
            raise ValueError("kernel must be 'linear' or 'rbf'.")
        if working_set != "second_order":
            raise ValueError("working_set must be 'second_order'.")
        gamma_value = self._resolve_gamma(gamma, X)

        self.X_train_ = X
        self.y_train_ = y
        self.C_per_sample_ = C
        self.n_features_in_ = X.shape[1]
        self.kernel = kernel
        self.gamma = gamma
        self.gamma_ = gamma_value
        self.seed = int(seed)
        self.tol = float(tol)
        self.gap_tol = float(gap_tol)
        self.alphas = np.zeros(len(y), dtype=np.float64)
        self.w = np.zeros(X.shape[1], dtype=np.float64) if kernel == "linear" else None
        self.b = 0.0
        self.n_iter = 0
        self.converged = False
        self.stop_reason = "max_iter"
        self.objective_history = []
        self.time_history = []
        self.dual_objective_history = []
        self.duality_gap_history = []
        self.relative_duality_gap_history = []
        self.kkt_violation_history = []
        self.history_iters = []
        self.returned_best_checkpoint = False
        self.returned_checkpoint_index = None
        self.returned_checkpoint_iteration = None
        best_checkpoint = None
        self._bound_eps = 16.0 * np.finfo(np.float64).eps * np.maximum(C, 1e-12)
        gradient = -np.ones(len(y), dtype=np.float64)  # Q alpha - 1
        kernels = _KernelColumns(X, kernel, gamma_value, float(cache_size_mb))
        if not np.all(np.isfinite(kernels.diagonal)):
            raise FloatingPointError("Kernel diagonal overflowed; rescale X.")

        def checkpoint():
            nonlocal gradient, best_checkpoint
            # Refresh linear scores in O(nd) at checkpoints to avoid drift.
            if kernel == "linear":
                self.w = X.T @ (self.alphas * y)
                scores_without_bias = X @ self.w
                gradient = y * scores_without_bias - 1.0
            else:
                scores_without_bias = y * (gradient + 1.0)
            self.b = self._bias(gradient)
            margins = y * (scores_without_bias + self.b)
            if kernel == "linear":
                norm_squared = float(self.w @ self.w)
            else:
                norm_squared = float((self.alphas * y) @ scores_without_bias)
                roundoff = 128.0 * np.finfo(float).eps * max(
                    1.0, float(np.abs(self.alphas * scores_without_bias).sum())
                )
                if norm_squared < -roundoff:
                    raise FloatingPointError("The RKHS norm became negative beyond roundoff.")
                norm_squared = max(0.0, norm_squared)
            primal = 0.5 * norm_squared + float(C @ np.maximum(0.0, 1.0 - margins))
            dual = float(self.alphas.sum()) - 0.5 * norm_squared
            gap = primal - dual
            roundoff = 256.0 * np.finfo(float).eps * max(1.0, abs(primal), abs(dual))
            if gap < -roundoff:
                raise FloatingPointError("Negative primal-dual gap beyond roundoff.")
            self.objective = float(primal)
            self.dual_objective = float(dual)
            self.duality_gap = float(max(0.0, gap))
            self.relative_duality_gap = self.duality_gap / max(1.0, abs(primal), abs(dual))
            self.kkt_violation = self._kkt_gap(gradient)
            self.box_violation = float(max(
                0.0, -float(self.alphas.min()), float(np.max(self.alphas - C))
            ))
            self.equality_violation = float(abs(y @ self.alphas))
            numerical_limit = 256.0 * np.finfo(float).eps * max(1.0, float(self.alphas.sum()))
            self.dual_feasible = (
                self.box_violation <= numerical_limit
                and self.equality_violation <= numerical_limit
            )
            if not self.dual_feasible:
                raise FloatingPointError("SMO lost dual feasibility beyond roundoff.")
            if not np.all(np.isfinite([primal, dual, self.b, self.kkt_violation])):
                raise FloatingPointError("Nonfinite objective or stopping diagnostic.")
            self.objective_history.append(self.objective)
            self.dual_objective_history.append(self.dual_objective)
            self.duality_gap_history.append(self.duality_gap)
            self.relative_duality_gap_history.append(self.relative_duality_gap)
            self.kkt_violation_history.append(self.kkt_violation)
            self.history_iters.append(self.n_iter)
            self.time_history.append(float(time.perf_counter() - started))
            if best_checkpoint is None or primal < best_checkpoint["objective"]:
                best_checkpoint = {
                    "objective": self.objective,
                    "alphas": self.alphas.copy(),
                    "gradient": gradient.copy(),
                    "w": None if self.w is None else self.w.copy(),
                    "b": self.b,
                    "index": len(self.objective_history) - 1,
                    "iteration": self.n_iter,
                }

        def certificate_met():
            return (
                self.dual_feasible
                and self.kkt_violation <= self.tol
                and self.relative_duality_gap <= self.gap_tol
            )

        def refresh_gradient():
            nonlocal gradient
            if kernel == "linear":
                self.w = X.T @ (self.alphas * y)
                gradient = y * (X @ self.w) - 1.0
                return True
            fresh_scores = np.zeros(len(y), dtype=np.float64)
            for index in np.flatnonzero(self.alphas):
                if time.perf_counter() - started >= budget:
                    return False
                fresh_scores += self.alphas[index] * y[index] * kernels.get(int(index))
            gradient = y * fresh_scores - 1.0
            return True

        checkpoint()
        while self.n_iter < max_iter:
            if time.perf_counter() - started >= budget:
                self.stop_reason = "time_budget"
                break
            up, low = self._working_sets()
            z = -y * gradient
            up_indices = np.flatnonzero(up)
            low_indices = np.flatnonzero(low)
            if not len(up_indices) or not len(low_indices):
                # Both classes and strictly positive C normally make these sets
                # nonempty. Do not silently claim convergence if that invariant fails.
                self.stop_reason = "numerical_stall"
                break
            i = int(up_indices[np.argmax(z[up_indices])])
            gap = float(z[i] - np.min(z[low_indices]))
            if gap <= tol:
                checkpoint()
                if certificate_met():
                    # Certify against the actual model, not an accumulated
                    # kernel gradient. Avoid this costly RBF refresh while the
                    # inexpensive primal-dual check already fails.
                    if not refresh_gradient():
                        self.stop_reason = "time_budget"
                        break
                    checkpoint()
                    if certificate_met() and time.perf_counter() - started < budget:
                        self.converged = True
                        self.stop_reason = "converged"
                        break
                if time.perf_counter() - started >= budget:
                    self.stop_reason = "time_budget"
                    break
                # A small KKT gap can coexist with a large weighted primal
                # gap. Continue with an actual feasible pair update, using the
                # refreshed gradient, rather than looping without progress.
                up, low = self._working_sets()
                up_indices = np.flatnonzero(up)
                low_indices = np.flatnonzero(low)
                if not len(up_indices) or not len(low_indices):
                    self.stop_reason = "numerical_stall"
                    break
                z = -y * gradient
                i = int(up_indices[np.argmax(z[up_indices])])

            column_i = kernels.get(i)
            candidates = low_indices[z[i] > z[low_indices]]
            if not len(candidates):
                self.stop_reason = "numerical_stall"
                break
            pair_gaps = z[i] - z[candidates]
            curvature = (
                kernels.diagonal[i] + kernels.diagonal[candidates]
                - 2.0 * column_i[candidates]
            )
            # A valid linear/RBF Gram matrix is PSD. Tiny negative values here
            # arise from cancellation for (nearly) duplicate points.
            curvature = np.maximum(curvature, 0.0)
            cap_i = C[i] - self.alphas[i] if y[i] > 0 else self.alphas[i]
            cap_j = np.where(
                y[candidates] > 0, self.alphas[candidates],
                C[candidates] - self.alphas[candidates],
            )
            caps = np.minimum(cap_i, cap_j)
            unconstrained = np.full(len(candidates), np.inf)
            np.divide(pair_gaps, curvature, out=unconstrained, where=curvature > 0)
            steps = np.minimum(caps, unconstrained)
            # Choose the feasible pair with the greatest quadratic improvement.
            # For zero curvature the optimum is the feasible endpoint.
            improvements = pair_gaps * steps - 0.5 * curvature * steps ** 2
            chosen = int(np.argmax(improvements))
            j = int(candidates[chosen])
            step = float(steps[chosen])
            old_i, old_j = float(self.alphas[i]), float(self.alphas[j])
            if step <= 0 or not np.isfinite(step):
                self.stop_reason = "numerical_stall"
                break

            # Clip one coordinate, derive the other from the equality constraint,
            # and handle a possible endpoint roundoff in the other coordinate.
            new_i = float(np.clip(old_i + y[i] * step, 0.0, C[i]))
            step = (new_i - old_i) / y[i]
            new_j = old_j - y[j] * step
            if new_j < 0.0 or new_j > C[j]:
                new_j = float(np.clip(new_j, 0.0, C[j]))
                step = (old_j - new_j) / y[j]
                new_i = old_i + y[i] * step
            delta_i, delta_j = new_i - old_i, new_j - old_j
            if delta_i == 0.0 or delta_j == 0.0:
                self.stop_reason = "numerical_stall"
                break
            column_j = kernels.get(j)
            self.alphas[i], self.alphas[j] = new_i, new_j
            beta_i, beta_j = delta_i * y[i], delta_j * y[j]
            gradient += y * (beta_i * column_i + beta_j * column_j)
            if kernel == "linear":
                self.w += beta_i * X[i] + beta_j * X[j]
            self.n_iter += 1
            if self.n_iter % log_every == 0:
                checkpoint()
        else:
            self.stop_reason = "max_iter"

        # Always record a real terminal checkpoint, even when no update occurred.
        # Duplicate iterate values are intentional; timestamps are measured, not
        # interpolated, and do not imply an unobserved tolerance crossing.
        checkpoint()
        if not self.converged and self.stop_reason == "max_iter":
            if time.perf_counter() - started >= budget:
                self.stop_reason = "time_budget"
            elif certificate_met():
                # The last allowed pair may already solve the problem. RBF
                # still needs the same fresh-gradient certificate as an early
                # convergence exit, and that refresh must fit in the budget.
                refreshed = kernel == "linear" or refresh_gradient()
                if kernel == "rbf" and refreshed:
                    checkpoint()
                if not refreshed or time.perf_counter() - started >= budget:
                    self.stop_reason = "time_budget"
                elif certificate_met():
                    self.converged = True
                    self.stop_reason = "converged"
        if self.stop_reason in ("max_iter", "time_budget"):
            # Snapshot all coupled state; restoring only w or b would detach
            # predictions from the returned dual variables and certificate.
            selected = best_checkpoint
            self.alphas = selected["alphas"].copy()
            gradient = selected["gradient"].copy()
            self.w = None if selected["w"] is None else selected["w"].copy()
            self.b = selected["b"]
            self.returned_best_checkpoint = True
            self.returned_checkpoint_index = selected["index"]
            self.returned_checkpoint_iteration = selected["iteration"]
            checkpoint()
        else:
            self.returned_checkpoint_index = len(self.objective_history) - 1
            self.returned_checkpoint_iteration = self.n_iter
        self.support_ = np.flatnonzero(self.alphas > self._bound_eps)
        self.support_vectors_ = X[self.support_].copy()
        self.dual_coef_ = (self.alphas[self.support_] * y[self.support_]).copy()
        self.cache_hits = kernels.hits
        self.cache_misses = kernels.misses
        self.cache_capacity_columns = kernels.capacity
        self.cache_bytes = len(kernels.columns) * len(y) * 8
        self.fit_time_sec = float(time.perf_counter() - started)
        self._is_fitted = True
        return self

    def decision_function(self, X):
        """Return finite raw scores of shape (n,), with positive meaning fraud."""
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError("SMO must be fitted before prediction.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X must be two-dimensional with the fitted feature count.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")
        if self.kernel == "linear":
            scores = X @ self.w + self.b
        elif not len(self.support_):
            scores = np.full(len(X), self.b, dtype=np.float64)
        else:
            scores = np.empty(len(X), dtype=np.float64)
            # Bound the transient prediction kernel block to roughly 8 MiB.
            batch = max(1, min(1024, 1024 ** 2 // len(self.support_)))
            support_norm = np.einsum("ij,ij->i", self.support_vectors_, self.support_vectors_)
            for start in range(0, len(X), batch):
                block = X[start:start + batch]
                distance = (
                    np.einsum("ij,ij->i", block, block)[:, None]
                    + support_norm[None, :] - 2.0 * block @ self.support_vectors_.T
                )
                np.maximum(distance, 0.0, out=distance)
                np.multiply(distance, -self.gamma_, out=distance)
                np.exp(distance, out=distance)
                scores[start:start + batch] = distance @ self.dual_coef_ + self.b
        if not np.all(np.isfinite(scores)):
            raise FloatingPointError("Prediction produced NaN or Inf; rescale X.")
        return np.asarray(scores, dtype=np.float64)

    def predict(self, X):
        """Return labels in {-1, +1}; a zero score maps to +1."""
        return np.where(self.decision_function(X) >= 0.0, 1, -1).astype(np.int8)

    def _working_sets(self):
        a, C, y, eps = self.alphas, self.C_per_sample_, self.y_train_, self._bound_eps
        up = ((y > 0) & (a < C - eps)) | ((y < 0) & (a > eps))
        low = ((y > 0) & (a > eps)) | ((y < 0) & (a < C - eps))
        return up, low

    def _kkt_gap(self, gradient):
        up, low = self._working_sets()
        if not np.any(up) or not np.any(low):
            return float("inf")
        z = -self.y_train_ * gradient
        return float(max(0.0, float(z[up].max() - z[low].min())))

    def _bias(self, gradient):
        z = -self.y_train_ * gradient  # y_i - f_without_bias(x_i)
        free = (self.alphas > self._bound_eps) & (
            self.alphas < self.C_per_sample_ - self._bound_eps
        )
        if np.any(free):
            return float(np.mean(z[free]))
        up, low = self._working_sets()
        if np.any(up) and np.any(low):
            return float(0.5 * (np.max(z[up]) + np.min(z[low])))
        raise FloatingPointError("Cannot recover bias from an empty KKT working set.")

    @staticmethod
    def _validate_training(X, y, C_per_sample):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        C = np.asarray(C_per_sample, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
            raise ValueError("X must have shape (n, d), with n >= 2 and d >= 1.")
        if y.ndim != 1 or C.ndim != 1 or len(y) != len(X) or len(C) != len(X):
            raise ValueError("y and C_per_sample must be one-dimensional and match X rows.")
        if not all(np.all(np.isfinite(v)) for v in (X, y, C)):
            raise ValueError("Training arrays must contain only finite values.")
        if not np.all(np.isin(y, (-1.0, 1.0))) or np.unique(y).size != 2:
            raise ValueError("y must contain both classes -1 and +1.")
        if np.any(C <= 0):
            raise ValueError("C_per_sample must be strictly positive.")
        return np.array(X, order="C", copy=True), y.copy(), C.copy()

    @staticmethod
    def _positive_scalar(value, name):
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number.")

    @staticmethod
    def _nonnegative_scalar(value, name):
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number.")

    @classmethod
    def _resolve_gamma(cls, gamma, X):
        if isinstance(gamma, str):
            if gamma == "auto":
                return 1.0 / X.shape[1]
            if gamma == "scale":
                variance = float(X.var())
                return 1.0 / (X.shape[1] * variance) if variance > 0 else 1.0
            raise ValueError("gamma must be 'scale', 'auto', or a positive number.")
        cls._positive_scalar(gamma, "gamma")
        return float(gamma)
