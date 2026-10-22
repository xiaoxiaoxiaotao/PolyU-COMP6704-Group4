# Solver Interface — WP2 (SMO) and WP3 (ADMM / PA-ADMM)

This is the contract between the evaluation harness (WP4) and the solver owners.
It follows the interface in the research proposal and `EXPERIMENTS.md`. **The data
pipeline and split files are frozen; do not regenerate or modify them without
coordinating with WP4.**

The full experiment literature is in `EXPERIMENTS.md`. This document is the
one-page implementation guide for solver owners.

---

## 1. Quick start

Implement the module/class pair expected by the registry:

| Registry key | Module | Class | Owner |
|---|---|---|---|
| `smo` | `src/svm_smo.py` | `SMO` | WP2 |
| `pegasos` | `src/svm_pegasos.py` | `Pegasos` | WP1 |
| `admm` | `src/svm_admm.py` | `ADMMSVM` | WP3 |
| `pa_admm` | `src/svm_admm.py` | `PAADMMSVM` | WP3 |
| `cvxpy` | `src/baseline_cvxpy.py` | `CVXPYLinearSVM` | WP1 |

Then, from the repository root:

```bash
# 1) contract check on synthetic data (fast, no raw data needed)
.venv/bin/python experiments/check_solver.py --method smo
.venv/bin/python experiments/check_solver.py --method admm
.venv/bin/python experiments/check_solver.py --method pa_admm

# 2) correctness run (10k subset; needs cvxpy too)
.venv/bin/python experiments/run_correctness.py --config configs/correctness.yaml --dry-run
.venv/bin/python experiments/run_correctness.py --config configs/correctness.yaml

# 3) fast matrix
.venv/bin/python experiments/run_main.py --config configs/main_fast.yaml --dry-run
.venv/bin/python experiments/run_main.py --config configs/main_fast.yaml
```

`check_solver.py` exits `0` only when the contract passes. Its JSON report is
written to `results/raw/solver_checks/{method}.json` (ignored by Git).

The repository venv is local and ignored. If it is missing:
`python3 -m venv --without-pip .venv` then bootstrap pip inside it, or install
`requirements.txt` with your own environment.

---

## 2. Data contract (frozen)

The harness passes NumPy arrays, not DataFrames:

| Object | Shape / dtype | Meaning |
|---|---|---|
| `X` | `(n_train, 30)` `float64` | `V1`…`V28` then `Amount`; `Time` excluded |
| `y` | `(n_train,)` `int8` | `-1` = legitimate, `+1` = fraud |
| `C_per_sample` | `(n_train,)` `float64` | per-sample penalty, strictly positive |
| `X_val`, `X_test` | `(n, 30)` `float64` | never used for fitting or threshold tuning |

Data facts (already verified and committed):

- Raw `data/raw/creditcard.csv`: 284,807 × 31, 492 frauds, SHA-256
  `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` (ignored by Git).
- Deduplicated: 283,726 rows, 473 frauds (0.167%).
- Committed splits: `data/splits/stratified_seed0..4.json`,
  `chronological_seed0.json`. Indices are positional into the deduplicated frame.
- `Amount` is standardized using the training-fold mean/std only.
- Validation/test thresholds are selected by the harness; a solver must not
  see validation/test labels during `fit`.
- Do **not** regenerate `data/splits/` or modify `data/raw/`.

---

## 3. The `fit` call

The harness constructs every solver with **no required constructor arguments**:

```python
model = SMO()                 # or ADMMSVM(), PAADMMSVM(), Pegasos(), CVXPYLinearSVM()
model.fit(X, y, C_per_sample, **kwargs)
```

Put hyperparameters in `fit`, not in the constructor. The harness passes:

| kwarg | Meaning |
|---|---|
| `seed` | deterministic RNG seed |
| `tol` | stopping tolerance (the harness also uses it as the relative objective-gap target for `time_to_tolerance`) |
| `time_budget_sec` | per-fit wall-clock budget; return the best iterate and set `converged=False` if you hit it |
| method-specific kwargs | from `configs/*.yaml`, e.g. `K`, `rho0`, `alpha`, `adaptive_rho`, `max_iter`, kernel options |

`fit` should return `self`. The harness filters kwargs by the signature; you may
declare explicit keyword arguments or `**kwargs`.

### `C_per_sample` is already cost-weighted

The harness computes

```text
C_i = C · w_{y_i},   w_+/w_- = c_FN/c_FP,
```

normalized so that the **sample mean of `w` is one**. The cost ratio is
preserved exactly. Values differ between capped/full training subsets because
each partition is normalized on its own rows.

- Use `C_per_sample` directly as the per-sample box bound/penalty.
- Do **not** apply class weights again inside `fit`.
- Do **not** renormalize by fraud prevalence.
- The proposal's class-average normalization
  (`w+ = 2c_FN/(c_FN+c_FP)`, `w- = 2c_FP/(c_FN+c_FP)`) differs from the
  sample-mean normalization by a global scalar only; that scalar is absorbed
  into the validation-selected `C`. Do not compensate for it inside the solver.

SMO dual box: `0 <= alpha_i <= C_per_sample[i]`.
ADMM z-update: use `C_per_sample[i] / rho`.

---

## 4. Required attributes after `fit`

The harness reads these. Keep them JSON-friendly (Python/NumPy scalars/arrays).

| Attribute | Type | Requirement |
|---|---|---|
| `w` | `(d,)` array, finite | final linear weights |
| `b` | float, finite | final intercept |
| `n_iter` | int | iterations/epochs actually run |
| `converged` | bool | stopping criterion met before the budget |
| `objective` | float | final **canonical** objective (recommended) |
| `objective_history` | 1-D list/array | objective at iterations/checkpoints, same convention |
| `time_history` | 1-D list/array | seconds since `fit` start, aligned with `objective_history` |
| `alphas` | `(n_train,)` array | SMO only: final dual variables |
| `rho_history` | 1-D list/array | ADMM/PA-ADMM only: penalty values over iterations |

Optional but useful: `residual_history`, `kkt_violation_history`, iterate
averages, support-vector indices. The harness ignores unknown attributes.
A non-iterative solver (for example CVXPY) can report a single-point
`objective_history=[objective]` and `time_history=[fit_time_sec]`; it will still
have a well-defined final objective, but its time-to-tolerance is not
comparable to iterative methods.

The objective convention is **unnormalized sum**:

```text
J(w,b) = 0.5 ||w||² + Σ_i C_i · max(0, 1 - y_i (wᵀx_i + b))
```

Do not divide by `n`. The history must use the same scale as the final
objective. The harness recomputes `J` from `w,b` and checks the final reported
value; a relative mismatch greater than `1e-3` marks the run as inconsistent and
suppresses its `time_to_tolerance` result rather than mixing scales.

### Histories and the primary metric

`time_to_tolerance` is computed after all methods have run:

```text
relative_gap_k = (J_k - J_ref) / |J_ref|
time_to_tolerance = t_k at the first k with relative_gap_k <= tol
```

where `J_ref` is CVXPY in correctness runs and the best observed canonical
objective in the same `(dataset, ratio, seed, n_train)` group otherwise.
Therefore:

- **`objective_history` is required** for the primary metric.
- **`time_history` is effectively required** for a non-null wall-clock time.
  If it is missing, the harness reports the final gap but no timing. Do not
  fabricate evenly spaced timestamps.
- Append at every iteration or at a logging checkpoint; align both arrays.
- The harness downsamples histories to `history_max_points` for JSON size.

The proposed metric originally also mentioned reaching a target **cost** per
example. The current v1 harness implements the objective-gap version because
per-iteration validation costs would require a separate solver callback and
would slow every fit. Final cost per example is reported at the
validation-selected threshold. This is a deliberate scope decision, documented
in `EXPERIMENTS.md`.

---

## 5. `decision_function` and `predict`

```python
def decision_function(self, X: np.ndarray) -> np.ndarray:
    """Return finite 1-D raw scores, shape (n,)."""

def predict(self, X: np.ndarray) -> np.ndarray:
    """Return -1/+1 labels, shape (n,)."""
```

- `decision_function(X_val)` and `decision_function(X_test)` are called after
  fitting; scores must be finite.
- The harness selects the threshold on validation scores and applies it to test;
  `predict` is still required by the protocol and checked by the contract
  checker. It may use the default zero-threshold implementation in
  `BaseBinarySVM`.
- Do not return `(n,1)`/`(1,n)` if avoidable; the checker accepts a singleton
  dimension but a 1-D vector is the contract.

Optional convenience base class (not required):

```python
from src.solver_api import BaseBinarySVM

class SMO(BaseBinarySVM):
    def fit(...): ...
    def decision_function(...): ...
    # predict is inherited: scores >= 0 -> +1
```

The harness uses structural typing, so any class with `fit`,
`decision_function`, and `predict` is accepted.

---

## 6. Hyperparameters by method

All hyperparameters live in YAML (`configs/*.yaml`), not in code defaults that
override the config. The harness merges `method_kwargs.<method>` into `fit`.

| Method | Typical `method_kwargs` |
|---|---|
| SMO | `max_iter`, `kernel` (linear for main), `cache_size_mb`, KKT/working-set options |
| Pegasos | `lambda_`, `epochs`, `mini_batch`, `average` |
| ADMM | `rho0`, `adaptive_rho`, `alpha` (over-relaxation), `max_iter` |
| PA-ADMM | `K` (Pegasos warm-start epochs), `rho0`, `adaptive_rho`, `alpha` |
| CVXPY | `solver`, `time_limit`, `verbose` |

The ablation config already sets the required grid:
`K ∈ {0,5,10,20}`, adaptive ρ on/off, `alpha ∈ {1.0,1.5,1.8}`,
`rho0 ∈ {0.1,1,10}`. PA-ADMM must count the warm-start cost inside `fit` and
inside `time_history`.

---

## 7. Evaluation protocol the solver must respect

The harness owns this protocol; solvers should not duplicate it:

1. For each `C` in the configured grid, fit on the training partition with the
   given `C_per_sample`.
2. Select the cost-minimizing threshold on **validation** scores.
3. Choose the `C` with the lowest validation cost (ties to smaller `C`).
4. Evaluate the selected model **once** on test with the frozen threshold.
5. Record `time_to_tolerance`, final gap, cost/example, PR-AUC,
   recall@FPR 0.001/0.01, balanced accuracy, iterations, fit/tuning time, memory.
6. Reference objectives are computed post-hoc per comparison group.

### Correctness reference

`configs/correctness.yaml` uses CVXPY (`reference_method: cvxpy`) as the
in-harness reference for ADMM/PA-ADMM. The SMO–LIBSVM agreement check remains
WP2's responsibility; record LIBSVM's objective and accuracy separately and
compare them with the SMO run. The harness deliberately does not add a sixth
method or fabricate a LIBSVM result.

Rules:

- Never tune on test.
- Never use validation/test data inside `fit`.
- Honor `time_budget_sec`; a fit that exceeds it is recorded as a timeout and
  is not selected.
- Same split, `C_i`, tolerance, budget, and single-threaded BLAS across methods.
- SMO ≤ 10k, CVXPY ≤ 20k; caps are enforced by the harness and recorded.
- The harness measures `fit_time_sec` and process peak memory; a solver's own
  timing fields are optional diagnostics.

---

## 8. Contract checker

`experiments/check_solver.py` fits a solver once on synthetic data and checks:

- constructor/`fit`/`decision_function`/`predict` exist and run;
- `w` and `b` are finite and correctly shaped;
- scores are finite 1-D, predictions are in `{-1,+1}`;
- `objective_history` exists;
- `time_history` is aligned if present;
- the reported objective matches the canonical recomputation within `1e-3`;
- `n_iter`/`converged`/`alphas`/`rho_history` are reported when available.

Errors fail the check; warnings indicate missing diagnostics that will weaken
the primary metric (especially a missing `time_history`).

Run it with different sizes/ratios as needed:

```bash
.venv/bin/python experiments/check_solver.py --method pa_admm --n 5000 --ratio 20
```

It writes `results/raw/solver_checks/{method}.json` (ignored). Once it passes,
run `run_correctness` and `run_main` using the committed configs.

The WP4 unit suite also covers the interface:

```bash
.venv/bin/python -m pytest -q
```

---

## 9. Minimal templates

### SMO

```python
import time
import numpy as np
from src.solver_api import BaseBinarySVM, validate_training_data

class SMO(BaseBinarySVM):
    def fit(self, X, y, C_per_sample, *, seed=0, tol=1e-4,
            time_budget_sec=300.0, max_iter=100000, **kwargs):
        X, y, C = validate_training_data(X, y, C_per_sample)
        start = time.perf_counter()
        # ... SMO/dual coordinate descent with 0 <= alpha_i <= C[i] ...
        self.w = ...
        self.b = ...
        self.alphas = ...
        self.n_iter = ...
        self.converged = ...
        self.objective = ...            # canonical unnormalized sum
        self.objective_history = [...]
        self.time_history = [...]
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=np.float64) @ self.w + self.b
```

### ADMM / PA-ADMM

```python
class ADMMSVM(BaseBinarySVM):
    def fit(self, X, y, C_per_sample, *, seed=0, tol=1e-4,
            time_budget_sec=300.0, rho0=1.0, adaptive_rho=True,
            alpha=1.6, max_iter=10000, **kwargs):
        X, y, C = validate_training_data(X, y, C_per_sample)
        # ... optional Pegasos warm start for PA-ADMM ...
        # ... ridge (w,b) update, z update with C[i]/rho, dual update ...
        self.w = ...
        self.b = ...
        self.rho_history = [...]
        self.n_iter = ...
        self.converged = ...
        self.objective = ...
        self.objective_history = [...]
        self.time_history = [...]
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=np.float64) @ self.w + self.b
```

PA-ADMM inherits or shares the ADMM loop and additionally sets
`self.warm_start_epochs = K`; the registry requires a separate class name
`PAADMMSVM`.

---

## 10. Common contract violations

- Constructor requires arguments such as `SMO(C=...)`; the harness calls
  `SMO()`. Put hyperparameters in `fit`.
- Applying class weights inside the solver (double weighting).
- Reporting the objective divided by `n` while the history uses the sum (or
  vice versa).
- Omitting `time_history` (no primary metric).
- Having a `time_history` whose length does not match `objective_history`.
- Returning scores with shape `(n, 1)` or containing NaN/Inf.
- Omitting `predict` (inheriting `BaseBinarySVM` provides a default).
- Accessing validation/test data inside `fit`.
- Not honoring `time_budget_sec` or `tol`.
- Regenerating data/splits instead of using the committed files.

---

## 11. Current status

- WP4 data pipeline, splits, metrics, harness, configs, contract checker, and
  tests are implemented.
- Solver modules (`src/svm_smo.py`, `src/svm_pegasos.py`, `src/svm_admm.py`,
  `src/baseline_cvxpy.py`) and `src/costs.py` are still pending.
- `run_correctness`/`run_main` will skip missing methods and write no fake
  results. `check_solver.py` reports a clear missing-module error.
- The committed split files must not be changed. If you believe a split bug
  exists, contact WP4 before touching `data/`.
