# Experiments (WP4 draft)

> Status: the WP4 data pipeline, splits, metrics, evaluation harness, configs, and experiment
> entry points are implemented and tested. The five solver modules are owned by WP1–WP3 and are
> not present yet. Consequently **this draft contains no final experimental numbers**. The
> tables in `results/tables/` are empty templates and the figures in `results/figures/` are
> explicitly labelled “Pending solver integration”.

## 1. Scope and ownership

WP4 owns the evaluation protocol, not the algorithms. Its deliverables are:

- `src/data.py`: loading, deduplication, stratified and chronological splits, `Amount`
  standardization, synthetic fraud data, training-positive subsampling, label noise.
- `src/metrics.py`: cost per example, PR-AUC, recall at fixed FPR, balanced accuracy,
  cost-minimizing threshold, bootstrap CI, Wilcoxon signed-rank test.
- `src/evaluate.py`: lazy solver registry, validation-only threshold and `C` selection,
  time-to-tolerance, one-run and full-matrix drivers, JSON logging, aggregation.
- `experiments/`, `configs/`, `tests/`, `data/splits/`, and report/slide assets.

WP4 does not implement or modify `src/costs.py`, `src/svm_smo.py`, `src/svm_pegasos.py`,
`src/svm_admm.py`, or `src/baseline_cvxpy.py`.

## 2. Data and preprocessing

### 2.1 Dataset and integrity checks

The ULB credit-card dataset is stored locally at `data/raw/creditcard.csv` and is ignored by
Git. The following checks were run on the local copy:

| Check | Result |
|---|---|
| Raw shape | 284,807 rows × 31 columns |
| Columns | `Time`, `V1`–`V28`, `Amount`, `Class` |
| `Class` values | `{0, 1}` |
| Raw positives | 492 (0.172%) |
| Exact duplicate rows | 1,081 |
| Deduplicated shape | 283,726 rows × 31 columns |
| Deduplicated positives | 473 (0.167%) |
| SHA-256 | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` |

The data are deduplicated **before** splitting. All split indices refer to the deduplicated,
reset-index frame returned by `src.data.clean_data`; they are not raw-CSV row numbers.

### 2.2 Feature policy and leakage controls

- `Time` is never a model feature. It is used only to construct the chronological split.
- The model inputs are `V1`–`V28` followed by `Amount`, in that order (30 features).
- `V1`–`V28` are kept as released.
- `Amount` is standardized with the **training-fold** mean and population standard deviation
  only; the same training statistics are applied to validation and test.
- Labels are mapped to `y ∈ {-1, +1}`.
- Thresholds and base `C` are selected on validation data only. The test partition is evaluated
  once, after the model and threshold are fixed.
- Training-only stress manipulations (positive subsampling, label flips) never touch validation
  or test data.

### 2.3 Splits

Two protocols are used:

1. **Stratified 70/15/15**, seeds 0–4. These are the primary splits for `main_fast`,
   `main_full`, `correctness`, `cost_sweep`, `ablation`, and the sampling/noise stress tests.
2. **Chronological 70/15/15**, seed 0 (and regenerated deterministically for other method
   seeds). Rows are sorted by `Time`; the test block is the latest 15% of transactions.

Committed split files:

| File | Train / validation / test rows | Fraud counts |
|---|---|---|
| `data/splits/stratified_seed0.json` … `stratified_seed4.json` | 198,608 / 42,559 / 42,559 | 331 / 71 / 71 |
| `data/splits/chronological_seed0.json` | 198,608 / 42,558 / 42,560 | 366 / 55 / 52 |

For the 50k experiments (`main_fast`, `cost_sweep`), the subset is drawn once with
`subset_seed: 0` and the run seed controls the 70/15/15 split. This keeps the underlying
transaction sample fixed when seed variance is reported. The 10k `correctness` subset is also
fixed (`subset_seed: 0`).

The chronological test fraud rate (0.122%) is materially lower than the training rate (0.184%).
The temporal experiment therefore measures both solver behaviour and distribution drift, and
its results should not be merged with the stratified estimates.

`experiments/generate_splits.py` regenerates all six files deterministically and can be used to
verify the committed indices.

### 2.4 Stress-test data variants

- **Imbalance:** adjust training prevalence to 0.1%, 0.5%, 1%, 5%, 10% while keeping
  validation/test fixed. Positive-only subsampling cannot reach rates above the natural
  cleaned training prevalence (0.167%), so the harness uses positive subsampling below that
  rate and training-negative subsampling above it. The chosen mechanism, before/after class
  counts, and realized rate are recorded in each run JSON. The higher-prevalence variants also
  reduce the number of training rows, which must be reported as a confound when comparing
  runtime.
- **Label noise:** flip 5% or 10% of training labels only. Test labels remain clean.
- **Temporal:** use the chronological split above.

The synthetic scaling generator (`src.data.make_synthetic_fraud`) creates standard-normal
features with a shifted informative subspace and labels in `{-1, +1}`. It is a controlled
stress test of the same problem, not a second application.

## 3. Evaluation protocol

### 3.1 Cost-sensitive weights

For cost ratio `r = c_FN / c_FP`, the harness builds

```text
w_- = 1,  w_+ = r,  normalized so that mean_i(w_{y_i}) = 1,
C_i = C · w_{y_i}.
```

This preserves `w_+ / w_- = c_FN / c_FP` and gives the same class-weight construction as the
proposal. Per-sample weights depend on the current training partition; capped subsets are
normalized on their own training rows. The canonical cost model remains owned by WP1.

### 3.2 Objective convention and reference

Solvers must expose `w`, `b`, `objective_history`, and, where available, a per-iteration
timestamp history. The harness recomputes the canonical weighted primal objective

```text
J(w, b) = 0.5 ||w||² + Σ_i C_i max(0, 1 - y_i (wᵀx_i + b))
```

from `w` and `b` and records the solver-reported objective as well. If the reported objective
differs from the recomputed value by more than `1e-3` relative error, the run is marked
inconsistent and `time_to_tolerance` is not reported for it, rather than silently mixing scales.

Reference objectives are assigned per `(dataset, cost ratio, seed, n_train)`:

- `correctness`: CVXPY is the in-harness reference for ADMM/PA-ADMM. The SMO–LIBSVM comparison
  is external to this harness and remains pending WP2; no LIBSVM numbers are fabricated.
- `main_*`, `cost_sweep`, `ablation`, `stress`, `scaling`: the best observed canonical training
  objective in the same comparison group. Runs with different training sizes (solver caps) are
  never put in the same reference group.

### 3.3 Primary metric: time-to-tolerance

For a run with objective history `J_0, J_1, …` and timestamps `t_0, t_1, …`,

```text
relative_gap_k = (J_k - J_ref) / |J_ref|
reached        = final relative_gap <= tol
time_to_tolerance = t_k at the first k with relative_gap_k <= tol.
```

If tolerance is not reached, `time_to_tolerance = null` and the final gap is recorded. If a
solver exposes no timestamp history, the final gap is still reported but timing is `null`; the
harness does not invent evenly spaced timestamps. The primary tolerance is `1e-3` for the main
full-data runs and `1e-4` for correctness runs; both are configurable in YAML.

### 3.4 Threshold and hyperparameter selection

For each candidate `C` in `{0.01, 0.1, 1, 10, 100}` (the grid is configurable):

1. Fit the method on the training partition with that `C`.
2. Select the cost-minimizing threshold on validation scores.
3. Compute validation cost per example.
4. Keep the `C` with the lowest validation cost (ties go to the smaller `C`).

Only the selected model is scored on test, using the validation threshold. The selected fit’s
wall-clock time is reported as `fit_time_sec`; the sum of all tuning fits is reported separately
as `tuning_fit_time_sec` so tuning cost is visible rather than hidden. `correctness`, `ablation`,
and `scaling` use a fixed `C=1` to isolate agreement/components/scaling; this is recorded in
their YAML notes.

### 3.5 Secondary metrics and statistics

- `cost_per_example = (c_FN·FN + c_FP·FP) / n_test`
- PR-AUC (average precision)
- recall at FPR 0.001 and 0.01
- balanced accuracy at the validation-selected threshold
- `objective`, `relative_gap`, `n_iter`, `converged`, `fit_time_sec`, `peak_memory_mb`
- `peak_memory_mb` is process-level maximum RSS (Linux `ru_maxrss`), not an isolated per-fit
  allocator measurement; this is stated in every run JSON.

Across seeds, tables report mean, standard deviation, count, and a percentile bootstrap 95%
confidence interval for the mean. `src.metrics.wilcoxon_test` provides paired signed-rank tests
for method comparisons once per-seed runs exist.

### 3.6 Failure, timeout, and fairness handling

- Missing solver modules are skipped with an explicit log message; no result JSON is written.
- Solver exceptions are recorded in a failure JSON with `status = "failed"` and the traceback;
  they are never dropped silently.
- A fit whose wall-clock time exceeds the per-fit budget is recorded as a timeout candidate and
  is not selected. In-process timeouts rely on solvers honouring `time_budget_sec`; the harness
  cannot kill an uncooperative solver.
- Same split indices, same cost ratio, same tolerance, same time budget, single-threaded BLAS,
  and the same machine are used across methods. SMO is capped at 10k and CVXPY at 20k, as
  required by the memory/scale plan; caps are recorded in each JSON and are not compared in the
  same reference group as full-data runs.

## 4. Experiment matrix

| Experiment | Data | Methods | Ratios | Seeds | C grid | Tolerance | Budget | Jobs |
|---|---|---|---|---|---|---|---|---|
| `correctness` | 10k stratified subset | SMO, ADMM, PA-ADMM, CVXPY | 10 | 0 | 1.0 | 1e-4 | 300 s | 4 |
| `main_fast` | 50k stratified subset | all five; SMO/CVXPY capped at 10k | 10, 20, 50 | 0–2 | 0.01–100 | 1e-3 | 300 s | 45 |
| `main_full` | full cleaned ULB | all five; SMO 10k, CVXPY 20k | 10, 20, 50 | 0–4 | 0.01–100 | 1e-3 | 900 s | 75 |
| `cost_sweep` | 50k subset | Pegasos, ADMM, PA-ADMM | 1, 3, 5, 10, 20, 50 | 0 | 0.01–100 | 1e-3 | 300 s | 18 |
| `ablation` | full cleaned ULB | ADMM + 10 PA-ADMM variants | 20 | 0–2 | 1.0 | 1e-3 | 900 s | 33 |
| `stress` | full ULB; 5 imbalance rates, 2 noise rates, chronological | SMO (noise only), Pegasos, ADMM, PA-ADMM | 20 | 0–2 | 0.01–100 | 1e-3 | 900 s | 78 |
| `scaling` | synthetic 10k–200k, d=20 | Pegasos, ADMM, PA-ADMM | 10 | 0–2 | 1.0 | 1e-3 | 900 s | 45 |

The ablation covers warm-start `K ∈ {0,5,10,20}`, adaptive ρ on/off, over-relaxation
`α ∈ {1.0,1.5,1.8}`, and `ρ0 ∈ {0.1,1,10}`. The exact variants are in
`configs/ablation.yaml`.

All seeds, cost ratios, `C` grids, tolerances, budgets, paths, caps, and method hyperparameters
live in `configs/*.yaml`; no experimental setting is hard-coded in the entry points.

## 5. Implementation and commands

### 5.1 Entry points

```bash
python experiments/run_correctness.py --config configs/correctness.yaml
python experiments/run_main.py        --config configs/main_fast.yaml
python experiments/run_main.py        --config configs/main_full.yaml
python experiments/run_cost_sweep.py  --config configs/cost_sweep.yaml
python experiments/run_ablation.py    --config configs/ablation.yaml
python experiments/run_stress.py      --config configs/stress.yaml
python experiments/run_scaling.py     --config configs/scaling.yaml
```

Every entry point supports `--dry-run`, which prints the resolved YAML and the planned
`(data variant, scale, seed, ratio, method, variant, cap, C grid, tolerance, budget)` jobs
without importing solvers or fitting anything. Example:

```bash
python experiments/run_main.py --config configs/main_fast.yaml --dry-run
```

The current environment has no solver modules, so a real run exits with an explicit
“no solver modules are available” message and writes no result JSON.

### 5.2 Harness smoke test

```bash
python experiments/run_smoke.py
```

This uses `sklearn.linear_model.SGDClassifier(loss="hinge")` on synthetic data and writes
`results/raw/harness_smoke/...`. It is a **harness smoke test only**, not a project result and
not a substitute for the WP1–WP3 solvers. The JSON carries the same warning in `data_meta`.

### 5.3 Results, aggregation, and assets

- Run JSON: `results/raw/{experiment}/{dataset}/{method}/ratio-{r}/seed-{s}.json`
  (ablation variants add a `variant-{name}` level). Ignored by Git.
- `src.evaluate.aggregate_runs` writes `results/tables/{experiment}_runs.csv`,
  `{experiment}_summary.csv`, and one `{experiment}_{metric}.csv` per metric.
- `experiments/make_report_assets.py` writes report figures. With no solver results it writes
  placeholder PDFs that state “Pending solver integration”; it never inserts synthetic numbers.
  Existing non-empty tables are not overwritten.
- `results/tables/_pending_solver_integration.csv` records which tables are still empty.

### 5.4 Tests

`pytest` covers deduplication/split invariants, training-only `Amount` scaling, cost/PR-AUC/
recall@FPR/threshold metrics, bootstrap/Wilcoxon helpers, the full `run_one` JSON path, the
matrix driver with reference-objective assignment, failure logging, and aggregation.

## 6. Notes and limitations of this section

This draft documents the protocol and the implementation. It deliberately makes no claim about
which solver is faster or which cost ratio is preferable. Those claims require the WP1–WP3
modules and the final runs; they will be added only from `results/tables/` and the committed
configs/splits.
