# Experiment Plan — Cost-Sensitive SVM for Credit Card Fraud Detection

This is the project's working document. It defines **what we build, what we run, and where
everything lives**. It is intentionally a plan, not an implementation: no algorithm or
experiment script is included in this initial commit.

The proposal is in [`docs/proposal/COMP6704_Group4_SVM_Research_Proposal.pdf`](docs/proposal/COMP6704_Group4_SVM_Research_Proposal.pdf).

---

## 1. Project in one paragraph

Credit-card fraud detection is a binary decision with **0.172% positives**, **asymmetric
costs** (a missed fraud costs more than a false alarm), **noisy labels**, and a **training
time budget**. We model it as a **cost-sensitive soft-margin SVM**: a convex QP with
class-dependent penalties `C_i = C * w_{y_i}` and `w_+ / w_- = c_FN / c_FP`. We compare five
solvers — SMO, weighted Pegasos, fixed-ρ ADMM, PA-ADMM (Pegasos-warm-started adaptive ADMM),
and a CVXPY/ECOS interior-point baseline — under one protocol. The primary metric is
**time-to-cost-tolerance**, not accuracy.

There is **one application only**: the ULB credit-card dataset. Synthetic data are controlled
stress tests of the same fraud problem (fraud rate, label noise, cost ratio, scaling), not a
second application.

---

## 2. Repository contract — where everything goes

### Current contents

```
EXPERIMENTS.md                         # this document
docs/proposal/
  COMP6704_Group4_SVM_Research_Proposal.pdf
data/
  raw/
    .gitkeep
    creditcard.csv                     # 144 MB; present locally, ignored by Git
.gitignore
```

The WP4 branch (`feat/eval`) adds `src/` (data/metrics/evaluation/solver API/utils),
`configs/`, `experiments/`, `tests/`, `data/splits/`, `docs/INTERFACES.md`,
and placeholder tables/figures under `results/`. Solver modules and `src/costs.py`
remain owned by WP1–WP3 and are not present yet. The solver-facing contract is in
[`docs/INTERFACES.md`](docs/INTERFACES.md).

### Directories to create when implementation starts

| Path | Contents | Committed? |
|---|---|---|
| `data/raw/creditcard.csv` | Raw ULB CSV | **No** (ignored) |
| `data/processed/` | Optional cleaned/standardized arrays or caches | **No** (ignored) |
| `data/splits/` | Reproducible split indices (`stratified_seed0.json`, …, `chronological_seed0.json`) | **Yes** (small, deterministic) |
| `src/` | Reusable library code (data, costs, metrics, solvers, evaluation) | **Yes** |
| `experiments/` | Runnable experiment entry points | **Yes** |
| `configs/` | Optional YAML configs for experiments | **Yes** |
| `results/raw/` | One JSON per run | **No** (ignored) |
| `results/tables/` | Aggregated CSV tables | **Yes** |
| `results/figures/` | Final plots (PDF/PNG) | **Yes** |
| `results/logs/` | Run logs | **No** (ignored) |

Rules:

- Never commit `data/raw`, `data/processed`, `results/raw`, or `results/logs`.
- Commit split indices and aggregated tables/figures.
- Keep one owner per file; coordinate before editing another WP's module.

---

## 3. Data

### 3.1 Dataset and current status

| Item | Value |
|---|---|
| Name | ULB Credit Card Fraud Detection |
| Local file | `data/raw/creditcard.csv` |
| Shape | **284,807 rows × 31 columns** (verified) |
| Columns | `Time`, `V1`–`V28`, `Amount`, `Class` |
| Target | `Class` (1 = fraud, 0 = legitimate) |
| Positives | **492** (0.172%) |
| Duplicates | 1,081 exact duplicate rows; drop before splitting |
| Rows after dedup | **283,726** (473 frauds, 0.167%) |
| SHA-256 | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` |
| Source | Kaggle `mlg-ulb/creditcardfraud`; downloaded via the public Hugging Face mirror `David-Egea/Creditcard-fraud-detection` |

The file is already downloaded and verified. It is **not** in Git. If another member needs it,
download the same file from the Kaggle page or the public mirror and confirm the shape and
fraud count before use.

### 3.2 Feature policy

- `Time` is **not** a model feature; it is used only for the chronological split.
- `V1`–`V28` are the released PCA components; keep them as-is.
- `Amount` is standardized using **training-fold** mean and standard deviation only.
- Labels are mapped to `y ∈ {-1, +1}`.
- Thresholds, `C`, and all hyperparameters are selected on validation data only.

### 3.3 Splits

| Split | Protocol | Seeds | Use |
|---|---|---|---|
| Stratified | 70% train / 15% validation / 15% test | 0–4 | Main comparison |
| Chronological | Sort by `Time`; 70% train / 15% validation / 15% test | 0 | Temporal robustness check |

- Save **integer row indices** as JSON in `data/splits/`; do not copy the data.
- Index convention: positional indices into the **deduplicated, reset-index** frame in
  `src/data.py::clean_data` (283,726 rows). The raw CSV is never indexed directly.
- All methods use the same split files.
- For imbalance stress tests, change **training prevalence only**; validation/test stay fixed.
  Positive-only subsampling can only lower prevalence below the natural cleaned rate (0.167%),
  so targets of 0.5%/1%/5%/10% cannot be reached that way. The harness therefore subsamples
  training positives for lower targets and training negatives for higher targets, and records
  the mechanism (`positive_subsample` / `negative_subsample`) in every run JSON.
- For label-noise stress tests, flip labels in the **training set only**; test labels stay clean.
- For the chronological check, the test set is the latest 15% of transactions; report the
  distribution shift explicitly.

Generated files (`experiments/generate_splits.py`, committed, ~11 MB total):

| File | Train / val / test rows | Fraud counts | Fraud rates |
|---|---|---|---|
| `stratified_seed0..4.json` | 198,608 / 42,559 / 42,559 | 331 / 71 / 71 | 0.167% / 0.167% / 0.167% |
| `chronological_seed0.json` | 198,608 / 42,558 / 42,560 | 366 / 55 / 52 | 0.184% / 0.129% / 0.122% |

The chronological test block is the latest 15% by `Time` (`Time` range 151,320–172,792).
Its fraud rate is about one-third lower than the training block, so the temporal check also
measures distribution drift; do not read it as a clean estimate of the same population.

### 3.4 When re-downloading

Canonical Kaggle dataset: `mlg-ulb/creditcardfraud`.

Manual options:

- Kaggle CLI: `kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip`
- Public mirror used here:
  `https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection`

After downloading, verify: shape `(284807, 31)`, columns include `Class`, 492 positives,
`Class ∈ {0, 1}`. Record the SHA-256 in the team chat or in this document.

---

## 4. Code to implement

### 4.1 Common interface

Every solver implements the same protocol:

```python
class BinarySVM:
    def fit(self, X, y, C_per_sample, **kwargs): ...
    def decision_function(self, X): ...
    def predict(self, X): ...
```

- `X`: `float64` array `(n, d)`; `y`: `int8` array `(n,)` in `{-1, +1}`.
- `C_per_sample`: `float64` array `(n,)`; `C_i = C * w_{y_i}`, normalized to mean one.
- After `fit`, record: `w`, `b`, `n_iter`, `converged`, `objective`,
  `objective_history`, `time_history`; method-specific state (`alphas`, `rho_history`).
  The harness measures `fit_time_sec` and a per-fit sampled RSS delta; at least two
  aligned history points are required for a non-null time-to-tolerance.
- `predict` returns `{-1, +1}`; `decision_function` returns raw scores.
- Solver owners must implement the module/class names in §4.2 and pass
  `experiments/check_solver.py --method <name>` before running experiments.
  Full implementation details, templates, objective convention, and common
  mistakes are in [`docs/INTERFACES.md`](docs/INTERFACES.md).

### 4.2 Modules under `src/`

| File | What to implement | Owner |
|---|---|---|
| `src/data.py` | Load CSV, drop duplicates, stratified/chronological splits, standardize `Amount`, build NumPy arrays, save/load split JSON, synthetic fraud generator, positive subsampling | WP4 |
| `src/costs.py` | Cost matrix, normalized class weights, per-sample `C_i`, expected cost | WP1 |
| `src/metrics.py` | Cost per example, PR-AUC, recall@FPR, balanced accuracy, cost-minimizing threshold, bootstrap CI, Wilcoxon test | WP4 |
| `src/svm_smo.py` | SMO / dual coordinate descent with per-class bounds; linear + optional RBF; kernel cache; KKT stopping; recover `w`, `b` | WP2 |
| `src/svm_pegasos.py` | Weighted Pegasos: primal stochastic subgradient, learning-rate schedule, optional iterate averaging | WP1 |
| `src/svm_admm.py` | Fixed-ρ ADMM and PA-ADMM: ridge solve, closed-form z-update, residuals, adaptive ρ, over-relaxation, warm start | WP3 |
| `src/baseline_cvxpy.py` | CVXPY/ECOS cost-sensitive linear SVM on a stratified subset | WP1 |
| `src/evaluate.py` | Unified fit/evaluate runner, time-to-tolerance, threshold selection, JSON logging, aggregation | WP4 |
| `src/solver_api.py` | Solver protocol, validation helpers, base class, contract checker | WP4 |
| `src/utils.py` | YAML loading/merging, seeding, timer, logger, single-thread configuration | Shared |

### 4.3 Algorithm requirements

#### SMO (WP2)

- Dual: `0 <= alpha_i <= C_i`, `sum alpha_i y_i = 0`.
- Working-set selection: maximal-violating pair, optionally second-order.
- Analytic two-variable update; clip to `[0, C_i]`.
- Maintain `g_i = y_i f(x_i) - 1`; stop on maximal KKT violation `< tol`.
- Linear kernel; RBF only on a small subset for a qualitative check.
- Recover `w = sum alpha_i y_i x_i`; recover `b` from free support vectors.
- Log objective and KKT violation histories.

#### Weighted Pegasos (WP1)

- Primal objective: `0.5||w||^2 + sum C_i max(0, 1 - y_i(w^T x_i + b))`.
- Update when `y_i f(x_i) < 1`:
  `w <- (1 - eta_t * lambda) w + eta_t * C_i * y_i * x_i`,
  `b <- b + eta_t * C_i * y_i`.
- `eta_t = 1 / (lambda * t)`; document how `lambda` is tied to `C` and `n`.
- Record objective history; optional iterate averaging; full-data and mini-batch modes.

#### ADMM and PA-ADMM (WP3)

- Splitting: `z_i = y_i (w^T x_i + b)`.
- Augmented Lagrangian with dual `u` and penalty `rho`.
- `(w,b)`-update: ridge regression; cache `X^T X`; Cholesky; refactor when `rho` changes.
- `z`-update: `a_i = y_i(w^T x_i + b) - u_i/rho`;
  `z_i = a_i` if `a_i >= 1`, else `min(1, a_i + C_i/rho)`.
- Dual update, primal/dual residuals, stopping on residuals + relative objective gap.
- Adaptive `rho`: multiply/divide by 2 when residuals differ by an order of magnitude;
  rescale `u`.
- Over-relaxation with `alpha ∈ [1.5, 1.8]`.
- PA-ADMM: weighted Pegasos warm start (`K` epochs), then the adaptive ADMM loop.
- Log `rho_history`, residuals, objective history.

#### CVXPY baseline (WP1)

- Primal cost-sensitive SVM with per-sample `C_i`; ECOS solver (fallback SCS).
- Runs on the same training rows as the other methods; no subset cap. At full n it may be
  infeasible/time out; the failure is recorded. `main_capped` is the fair all-five fallback.
- Returns `w`, `b`, objective, solver status/solve time.

### 4.4 Experiment entry points under `experiments/`

| File | Runs |
|---|---|
| `run_correctness.py` | SMO vs LIBSVM; ADMM/PA-ADMM vs CVXPY on a 10k subset |
| `run_main.py` | `main_fast` (50k, 3 seeds), `main_full` (283,726 cleaned, 5 seeds), `main_capped` (common 10k, 3 seeds), and `predictive` (validation-tuned C on the common 10k) |
| `run_cost_sweep.py` | Cost ratios `1, 3, 5, 10, 20, 50` on a 50k subset |
| `run_ablation.py` | PA-ADMM component ablation |
| `run_stress.py` | Fraud-rate, label-noise and chronological stress tests |
| `run_scaling.py` | Synthetic `n = 10k–200k`, `d = 20` |
| `check_solver.py` | Contract check for one solver on synthetic data (WP2/WP3 handoff) |
| `finalize_results.py` | End-of-campaign one-shot: refresh `correctness`, re-aggregate every experiment, regenerate `results/tables` + `results/figures`, rebuild the generated PART IV deck with `--check`, and print a per-method completeness report |
| `run_member2.py` | WP2's standalone SMO-vs-LIBSVM numerical cross-check. It writes to `results/member2/` (ignored) and is **not** part of the unified matrix; the official SMO results come from `run_correctness.py` / `run_main.py` through the registry, on the same rows and protocol as every other method. |

Configs under `configs/` are optional but recommended: keep all paths, seeds, cost ratios,
grids, tolerances and time budgets in YAML rather than hard-coding them.

---

## 5. Experiments to run

| ID | Data | Methods | Cost ratios | Seeds | Purpose |
|---|---|---|---|---|---|
| `correctness` | 10k stratified subset | SMO vs LIBSVM; ADMM/PA-ADMM vs CVXPY | 10:1 | 1 | Verify objectives, KKT/residuals, predictions |
| `main_fast` | 50k stratified subset | all five on the same 50k | 10, 20, 50 | 3 | Fast iteration; debug the matrix |
| `main_full` | full cleaned ULB (283,726; train 198,608) | all five on the same full training set | 10, 20, 50 | 5 | Full-data comparison; SMO/CVXPY may fail |
| `main_capped` | common 10k subset | all five on the same 10k | 10, 20, 50 | 3 | Fair all-method fallback if full SMO/CVXPY fail |
| `predictive` | common 10k subset | all five on the same 10k | 10, 20, 50 | 3 | Validation-tuned `C`; predictive/operational metrics only |
| `cost_sweep` | 50k subset | Pegasos, ADMM, PA-ADMM | 1, 3, 5, 10, 20, 50; **evaluated at fixed 20:1** | 1 | How training-weight asymmetry changes a common-cost deployment metric |
| `ablation` | full data | PA-ADMM variants | 20:1 | 3 | `K ∈ {0,5,10,20}`; adaptive ρ on/off; `alpha ∈ {1.0,1.5,1.8}`; `rho0 ∈ {0.1,1,10}` |
| `stress_imbalance` | training prevalence adjusted to 0.1/0.5/1/5/10% (positive or negative subsampling, recorded) | Pegasos, ADMM, PA-ADMM | 20:1 | 3 | Extreme-imbalance behaviour |
| `stress_noise` | 5%/10% training labels flipped | SMO, Pegasos, ADMM, PA-ADMM on the same full training rows | 20:1 | 3 | Objective–generalization gap |
| `temporal` | chronological 70/15/15 | Pegasos, ADMM, PA-ADMM | 20:1 | 1–3 | Distribution drift |
| `scaling` | synthetic `n = 10k, 25k, 50k, 100k, 200k`; `d = 20` | Pegasos, ADMM, PA-ADMM | 10:1 | 3 | Runtime/memory vs `n` |

### 5.1 Same-data policy, feasibility, and timeouts

- **No per-method training cap.** Every method inside an experiment receives exactly the same
  `X_train`, `y_train`, split, and `C_i`. Mixed-size comparisons are not allowed.
- `main_full` attempts all five methods on the full training set. SMO/CVXPY may time out or run
  out of memory; those runs are recorded as failures and are not ranked. Successful methods are
  compared against each other on the same full data.
- **`data.subset_n` counts TOTAL rows before the 70/15/15 split.** `main_capped` and
  `predictive` use `subset_n: 10000`, which is 7,000 training rows (12 frauds at
  subset_seed 0/seed 0), not 10,000 training rows; `main_fast` uses 50,000 rows and
  trains on 34,999.
- `main_capped` is the fair all-five fallback on the common 10k-row subset; use it when
  SMO/CVXPY cannot complete the full-data matrix.
- **Fixed-QP rule for solver ranking.** `main_fast`, `main_full`, and `main_capped` fix
  `C_grid: [1.0]` for every method, so `objective`, `relative_gap`, and `time_to_tolerance`
  compare the same optimization problem. Validation-only threshold selection remains.
  `configs/predictive.yaml` tunes `C` per method for cost/PR-AUC/recall comparisons; its
  objective-based metrics are grouped by the selected `C` and are not a solver ranking.
- ADMM/PA-ADMM full data is feasible because `d = 29` (V1–V28 + Amount; Time is not a feature)
  and the linear system is 30×30. Pegasos is O(d) per iteration.
- **ADMM rho protocol (decided by preflight).** On the common 50k split, fixed `rho0=1` hit the
  10k iteration cap without convergence, while `adaptive_rho=true` converged near the optimum in
  about 8.5k iterations. The main configs therefore run `admm` with `rho0=1`,
  `adaptive_rho=true`, `alpha=1`, `max_iter=20000`; the fixed-rho cold baseline remains the
  `admm-cold` variant in the ablation. Report `converged`/gap and never present an unconverged
  fit as a converged speed record.
- Per-fit time budget and hard timeout are both **5400 s (90 minutes)**. Timeout/OOM is recorded
  as a failure; do not drop it silently. Solver C-grid tuning is included in
  `tuning_fit_time_sec`, separate from the selected fit's `fit_time_sec`.

---

## 6. Metrics and evaluation protocol

### 6.1 Primary metric

**`time_to_tolerance`** — wall-clock seconds to reach a relative weighted objective gap
`<= tol` (default `1e-4`; `1e-3` for full data). If a method does not reach tolerance within
the budget, record `reached = false`, `time_to_tolerance = null`, and the final gap. This is
the main "faster" claim.

A run needs at least two aligned objective/timestamp points for a non-null time-to-tolerance;
single-checkpoint solvers (e.g. an interior-point solve reported as one final point) get
`time_to_tolerance = null`. A comparison group needs at least two successful methods; a
single successful run never uses itself as its own reference. The group key includes the
**selected `C`**: changing `C` changes the QP, so runs with different selected `C` are never
compared on objective, relative gap, or time-to-tolerance. `main_fast`, `main_full`, and
`main_capped` fix `C=1` so all methods share one QP.

### 6.2 Secondary metrics

- `cost_per_example = (c_FN * FN + c_FP * FP) / n_test`;
- `pr_auc` (average precision);
- `recall_at_fpr_0.001` and `recall_at_fpr_0.01`;
- `balanced_accuracy`;
- `objective` (weighted training objective) and `relative_gap`;
- `n_iter`, `fit_time_sec`, `tuning_fit_time_sec`, and per-fit sampled-RSS
  `peak_memory_mb` (`process_rss_peak_mb` is retained for audit);
- `rho_history` (ADMM/PA-ADMM) and `alphas_summary` (SMO);
- `config_hash`, `training_cost_ratio`, `evaluation_cost_ratio`, `costs_source`.

### 6.3 Reference objective

- Correctness runs: CVXPY/LIBSVM reference.
- Full-data runs: the best observed training objective among the methods on the same split,
  the same training rows, and the **same selected `C`**. Groups with only one successful method
  at that `C` get no time-to-tolerance.
- Store `reference_objective` in every JSON so the gap is reproducible.
- Only runs whose canonical objective the harness can recompute from `w,b` join a reference
  group; nonlinear models keep their predictive metrics but get
  `tolerance_note: objective_not_recomputable_from_w_and_b` instead of a gap.
- Never use test data to choose the reference or the threshold.
- Split files are validated against the raw CSV SHA-256, cleaned row count, and positive count
  before use.

### 6.4 Threshold and tuning

- Select the decision threshold on **validation** to minimize validation cost.
- Solver-ranking runs (`correctness`, `main_fast`, `main_full`, `main_capped`, `ablation`,
  `scaling`): fix `C=1` so every method solves the same QP.
- Predictive runs (`configs/predictive.yaml`): select base `C` on validation cost from
  `{0.01, 0.1, 1, 10, 100}`; report cost/PR-AUC/recall, and do not compare objective-based
  metrics across different selected `C`. The harness enforces this by including `C` in the
  reference and pairwise-objective grouping keys.
- Class weights are fixed by the cost ratio; normalize to mean one.
- Until WP1's `src/costs.py` lands, the harness uses its documented sample-mean placeholder
  and records `costs_source` in every JSON. Reconcile the two normalizations before final runs.
- Training weights and deployment cost are separate: `cost_sweep` trains at each ratio but
  selects thresholds and reports test cost at the common `evaluation_cost_ratio: 20`.
- SMO: linear kernel for the main matrix; an RBF run, if included, is a separate
  qualitative 10k check and is never mixed into the main full-data ranking. The harness
  enforces this: an objective that cannot be recomputed from `w,b` (nonlinear kernels have
  no `w`) is marked `objective_recomputable: false`, keeps its predictive metrics, and is
  excluded from reference, gap, time-to-tolerance, and objective-aggregation groups.
- Pegasos: `C` grid and learning-rate/epoch settings.
- ADMM/PA-ADMM: main configs use adaptive ρ, `rho0=1`, `alpha=1` (ADMM) / `1.6`
  (PA-ADMM), `max_iter=20000`; PA-ADMM also `K=10`. The fixed-rho control is the
  `admm-cold` ablation variant.
- No test-set tuning; no per-method early stopping on test.

### 6.5 Fairness controls

- Same split indices, training rows, `C_i`, tolerance, single-threaded execution, and hardware.
  No method gets a different training subset.
- Same hard timeout per fit (5400 s); report both per-iteration and per-time behaviour.
- Do not overwrite run JSONs; every run records a config hash.
- Report `fit_time_sec` with `tuning_fit_time_sec`, not the selected fit alone.
- Warm start is used **only** by PA-ADMM; cold-start ADMM is the baseline.
- Record CPU/RAM in run metadata.
- Five seeds for `main_full`; mean ± std, bootstrap 95% CI, Wilcoxon signed-rank tests.

---

## 7. Results — where they go

```
results/
  raw/
    {experiment}/
      {dataset}/
        {method}/
          ratio-{r}/
            seed-{s}.json
  tables/
    {experiment}_{metric}.csv
  figures/
    {experiment}_{plot}.pdf
  logs/
    {experiment}_{timestamp}.log
```

### 7.1 JSON schema (one run)

Schema illustration only; the numeric values are placeholders, not results. The
real full-data protocol has `d = 29` (V1–V28 + Amount) and 198,608/42,559/42,559
train/validation/test rows.

```json
{
  "experiment": "main_full",
  "dataset": "ulb",
  "method": "pa_admm",
  "cost_ratio": 20,
  "seed": 0,
  "n_train": 198608,
  "n_val": 42559,
  "n_test": 42559,
  "d": 29,
  "C": 1.0,
  "hyperparams": {"K": 10, "rho0": 1.0, "alpha": 1.6, "adaptive_rho": true},
  "fit_time_sec": 42.1,
  "n_iter": 87,
  "converged": true,
  "objective": 1234.5,
  "reference_objective": 1200.0,
  "relative_gap": 0.0287,
  "time_to_tolerance_sec": 38.4,
  "reached_tolerance": true,
  "pr_auc": 0.81,
  "recall_at_fpr_0.001": 0.72,
  "recall_at_fpr_0.01": 0.88,
  "cost_per_example": 0.31,
  "balanced_accuracy": 0.84,
  "peak_memory_mb": 512.0
}
```

Additional audit fields in the implementation: `training_cost_ratio`, `evaluation_cost_ratio`,
`config_hash`, `costs_source`, `hard_timeout_sec`, `history_points`,
`time_to_tolerance_upper_bound`, `selection_incomplete`, `tuning_fit_time_sec`,
`process_rss_peak_mb`, `objective_recomputable`, `kernel`, `gamma`, and
`alphas_summary`/`rho_history` for the relevant solvers. Reused runs add `reused` and
`reuse_config_hash`.
`peak_memory_mb` is a per-fit sampled process-RSS delta, not a lifetime maximum.

### 7.2 Tables and figures

- Tables: one row per run; aggregated tables with mean/std and `n_seeds`.
- Figures: convergence (objective vs time), time-to-tolerance bars, cost vs cost ratio,
  PR curves, recall@FPR, scaling (runtime/memory vs n), ρ sensitivity, and the
  all-five `main_capped` comparison.
- Final figures for the report go in `results/figures/`; slides reuse them.
  Member 4's deck is generated by `slides/build_part4_deck.py --template … --check`
  from these tables and the run JSONs (see `slides/README.md`); it is never
  hand-edited, so a re-run after new results cannot leave a stale number.

### 7.3 Naming rules

- Experiment IDs: `correctness`, `main_fast`, `main_full`, `main_capped`, `predictive`,
  `cost_sweep`, `ablation`, `stress_imbalance`, `stress_noise`, `temporal`, `scaling`.
- Method IDs: `smo`, `pegasos`, `admm`, `pa_admm`, `cvxpy`.
- Dataset IDs: `ulb`, `synthetic`.
- Cost ratio: integer `c_FN/c_FP` (`ratio-20`).
- Seed: `seed-0`, no padding.
- Runs are not overwritten by default (`overwrite: false`); every JSON stores a `config_hash`.
  For a deliberate rerun, use a fresh results directory or explicitly enable overwrite.
- The main configs set `reuse_existing: true`. A successful existing run is reused only when
  its dataset, rows/positives, ratios, C grid, hyperparameters, tolerance, budgets, costs
  source, and split metadata match the current job; a mismatch raises instead of silently
  reusing a different experiment. Existing failed runs are retried. `overwrite: true` forces
  a full refit. This supports adding remaining methods to `main_full` without refitting
  ADMM/PA-ADMM.

---

## 8. Ownership and Git workflow

| WP | Owner | Files | Branch |
|---|---|---|---|
| WP1 | Member 1 | `src/costs.py`, `src/svm_pegasos.py`, `src/baseline_cvxpy.py`, theory notes | `feat/costs-pegasos` |
| WP2 | Member 2 | `src/svm_smo.py`, SMO tests | `feat/smo` |
| WP3 | Member 3 | `src/svm_admm.py`, ADMM/PA-ADMM tests | `feat/admm` |
| WP4 | Member 4 | `src/data.py`, `src/metrics.py`, `src/evaluate.py`, `src/solver_api.py`, `experiments/`, `configs/`, results | `feat/eval` |

Rules:

- One owner per file; ask before editing another WP's module.
- Every PR: tests pass, a small run is logged, and this document is updated if the protocol
  changes.
- Do not commit raw data, processed data, virtual environments, or `results/raw`.
- Commit split indices and aggregated tables/figures.
- PR title: `feat(smo): ...`, `fix(admm): ...`, `docs(experiments): ...`.
- One reviewer from another WP before merge.

---

## 9. Definition of done

A work package is done when:

- the module implements the documented interface;
- unit tests pass on a small synthetic dataset;
- correctness is checked against LIBSVM or CVXPY (objective/accuracy within tolerance);
- a smoke run produces a JSON in `results/raw/` with all required fields;
- the relevant section of this document is updated.

The whole project is done when:

- `main_full`, `cost_sweep`, `ablation`, `stress_*`, `temporal`, and `scaling` have run;
- aggregated tables and figures are in `results/`;
- the report and 40-minute presentation use numbers generated from `results/tables/`;
- a fresh clone can reproduce the main tables from the committed configs and split indices.

---

## 10. Risks and practical notes

- **Extreme imbalance:** PR-AUC and recall@FPR are noisy; use five seeds, bootstrap CIs, and
  report variance. Do not report a single split's accuracy.
- **Cost ratios are assumptions:** the main ratio grid is illustrative. Report the full sweep
  and frame conclusions as sensitivity, not a deployment recommendation.
- **SMO memory:** an explicit O(n²) kernel matrix is impossible at full n (~316 GB at 198k).
  `main_full` still attempts SMO on the full data with a 90-minute hard timeout; if it fails,
  that is a recorded finding. Use `main_capped` for the fair all-five 10k comparison; do not
  silently substitute a capped run into the full-data ranking.
- **ADMM's dense solve:** `d = 29` makes the 30×30 system cheap; do not extrapolate to
  high-dimensional data without a sparse-aware variant.
- **Fixed-rho ADMM is only the ablation control:** the main configs use adaptive rho and a
  20k iteration ceiling after the 50k preflight; the `admm-cold` ablation keeps fixed rho=1.
  Report `converged`/gap and never present an unconverged fit as a converged speed record.
- **PA-ADMM may not win:** report the component ablation and the negative result. The
  cost-aware benchmark and regime map remain the contribution.
- **Data leakage:** fit standardization, thresholds, and hyperparameters on training/validation
  only. `Time` is not a feature in the main model.
- **Hardware:** run single-threaded on the same machine for the main comparison; record CPU/RAM.
- **Missing rubric:** the Canvas instruction file was not in the downloaded course folder. If
  the report format/rubric appears, update this plan and the report.

---

## 11. TODO

- [x] Create repository and directory contract.
- [x] Download and verify ULB data: `data/raw/creditcard.csv`, 284,807 × 31, 492 frauds.
- [x] Add the proposal PDF under `docs/proposal/`.
- [x] Implement WP4 data/metric utilities, reproducible split files, evaluation harness,
      configs, experiment entry points, and the solver contract/checker.
- [ ] Implement `src/costs.py`, `src/svm_pegasos.py`, and `src/baseline_cvxpy.py`
      (WP1). WP2's `src/svm_smo.py` and WP3's `src/svm_admm.py` are merged.
      Start with `docs/INTERFACES.md` and `experiments/check_solver.py`.
- [x] ADMM preflight: main configs use adaptive rho with `max_iter=20000`; fixed-rho
      remains the `admm-cold` ablation control.
- [ ] Run `correctness` and `main_fast` (blocked on solver modules).
- [ ] Run `predictive` for validation-tuned predictive metrics; keep it separate from the
      fixed-C solver ranking.
- [ ] Freeze configs and run `main_full`.
- [ ] Produce final tables/figures; empty tables and "pending solver integration"
      placeholder figures are already committed, and no final numbers are fabricated.
