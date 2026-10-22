# Discussion and Conclusion (WP4 draft)

> Status: this is an interpretation plan, not a results narrative. The solver modules are still
> pending, so no empirical claim below has been confirmed. Text that depends on final outcomes
> is marked **TBD after final runs**.

## 1. What the evaluation is designed to separate

The project compares optimization methods, not just classifiers. Three quantities must be kept
apart:

1. **Optimization efficiency** — wall-clock seconds to a relative weighted-objective gap
   (`time_to_tolerance`), with iteration counts reported alongside because cheap iterations can
   hide slow convergence.
2. **Statistical quality under imbalance** — PR-AUC, recall at FPR 0.1% and 1%, and balanced
   accuracy at a validation-selected threshold.
3. **Application cost** — `(c_FN·FN + c_FP·FP)/n_test` under the assumed cost ratio.

A method can win on (1) and lose on (3), or match on (2) while producing a very different
operating point. The report must present all three rather than collapse them into one ranking.
Because the cost ratios are assumptions, the sweep is a sensitivity analysis, not a deployment
recommendation.

## 2. Expected trade-offs to test (hypotheses, not findings)

The proposal lists five hypotheses. The evaluation maps to them as follows:

### H1 — cost-sensitive weighting reduces total cost

Compare weighted runs at ratios 10/20/50 with the unweighted ratio-1 run in `cost_sweep`, and
compare validation-selected operating points at fixed FPR/recall. **What would support H1:** a
consistent reduction in test cost for larger ratios, especially at lower fraud rates, without a
collapse in PR-AUC. **What would complicate it:** lower training objective but higher test cost
under label noise, or a threshold shift that only moves errors between FN and FP.

The harness does not tune on test, so any cost improvement must arise from the weighted training
objective plus a validation-selected threshold.

### H2 — PA-ADMM is faster than cold-start ADMM

The primary evidence is the paired difference in `time_to_tolerance` for ADMM and PA-ADMM on
the same split, ratio, seed, tolerance, and budget. A significant Wilcoxon result is necessary
but not sufficient: report effect size and the fraction of seeds where tolerance was reached at
all. **What would support H2:** lower median time-to-tolerance and fewer non-converged /
budget-exceeded runs. **What would refute it:** comparable or better cold-start ADMM, or a
PA-ADMM advantage that disappears once the warm-start time is counted (it is counted inside
`fit_time_sec` in this harness).

### H3 — adaptive ρ and over-relaxation help, and warm start contributes most

The ablation separates `K ∈ {0,5,10,20}`, adaptive ρ on/off, `α ∈ {1.0,1.5,1.8}`, and
`ρ0 ∈ {0.1,1,10}`. Component claims should be based on paired runs across seeds and should
report both time-to-tolerance and iterations. A smaller `ρ0` sensitivity with adaptive ρ on is
evidence for that component; a flat `K` curve is a legitimate negative result.

### H4 — Pegasos scales, SMO is strong on subsets, PA-ADMM is competitive at tolerance

The scaling experiment measures runtime/memory versus `n` on synthetic data; `main_full`
measures the capped/full-data comparison on ULB. SMO's cap (10k) is itself a finding: O(n²)
kernel memory makes full-data SMO infeasible, and the report should show the memory wall rather
than hide the cap. **What would support H4:** near-linear Pegasos scaling, competitive PA-ADMM
time-to-tolerance at large `n`, and SMO's advantage only on small subsets.

### H5 — no universal winner; the ranking is regime-dependent

The regime map combines cost ratio, fraud rate, label noise, chronological drift, and `n`.
**What would support H5:** crossover interactions (for example, a dual method winning on small
_n_ or low ratios and a primal/stochastic method winning on large _n_ or high ratios) that are
stable across seeds. A single aggregate leaderboard would then be misleading; the report should
show slices, not only global averages.

## 3. Reading the objective–cost relationship

A lower weighted training objective is not automatically a lower test cost. Three mechanisms
can break the link:

- **Overfitting the weighted objective:** high `C` and high positive weight can fit noisy
  labels, shifting the validation threshold in a way that raises test cost.
- **Distribution drift:** the chronological test block has a lower fraud prevalence than
  training (0.122% vs 0.184%), so a threshold calibrated on a validation block from the same
  period may over-alert later.
- **Threshold selection noise:** validation has few positives, so the cost-minimizing threshold
  can move substantially across seeds. Bootstrap intervals and per-seed operating points should
  accompany any single threshold.

The harness records validation and test cost separately, plus the selected `C` and threshold,
which makes these mechanisms diagnosable rather than speculative.

## 4. Limitations and threats to validity

### Data and labels

- One public, anonymized dataset. `V1`–`V28` are PCA components, so feature-level
  interpretation is limited.
- The natural cleaned prevalence is 0.167%, so prevalence targets above it cannot be reached
  by positive subsampling. Those variants subsample training negatives instead, which shrinks
  `n_train`; any runtime difference across prevalence levels is therefore confounded with
  training-set size. The mechanism is recorded per run, and the report should present cost and
  PR metrics alongside timing.
- Fraud labels are incomplete: some fraud is never reported, which is a form of label noise
  that cannot be removed by the training-only flip experiments.
- Duplicate removal changes the positive count from 492 to 473; all reported rates use the
  cleaned frame. This is documented to avoid mixing raw and cleaned denominators.

### Cost model

- The cost ratios 1:1 to 50:1 are illustrative. They encode an assumption about issuer losses
  and customer friction, not a measured bank cost matrix.
- The harness uses a mean-one class-weight normalization. Results are conditional on that
  convention; WP1 owns the canonical cost model and should confirm equivalence.
- Cost per example is an average. It hides the distribution of transaction amounts and the
  difference between a missed fraud and a declined legitimate purchase.

### Optimization protocol

- SMO is capped at 10k and CVXPY at 20k because of memory/solver limits. Capped and full-data
  runs are separated in reference-objective groups; the report must not compare their raw
  objectives as if the training problems were identical.
- Hyperparameter search multiplies wall-clock effort. The harness reports tuning time
  separately, but a method that is cheap per fit may still be expensive after the full `C` grid.
- Time-to-tolerance depends on solver-reported iteration timestamps. If a solver does not log
  them, the harness reports a null time rather than fabricating a schedule. Objective histories
  whose final value disagrees with the recomputed primal objective are flagged and excluded
  from timing claims.
- In-process time budgets require solver cooperation. A solver that ignores the budget can
  overshoot before the harness records a timeout.
- `peak_memory_mb` is process-level maximum RSS, not per-fit allocation. It is comparable across
  sequential runs but not an exact solver-memory measurement.
- Wall-clock results are machine-specific and single-threaded by configuration. They should be
  interpreted comparatively, on the same machine, not as absolute deployment latencies.

### Evaluation

- Extreme imbalance makes PR-AUC and recall@FPR noisy. Five seeds for `main_full`, bootstrap
  intervals, and paired tests reduce but do not eliminate this.
- The chronological split confounds algorithm behaviour with temporal drift. It is a robustness
  check, not an unbiased estimate of deployment performance.
- Synthetic scaling tests use a simple Gaussian generator. They test computational scaling, not
  whether the fraud geometry matches real data.

## 5. WP4 implementation lessons

- **Splits must be an artifact, not a side effect.** The committed index files make the same
  rows available to every method and make it possible to audit a run without the raw CSV.
- **The threshold is part of the model.** Selecting it on validation and freezing it before
  test is the only defensible protocol at this imbalance level. The harness logs the threshold
  with every run so a cost difference can be traced to the operating point.
- **Lazy imports are the right boundary.** WP4 can test the whole pipeline with a temporary
  sklearn stand-in while WP1–WP3 modules are absent, and the harness refuses to write results
  for missing methods instead of silently substituting a different algorithm.
- **Placeholders are honest.** Empty tables and “Pending solver integration” figures prevent
  accidental citation of smoke-test numbers as final results.

## 6. Conclusion (to be completed after final runs)

The project's contribution will be a unified, cost-aware comparison of SMO, Pegasos, fixed-ρ
ADMM, PA-ADMM, and an interior-point baseline on the ULB fraud problem, with reproducible
splits and a validation-only threshold protocol. The conclusion must state, from
`results/tables/` only:

- whether cost-sensitive weighting reduced test cost at the assumed ratios (H1);
- which methods reached tolerance and at what wall-clock cost (H2, H4);
- which PA-ADMM components mattered, including negative results (H3);
- where the regime map changes the ranking (H5);
- whether a lower training objective translated into lower test cost, especially under label
  noise and temporal drift.

**TBD after final runs:** the actual ordering of methods, the magnitude and uncertainty of any
time-to-tolerance advantage, and the cost-ratio regions where the ranking changes. Until those
numbers exist, no such claim should appear in the report or slides.
