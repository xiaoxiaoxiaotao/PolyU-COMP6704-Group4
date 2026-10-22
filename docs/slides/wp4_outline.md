# WP4 slides outline — Experiments, Results, Discussion, Conclusion

Target: **10 minutes** (part of the 40-minute group presentation). 10 content slides +
2 backup slides. All result numbers must be read from `results/tables/` after the final runs;
the current deck should use the “pending solver integration” wording, not placeholder numbers.

---

## Slide 1 — WP4 roadmap (0:30)

**Title:** From costs to a fair algorithm comparison

- WP4 owns: data pipeline → splits → metrics → harness → experiment matrix → tables/figures.
- Does not own: solvers (WP1–WP3) or the canonical cost object (`src/costs.py`, WP1).
- One application: ULB credit card fraud; synthetic data only as controlled stress/scaling tests.

**Speaker note:** State up front that the evaluation is about optimization speed *and*
application cost, not accuracy.

---

## Slide 2 — Problem in one slide (0:45)

**Title:** Why a cost-sensitive, time-aware evaluation?

- 0.172% fraud (492/284,807 raw; 473/283,726 after deduplication).
- Missed fraud ≠ false alarm: `c_FN/c_FP` from 1 to 50.
- Accuracy is meaningless; a “always legitimate” rule gets 99.8%.
- The model is a weighted soft-margin SVM: `C_i = C·w_{y_i}`, `w_+/w_- = c_FN/c_FP`.
- Primary metric: **time-to-tolerance**, not final accuracy.

**Visual:** one small cost-matrix schematic and the imbalance numbers.

---

## Slide 3 — Data pipeline and leakage controls (1:00)

**Title:** One dataset, auditable splits

- Raw 284,807 × 31; 1,081 duplicates removed → 283,726 × 31; 473 frauds.
- Features: `V1`–`V28` + `Amount` (30). `Time` is **not** a feature.
- `Amount` standardized with training-fold mean/std only; `V1`–`V28` unchanged.
- Labels: `y ∈ {-1,+1}`.
- Stratified 70/15/15, seeds 0–4; chronological 70/15/15, seed 0.
- Split JSONs are committed; indices refer to the deduplicated frame.

**Visual:** table of split sizes/counts (verified numbers from `docs/report/results.md` §1).
Call out chronological drift: test fraud rate 0.122% vs train 0.184%.

---

## Slide 4 — Evaluation protocol (1:15)

**Title:** Validation chooses; test reports once

- For each `C` in the configured grid: fit → validation cost-minimizing threshold →
  validation cost.
- Keep the best `C`; evaluate test once with the frozen threshold.
- `C_i` normalized to mean weight one; class ratio preserved.
- Objective convention: `0.5||w||² + Σ C_i hinge`, recomputed from `w,b` to check scale.
- Reference objective: CVXPY for correctness; best observed within the same
  `(dataset, ratio, seed, n_train)` group otherwise.
- Solver caps: SMO ≤ 10k, CVXPY ≤ 20k; caps recorded and not mixed in reference groups.

**Visual:** a simple flow `train → validation threshold/C → freeze → test metrics`.
Emphasize: **never tune on test**.

---

## Slide 5 — Metrics and statistics (0:45)

**Title:** Three lenses on the same run

- Primary: `time_to_tolerance` at `1e-3` (main) / `1e-4` (correctness).
- Application: cost per example `(c_FN·FN + c_FP·FP)/n_test`.
- Ranking: PR-AUC, recall@FPR 0.1% and 1%, balanced accuracy.
- Engineering: iterations, fit time, tuning time, process peak memory.
- Across seeds: mean ± sd, bootstrap 95% CI, paired Wilcoxon signed-rank.
- Failures/timeouts are logged, never silently dropped.

**Speaker note:** time-to-tolerance is only reported when the solver's objective history is
consistent with the recomputed primal objective; otherwise the run is flagged.

---

## Slide 6 — Experiment matrix (0:45)

**Title:** Seven experiments, one protocol

| Experiment | Data | Ratios | Seeds | Jobs |
|---|---|---|---:|---:|
| correctness | 10k subset | 10 | 1 | 4 |
| main_fast | 50k subset | 10/20/50 | 3 | 45 |
| main_full | full ULB | 10/20/50 | 5 | 75 |
| cost_sweep | 50k subset | 1…50 | 1 | 18 |
| ablation | full ULB | 20 | 3 | 33 |
| stress | 5 imbalance + 2 noise + chronological | 20 | 3 | 78 |
| scaling | synthetic 10k–200k, d=20 | 10 | 3 | 45 |

- Ablation: `K ∈ {0,5,10,20}`, adaptive ρ on/off, `α ∈ {1.0,1.5,1.8}`, `ρ0 ∈ {0.1,1,10}`.
- All settings live in `configs/*.yaml`; `--dry-run` prints the plan.

---

## Slide 7 — Harness status and current evidence (0:30)

**Title:** What is implemented, what is pending

- Implemented and tested: data/splits/metrics/harness/configs/entry points; 27 pytest cases.
- Verified: dry-run `main_fast` = 45 jobs; split files reproduce disjoint covers.
- Pending: WP1–WP3 solver modules and `src/costs.py`.
- The harness refuses to run missing methods and writes no fake result.
- Smoke test uses sklearn SGD hinge only to validate JSON plumbing; **not a result**.

**Visual:** a status checklist (implemented / pending). No metric numbers from smoke.

---

## Slide 8 — Main comparison results (1:30) — **TBD after final runs**

**Title:** Time-to-tolerance and test cost

- One chart: time-to-tolerance by method/ratio (from
  `results/figures/main_full_time_to_tolerance.pdf`).
- One table: cost/example, PR-AUC, recall@FPR, balanced accuracy
  (from `results/tables/main_full_*.csv`).
- State seed count, mean ± sd or CI, and whether tolerance was reached.
- Do not compare capped SMO/CVXPY training objectives directly with full-data runs.

**Speaker note:** If a solver did not reach tolerance, show the final gap and mark it; a missing
bar is not zero.

---

## Slide 9 — Regime map: cost sweep, stress, ablation (1:30) — **TBD**

**Title:** Where does the ranking change?

- Left: test cost vs cost ratio (`cost_sweep_cost_vs_ratio.pdf`).
- Middle: fraud-rate / label-noise / chronological cost
  (`stress_cost_under_shift.pdf`).
- Right: PA-ADMM component ablation (`ablation_rho_sensitivity.pdf`).
- Report negative results as clearly as positive ones; the component ablation may refute
  PA-ADMM's advantage.

**Speaker note:** the chronological test block has lower fraud prevalence; do not pool it with
stratified results. For prevalence stress, note that targets above 0.167% use training-negative
subsampling (and thus smaller `n_train`), because positive-only subsampling cannot raise
prevalence.

---

## Slide 10 — Scaling and conclusion (1:00) — **TBD**

**Title:** Runtime/memory versus n, and what we learned

- Scaling plot (`scaling_runtime_vs_n.pdf`, `scaling_memory_vs_n.pdf`).
- One sentence per research question, conditional on the final tables:
  - RQ1 cost-sensitive weighting vs unweighted;
  - RQ2 solver ranking in time-to-tolerance;
  - RQ3 PA-ADMM component effects;
  - RQ4 objective → test cost under noise;
  - RQ5 regime map.
- Limitations: illustrative cost ratios, one dataset, capped SMO/CVXPY, wall-clock hardware,
  noisy labels.

---

## Backup slides

### B1 — Reproducibility commands (0:30 if asked)

```bash
python experiments/run_main.py --config configs/main_fast.yaml --dry-run
python experiments/run_correctness.py --config configs/correctness.yaml
python experiments/run_main.py --config configs/main_full.yaml
python experiments/run_cost_sweep.py --config configs/cost_sweep.yaml
python experiments/run_ablation.py --config configs/ablation.yaml
python experiments/run_stress.py --config configs/stress.yaml
python experiments/run_scaling.py --config configs/scaling.yaml
python experiments/make_report_assets.py
```

### B2 — Protocol defenses (0:30 if asked)

- Same split file, cost ratio, tolerance, budget, hardware, single-threaded BLAS.
- Threshold and `C` on validation only; test evaluated once.
- Objectives recomputed from `w,b`; inconsistent solver histories flagged.
- Failures/timeouts recorded as JSON; missing modules produce no results.
- Split files committed; raw CSV ignored by Git.

---

## Timing check

| Slide | Time |
|---|---:|
| 1 | 0:30 |
| 2 | 0:45 |
| 3 | 1:00 |
| 4 | 1:15 |
| 5 | 0:45 |
| 6 | 0:45 |
| 7 | 0:30 |
| 8 | 1:30 |
| 9 | 1:30 |
| 10 | 1:00 |
| **Total** | **9:30** + 0:30 buffer = **10:00** |

## Rules for the final deck

1. Every number comes from `results/tables/` (or verified split metadata).
2. Never quote the smoke-test metrics.
3. If a result is pending, say “pending solver integration” rather than showing a plausible
   number.
4. Put full tables and extra seeds in backup, not on the main slides.
