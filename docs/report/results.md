# Results (WP4 draft)

> **No final solver results yet.** The solver modules owned by WP1–WP3 are not present in this
> branch. Every table below is a template, and every figure file in `results/figures/` is a
> placeholder labelled “Pending solver integration”. No numbers have been invented or copied
> from the smoke test.

## 1. What is already verified

These are data-pipeline facts, not model results.

| Item | Value |
|---|---|
| Raw ULB file | 284,807 × 31, SHA-256 `76274b…1a89` |
| Duplicate rows removed | 1,081 |
| Cleaned rows | 283,726 |
| Cleaned positives | 473 (0.167%) |
| Stratified splits | seeds 0–4; train/val/test = 198,608 / 42,559 / 42,559 |
| Chronological split | train/val/test = 198,608 / 42,558 / 42,560; test is latest 15% |
| Chronological fraud rates | train 0.184%, validation 0.129%, test 0.122% |
| Harness tests | `pytest`: 27 passed (data, metrics, harness, failure logging, aggregation) |
| Dry-run | `run_main.py --config configs/main_fast.yaml --dry-run`: 45 planned jobs |

The harness smoke test writes a valid JSON under `results/raw/harness_smoke/` using sklearn
`SGDClassifier(loss="hinge")`. It is excluded from Git and from all tables. Its metrics are not
reported here because it is a temporary stand-in, not a result.

## 2. Main comparison (`main_full`)

Primary outcome: **time-to-tolerance** at `tol = 1e-3`, per cost ratio and seed. Secondary
outcomes: cost per example, PR-AUC, recall@FPR, balanced accuracy, iterations, memory.

| Method | Training rows | Time-to-tol (s), mean ± sd | Cost/example | PR-AUC | Recall@FPR 0.1% | Recall@FPR 1% | Balanced acc. |
|---|---:|---|---|---|---|---|---|
| SMO | ≤10,000 cap | TBD | TBD | TBD | TBD | TBD | TBD |
| Pegasos | full | TBD | TBD | TBD | TBD | TBD | TBD |
| ADMM | full | TBD | TBD | TBD | TBD | TBD | TBD |
| PA-ADMM | full | TBD | TBD | TBD | TBD | TBD | TBD |
| CVXPY | ≤20,000 cap | TBD | TBD | TBD | TBD | TBD | TBD |

The final table will be generated from
`results/tables/main_full_summary.csv` and
`results/tables/main_full_time_to_tolerance_sec.csv` after `configs/main_full.yaml` runs.
Mean ± sd, bootstrap 95% CI, and `n_seeds` must come from those files, not from manual copying.

## 3. Fast comparison (`main_fast`)

| Method | Training rows | Time-to-tol (s) @ ratio 10 | @ ratio 20 | @ ratio 50 |
|---|---:|---|---|---|
| SMO | 10,000 cap | TBD | TBD | TBD |
| Pegasos | 50,000 | TBD | TBD | TBD |
| ADMM | 50,000 | TBD | TBD | TBD |
| PA-ADMM | 50,000 | TBD | TBD | TBD |
| CVXPY | 10,000 cap | TBD | TBD | TBD |

Source: `results/tables/main_fast_*.csv` (3 seeds). This table is for debugging and for a
preliminary ranking only; `main_full` is the final comparison.

## 4. Cost-ratio sweep (`cost_sweep`)

Ratios: 1, 3, 5, 10, 20, 50. Methods: Pegasos, ADMM, PA-ADMM. Data: 50k subset, seed 0.

| Ratio | Pegasos cost/example | ADMM cost/example | PA-ADMM cost/example | Pegasos t-tol | ADMM t-tol | PA-ADMM t-tol |
|---:|---|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD | TBD | TBD | TBD |
| 10 | TBD | TBD | TBD | TBD | TBD | TBD |
| 20 | TBD | TBD | TBD | TBD | TBD | TBD |
| 50 | TBD | TBD | TBD | TBD | TBD | TBD |

Source: `results/tables/cost_sweep_cost_per_example.csv` and
`results/tables/cost_sweep_time_to_tolerance_sec.csv`.
Interpretation must treat the cost ratios as illustrative assumptions, not a bank loss model.

## 5. Stress tests (`stress`)

Each row is a separate dataset variant; validation and test labels are unchanged. Ratio 20:1,
3 seeds. SMO is included only in the label-noise variants (10k cap). Prevalence targets below
the natural 0.167% use training-positive subsampling; higher targets use training-negative
subsampling, so their training sets are smaller. Report `prevalence_mechanism` and `n_train`
with every stress result.

| Variant | Method | Cost/example | PR-AUC | Recall@FPR 0.1% | Time-to-tol |
|---|---|---|---:|---:|---|
| fraud rate 0.1% | Pegasos / ADMM / PA-ADMM | TBD | TBD | TBD | TBD |
| fraud rate 0.5% | Pegasos / ADMM / PA-ADMM | TBD | TBD | TBD | TBD |
| fraud rate 1% | Pegasos / ADMM / PA-ADMM | TBD | TBD | TBD | TBD |
| fraud rate 5% | Pegasos / ADMM / PA-ADMM | TBD | TBD | TBD | TBD |
| fraud rate 10% | Pegasos / ADMM / PA-ADMM | TBD | TBD | TBD | TBD |
| label noise 5% | SMO / Pegasos / ADMM / PA-ADMM | TBD | TBD | TBD | TBD |
| label noise 10% | SMO / Pegasos / ADMM / PA-ADMM | TBD | TBD | TBD | TBD |
| chronological | Pegasos / ADMM / PA-ADMM | TBD | TBD | TBD | TBD |

Source: `results/tables/stress_*.csv` (dataset IDs distinguish variants). The chronological
variant has a lower test fraud rate (0.122%) than training (0.184%); report the drift and do not
pool it with the stratified results.

## 6. Ablation (`ablation`)

Full data, ratio 20:1, 3 seeds, fixed `C=1` for component isolation.

| Variant | Warm start K | Adaptive ρ | α | ρ0 | Time-to-tol | Iterations |
|---|---:|---|---:|---:|---|---|
| `admm-cold` | 0 | off | 1.0 | 1.0 | TBD | TBD |
| `pa-K0` | 0 | on | 1.6 | 1.0 | TBD | TBD |
| `pa-K5` | 5 | on | 1.6 | 1.0 | TBD | TBD |
| `pa-K10` | 10 | on | 1.6 | 1.0 | TBD | TBD |
| `pa-K20` | 20 | on | 1.6 | 1.0 | TBD | TBD |
| `pa-adaptive-off` | 10 | off | 1.6 | 1.0 | TBD | TBD |
| `pa-alpha-1.0` | 10 | on | 1.0 | 1.0 | TBD | TBD |
| `pa-alpha-1.5` | 10 | on | 1.5 | 1.0 | TBD | TBD |
| `pa-alpha-1.8` | 10 | on | 1.8 | 1.0 | TBD | TBD |
| `pa-rho0-0.1` | 10 | on | 1.6 | 0.1 | TBD | TBD |
| `pa-rho0-10` | 10 | on | 1.6 | 10.0 | TBD | TBD |

Source: `results/tables/ablation_time_to_tolerance_sec.csv` and
`results/tables/ablation_n_iter.csv`. Report component effects even if PA-ADMM does not win.

## 7. Scaling (`scaling`)

Synthetic data, `d=20`, fraud rate 1%, ratio 10:1, fixed `C=1`, 3 seeds.

| n | Pegasos time / memory | ADMM time / memory | PA-ADMM time / memory |
|---:|---|---|---|
| 10,000 | TBD | TBD | TBD |
| 25,000 | TBD | TBD | TBD |
| 50,000 | TBD | TBD | TBD |
| 100,000 | TBD | TBD | TBD |
| 200,000 | TBD | TBD | TBD |

Source: `results/tables/scaling_fit_time_sec.csv` and
`results/tables/scaling_peak_memory_mb.csv`. Wall-clock and process-level memory are
hardware-dependent; report the run environment with them.

## 8. Figures

All files below currently contain the placeholder text “Pending solver integration”.
`experiments/make_report_assets.py` regenerates them from tables/run JSONs once results exist.

| Figure | File | Status |
|---|---|---|
| Time-to-tolerance, main full | `results/figures/main_full_time_to_tolerance.pdf` | placeholder |
| Objective vs time | `results/figures/main_full_convergence.pdf` | placeholder |
| Test cost vs cost ratio | `results/figures/main_full_cost_vs_ratio.pdf` | placeholder |
| PR curves | `results/figures/main_full_pr_curves.pdf` | placeholder |
| Recall at FPR | `results/figures/main_full_recall_at_fpr.pdf` | placeholder |
| Cost sweep | `results/figures/cost_sweep_cost_vs_ratio.pdf` | placeholder |
| ρ/component sensitivity | `results/figures/ablation_rho_sensitivity.pdf` | placeholder |
| Stress cost | `results/figures/stress_cost_under_shift.pdf` | placeholder |
| Scaling runtime | `results/figures/scaling_runtime_vs_n.pdf` | placeholder |
| Scaling memory | `results/figures/scaling_memory_vs_n.pdf` | placeholder |

## 9. How to fill these tables

1. Wait for the WP1–WP3 solver modules and `src/costs.py`.
2. Run, in order: `correctness`, `main_fast`, then freeze and run `main_full`, `cost_sweep`,
   `ablation`, `stress`, `scaling` using the committed YAMLs.
3. Run `python experiments/make_report_assets.py` to regenerate tables/figures from
   `results/raw/` and `results/tables/`.
4. Replace every `TBD` in this document with values read from the generated CSVs, including
   seed count and uncertainty. Do not hand-edit raw run JSONs.

Until then, this section must not be cited as evidence for any hypothesis.
