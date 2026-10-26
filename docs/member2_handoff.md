# Member 2 SMO handoff

## Scope and provenance

Member 2 owns cost-sensitive SMO, its correctness tests, a small RBF correctness check, and the SMO method section. WP1 owns the cost model, convex formulation, Pegasos, and CVXPY baseline. WP3 owns ADMM and PA-ADMM. WP4 owns preprocessing, splits, tuning, metrics, the unified evaluator, and the unified experimental presentation. Member 2's dedicated runner verifies SMO against LIBSVM; it does not replace WP4's benchmark.

The initial implementation used upstream files exported from GitHub Code views at main commit `ac49e434d8321459420f9176a72e688946307487`. That historical local workspace was a browser export, not a Git clone; its export manifests were retained locally. For this code release, the six Member 2 files were checked in an isolated snapshot of current main `401dce6e2ca89738b094f4c1ad728042ac904eae`.

Upstream `docs/INTERFACES.md` and `EXPERIMENTS.md` are not modified by this release. This note records the current integration checks separately from the historical numerical study.

## Code release validation

Against main `401dce6e2ca89738b094f4c1ad728042ac904eae`, `python -m pytest -q tests` completed with **140 passed in 3.73 s**, using the existing local virtual environment. No dependencies were installed and the historical real-data experiment was not rerun. The code release contains the SMO implementation, its tests, the dedicated correctness runner and configuration, and this handoff note; it does not contain the historical `results/member2/` or `outputs/member2/` attachments cited below.

With the project environment active, run these lightweight checks from the repository root:

```bash
python -m pytest -q tests/test_smo.py tests/test_member2_experiment.py
python experiments/check_solver.py --method smo --n 500 --d 29 --ratio 10 --budget 30
python experiments/run_member2.py --config configs/member2_correctness.yaml --dry-run
```

The contract check uses synthetic data and writes a diagnostic JSON under `results/raw/solver_checks/`. The dry run only validates the configuration and lists the ten planned SMO–LIBSVM pairs; it starts no fits or downloads. Running the real-data configuration without `--dry-run` is a separate, budget-approved correctness experiment. WP4 remains responsible for the unified experiment campaign and results presentation. Follow the repository's requirement for review by one other WP before merge.

## Integration contract

The registry already maps `smo` to `src.svm_smo.SMO`. Construction requires no arguments. The public call is:

```python
model = SMO()
model.fit(X, y, C_per_sample, seed=0, tol=1e-4,
          time_budget_sec=..., kernel="linear", ...)
scores = model.decision_function(X_new)
labels = model.predict(X_new)
```

The `...` symbols above indicate configured values, not a runnable command. Use the checked experiment scripts and their documented arguments for execution.

| Item | Contract |
|---|---|
| `X` | Finite `float64`, shape `(n, d)`; current ULB linear protocol uses 29 features |
| `y` | Shape `(n,)`, labels `-1` legitimate and `+1` fraud; both classes present |
| `C_per_sample` | Finite strictly positive vector, shape `(n,)`, already cost weighted |
| `decision_function` | Finite raw scores, shape `(n,)` |
| `predict` | Labels in `{-1,+1}`, zero score maps to `+1` |
| Linear state | `w`, `b`, `alphas`, `n_iter`, `converged`, `objective` |
| Trace | Aligned `objective_history` and `time_history`, plus KKT and duality diagnostics |

Fit uses training data only. The harness selects C and the threshold with validation data, then evaluates the selected model on test. Solvers must not standardize the data again, choose a validation threshold, or read held-out labels.

## Cost and objective conventions

The solver minimizes the unnormalized primal objective

\[
P(w,b)=\tfrac12\|w\|^2+\sum_i C_i\max(0,1-y_i(x_i^Tw+b)).
\]

Use `C_per_sample[i]` directly as the dual bound. Do not add class weights again or divide the objective by n.

The current evaluator normalizes raw class-cost weights to **sample mean one**. The proposal instead normalizes the mean of the two class weights to one. For a fixed training partition, these conventions differ by a positive global scalar. A compensating change in C can express the same objective, but an unchanged finite C grid need not contain exactly matching values. Preserve the existing interface and record `costs_source`; coordinate the final convention with WP1/WP4 instead of compensating inside SMO.

## Implemented algorithm and release checks

1. The dual constraints are `0 <= alpha_i <= C_i` and `y.T @ alpha = 0`. Pair updates preserve the equality exactly in algebra and to a stated floating-point tolerance in execution.
2. For fixed old values a_i and a_j, update t = alpha_j within:

   | Labels | L | H |
   |---|---|---|
   | Different | `max(0, a_j-a_i)` | `min(C_j, C_i+a_j-a_i)` |
   | Equal | `max(0, a_i+a_j-C_i)` | `min(C_j, a_i+a_j)` |

   Recover `alpha_i = a_i + y_i*y_j*(a_j-t)`. Near-zero curvature requires an explicit endpoint or no-move treatment.
3. The working-set design chooses a violating first coordinate and selects the partner by the **actual gain after clipping** the pair step. Describe this as a max clipped gain selector. Do not label it as the original Fan WSS2 algorithm or claim its published convergence theorem applies unchanged.
4. The final design requires a feasible iterate, working-set KKT gap `<= tol`, and relative primal-dual gap `<= gap_tol`. `gap_tol=None` defaults to `tol`. The relative gap is `(P-D)/max(1, abs(P), abs(D))`. Material negative gaps signal an error; they must not be silently hidden.
5. Budget or iteration exit returns the lowest-primal-objective state among measured checkpoints. Restore all associated parameters and synchronize the last history entry with the returned model. A convergence exit uses the current certified state. Report `converged=False` on an unfinished solve and retain the stopping reason.
6. Keep elapsed times measured from the start of `fit`, including kernel/cache preparation. Use at least two genuine aligned checkpoints for a time-to-tolerance trace. Do not fabricate regular timestamps.

These decisions match the current `src/svm_smo.py` inspected during this task. `relative_duality_gap`, `dual_feasible`, `returned_best_checkpoint`, `returned_checkpoint_index`, and `returned_checkpoint_iteration` make the stopping and return state auditable. The completed numerical study is recorded below.

## Fair comparison and data limits

- Use the same training rows, labels, C_per_sample, kernel, and kernel parameters for objective comparisons. A fixed C grid with a single value avoids comparing different objectives after independent validation tuning.
- Historical background: the original export did not group reference objectives by selected C. Current main fixes `C=1` for solver-ranking configurations and includes selected C in reference and objective-dependent aggregation/pairwise comparisons (`EXPERIMENTS.md`, `src/evaluate.py`). Kernel and gamma are still not reference-group keys: keep RBF in the separate kernel-aware correctness check and do not mix linear and RBF objectives in a main comparison.
- `main_capped.yaml` sets `data.subset_n: 10000` before the 70/15/15 split. The completed study has 7,000/1,500/1,500 train/validation/test rows, with 12/2/3 frauds, respectively. Several upstream comments call this a “10k training subset”; that is not the measured training count.
- At the natural fraud prevalence, a small stratified subset has very few positive samples in validation/test. Recall changes in large steps, confidence intervals can be wide, and small differences cannot support strong generalization claims.
- Existing committed PDF figures and CSV tables are placeholders. `EXPERIMENTS.md` contains an illustrative JSON record with sample numbers and an outdated `d=30`; those values are not results.
- Current features are `V1` through `V28` and `Amount`. `Time` is excluded from the model and used for chronological splitting only.
- The original memory monitor reads Linux `/proc/self/status`. A missing memory value on macOS is not zero memory usage.

## LIBSVM and RBF verification

LIBSVM remains a correctness reference rather than a new main-ranking solver. With scikit-learn `SVC`, the study uses `C=1.0` and `sample_weight=C_per_sample`, with no additional class weighting, reference `tol=1e-7`, `shrinking=False`, and probability fitting disabled. Each pair uses the same kernel and resolved gamma. The RBF case resolves `gamma="scale"` to `0.03273701687893006` on its prepared training data.

Compare the recomputed primal objective, dual feasibility, KKT and duality diagnostics, and decision outputs. Equal accuracy alone is insufficient. Report score or parameter differences with tolerance because non-unique solutions and tie handling can differ.

For RBF, use `v = alpha * y`, `||w||_H^2 = v.T @ K @ v`, and `f = K @ v + b`. The existing linear checker assumes a d-dimensional w. Use a kernel-aware check and separate output path rather than creating a fake linear w.

## Verification and result record

This section preserves the earlier local study, not a rerun on the code-release base. All `results/member2/` and `outputs/member2/` paths below refer to historical local delivery attachments, which are excluded from this code release. In particular, the 105-test record below belongs to that earlier snapshot; current integration validation is recorded above.

The bounded ULB study completed all ten SMO–LIBSVM pairs without timeouts. Nine linear cases cover cost ratios 10/20/50 and seeds 0/1/2; the separate RBF case uses ratio 10 and seed 0. All cases fix `C=1`, use 29 features, and retain the harness's sample-mean cost normalization. The same prepared arrays and C_per_sample are used within each pair. Seeds select different stratified partitions of the fixed 10k subset; these are not ten independent datasets.

| Evidence | Record | Result |
|---|---|---|
| Unit tests | `.venv/bin/python -m pytest -q tests` | 105 passed in 2.00 s, including 49 SMO tests; recorded in `results/member2/validation_summary.json` |
| Contract check | `results/member2/final_solver_contract.json` | `passed=true`, no errors/warnings, configuration n=500, d=29; P=`24.3670379844`, 2,656 updates, 30 checkpoints |
| Linear SMO | Nine records in `results/member2/raw/` | 9/9 satisfy feasibility, KKT and relative duality-gap certificates |
| Linear LIBSVM references | Same nine records | 6/9 satisfy the same certificates; largest relative objective difference for those six is `7.99556e-5` |
| RBF SMO and LIBSVM | `ulb-n10000-rbf-ratio10-seed0-C1.json` | Both certified; relative objective difference `8.67914e-5` |
| Convergence figure | `outputs/member2/figures/smo_convergence.pdf` and `.png` | All recorded checkpoints, unsmoothed, seed 0 linear cases |
| Summary | `outputs/member2/results_summary.json` | Derived from recorded diagnostics, with uncertified reference cases identified |

All ten SMO fits report `converged=True` and `stop_reason="converged"`. Across them, maximum working-set KKT gap is `9.86885e-5`, maximum relative primal-dual gap `9.97231e-5`, maximum equality violation `2.13163e-14`, and maximum box violation zero. The certificate thresholds are KKT/relative gap `1e-4` and box/equality `1e-8`.

Three linear LIBSVM references exceed the relative primal-dual threshold despite normal library termination:

| Ratio | Seed | LIBSVM relative primal-dual gap |
|---:|---:|---:|
| 20 | 0 | `1.11921e-4` |
| 50 | 0 | `2.47200e-4` |
| 50 | 1 | `1.10675e-4` |

These references are **uncertified at the requested precision**, not evidence that SMO failed. Their objective values and fit records are retained. Report the result as “10 SMO certificates; 7 certified references”; do not reuse the preliminary “9 passed / 1 failed” comparison summary. Objective difference is `abs(P_SMO-P_LIBSVM)/abs(P_LIBSVM)` and is interpreted as a certified reference check only when both models meet the certificate.

The Apple M5 single-thread campaign finished in approximately 16.96 s of campaign wall time. SMO fit times range from `0.0521421` to `0.8735000` s. These are different timing quantities. The records do not establish a speed ranking against Pegasos, ADMM, or PA-ADMM. Validation and test contain only 2 and 3 frauds; numerical certification does not establish generalization superiority.

The contract report's objective consistency relative error is approximately `1.46e-16`. This verifies numerical agreement within that synthetic check; it is not a formal ULB result. A high-penalty regression also shows why the duality-gap gate is needed: the KKT-only candidate at update 9 has relative primal-dual gap about `5.29e-4`, while update 10 reaches about `1.16e-5`. Keep this example labeled as a controlled regression.

The study used the user's approved download and bounded 30-minute single-thread CPU budget. No GPU experiment or full-dataset comparison is claimed.

### Source provenance after the run

Each raw record preserves the SHA256 values of the source files that actually produced its fits, including the original SMO hash `093bf31339380e98f84842bbab972f900d91945e4a7995304cc208e8fda42676`. The final source subsequently received an RBF iteration-limit boundary fix; all ten recorded fits used the normal convergence exit, so these data are not evidence of executing the repaired boundary path. That path is covered by targeted regression tests.

Reference-certification postprocessing reassessed the saved diagnostics without rerunning either fit. `results/member2/validation_summary.json` records `formal_refits_after_batch=0`, the original source hashes, and preservation checks for every fit/history payload. Each raw record also retains `comparison_before_reference_certificate_check` and `comparison_postprocessing`. The final solver SHA256 is `2102981f2516532818da93a1ecd488697ee612a9ef673789c3cf6c795c2b67d9`; this is separate from the preserved original fit hash. The historical browser-export manifest was not a Git commit and does not identify this code release.

## Report and presentation handoff

Use the narrative WP1 convex formulation and Pegasos, then Member 2 SMO, then WP3 ADMM/PA-ADMM, then WP4 unified evaluation. PA-ADMM combines a Pegasos warm start with an adaptive penalty, not penalty adaptation alone.

The course talk is 30 minutes plus 10 minutes of Q&A. Allocate about 7 minutes to Member 2. Continue the full deck's numbering and style without another cover or repeated dataset introduction.

Opening bridge:

> We have defined the cost-sensitive objective and its optimality conditions. Pegasos works with the primal variables. I will now show how SMO solves the same model through feasible updates of the dual variables.

Closing bridge:

> SMO preserves the dual constraints through pair updates. The next section studies ADMM and then PA-ADMM, which adds a Pegasos warm start and adaptive penalty. The final section compares the recorded runs under our shared evaluation protocol.

The earlier local English report and slides used the numerical study above. For the group presentation, Member 2 covers the method and its correctness checks; WP4 owns the unified experiments and results presentation. Historical numerical plots and detailed LIBSVM/RBF comparisons can be handed to WP4 or retained as backup material. Preserve the three uncertified references and the small-positive-count limitation whenever these results are used. Member 2's deliverables do not establish that WP1/WP3 implementations or the complete group benchmark are finished.

## Primary references

- [Platt 1998, Sequential Minimal Optimization](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/): classical pair-decomposition basis.
- [Fan, Chen and Lin 2005, Working Set Selection Using Second Order Information](https://jmlr.org/papers/v6/fan05a.html): related working-set literature; its WSS2 convergence theorem is not asserted for our max clipped gain selector.
- [scikit-learn SVC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html): `sample_weight` rescales C per sample, supporting the matched-bound LIBSVM configuration.
