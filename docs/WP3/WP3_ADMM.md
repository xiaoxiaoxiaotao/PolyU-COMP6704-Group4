# WP3 — ADMM and PA-ADMM handoff

Owner: Member 3. Implementation: `src/svm_admm.py`; numerical and integration
tests: `tests/test_admm.py`. Both registered classes have no required constructor
arguments and work with the existing WP4 registry. WP1/WP2 solver modules,
cost normalization, data preprocessing, and committed splits are unchanged.

## Model and linear system

Use the supplied **already weighted** `C_per_sample` directly:

\[
 P(w,b)=\tfrac12\lVert w\rVert^2+
 \sum_i C_i[1-y_i(x_i^T w+b)]_+.
\]

Let \(H=[X,\mathbf1]\), \(A=\operatorname{diag}(y)H\),
\(\theta=(w,b)\), and \(R=\operatorname{diag}(1,\ldots,1,0)\).
Split with \(z=A\theta\). The final zero in \(R\) is essential: **the intercept
is not regularized**. With the constraint written as \(z-A\theta=0\), use
unscaled dual \(u\) and scaled dual \(q=u/\rho\):

\[
 L_\rho=\tfrac12\theta^T R\theta + \sum_i C_i[1-z_i]_+
 +u^T(z-A\theta)+\tfrac\rho2\lVert z-A\theta\rVert^2.
\]

The first update is

\[
 (A^TA+R/\rho)\theta^{k+1}=A^T(z^k+q^k).
\]

Since \(y_i^2=1\), \(A^TA=H^TH\). Cache this Gram matrix once per fit and
factor with Cholesky. Refactor only when \(\rho\) changes. With the repository's
29 features, this is a **30 × 30** system. Neither an n × n kernel matrix nor
an inverse is formed. Building the Gram matrix costs O(nd²); each iteration
costs O(nd+d²), including diagnostics, plus O(d³) for a factorization when rho
changes (and on the first solve). Storage is O(nd+n+d²), including the
signed design and samplewise z/dual vectors. The proposal's O(d²) memory figure
does not include these arrays.

## Proximal update, relaxation, and penalty adaptation

For fixed-rho ADMM, \(\alpha=1\). For relaxed ADMM, define

\[
 m=A\theta^{k+1},\qquad
 \widehat m=\alpha m+(1-\alpha)z^k,\qquad a=\widehat m-q^k.
\]

Then

\[
 z_i^{k+1}=\begin{cases}
 a_i,&a_i\ge1,\\
 \min(1,a_i+C_i/\rho),&a_i<1,
 \end{cases}\qquad
 q^{k+1}=q^k+z^{k+1}-\widehat m.
\]

Relaxation is used in **both** the proximal and dual updates. This follows the
standard scheme in [Boyd et al., §§3.3–3.4](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf).
It resolves the inconsistent placement of relaxation in the proposal's
pseudocode. The admissible range is 0 < alpha < 2; the ablations use 1/1.5/1.8.

Record the actual primal feasibility residual
\(r=z-m\) and dual residual \(s=\rho A^T(z^{k+1}-z^k)\), with tolerances

\[
 \epsilon_p=\sqrt n\,\tau+\tau\max(\lVert m\rVert,\lVert z\rVert),\quad
 \epsilon_d=\sqrt{d+1}\,\tau+\tau\lVert A^T u\rVert.
\]

If ||r|| > 10||s||, double rho; if ||s|| > 10||r||, halve it. Values are
bounded to [1e-8, 1e8]. On a change from rho_old to rho_new, rescale
`q *= rho_old / rho_new`, keeping **u = rho*q unchanged**. The proposal calls
its multiplier u but also says to rescale it: rescaling applies to the scaled
variable q, not to the unscaled multiplier in its displayed Lagrangian.

## Certified stopping and best-iterate return

Successive objective stagnation does not prove optimality. Each ADMM iteration
also constructs feasible SVM dual variables from `clip(rho*q, 0, C_i)`.
Shrink the class with the larger total dual mass to enforce
\(\sum_i\lambda_i y_i=0\), preserving all box constraints. This is a cheap
feasibility construction, not a Euclidean projection. It gives the lower bound

\[
 D(\lambda)=\sum_i\lambda_i-
 \tfrac12\lVert X^T(y\odot\lambda)\rVert^2\le P^*.
\]

Keep the largest observed dual lower bound and the smallest observed primal
objective. Set `converged=True` only when **both residuals** pass their
thresholds and `(best_primal - best_dual)/max(abs(best_dual),1e-12) <= tol`.
When the optimum is zero, this is a deliberately strict denominator floor;
the single-class regression tests cover an exactly attained zero objective.

Return the best primal iterate on convergence, iteration limit, or budget
exit. If this selects an earlier point, append a timed `best_iterate`
checkpoint, ensuring `objective_history[-1] == objective`. Samplewise z/q and
residuals describe the last ADMM iteration; the returned w/b may be an earlier
best iterate. `stop_reason` distinguishes `converged`, `max_iter`, and
`time_budget`.

The loop reserves up to 0.25 seconds of the fit budget for an atomic matrix
operation and final return; this is necessary because WP4 rejects any measured
fit overrun. A single large factorization remains an atomic operation; the
harness's hard timeout provides the external limit. `n_iter` counts ADMM
iterations only. Warm-start epochs and updates have separate counters.

The internal **certified** gap is different from WP4's post-hoc gap against the
best observed solver objective. Both are useful; do not present the latter as
a certificate of global accuracy when every fit stops early.

## Weighted Pegasos warm start

`PAADMMSVM` shares the ADMM loop. Its defaults are K=10, adaptive_rho=True,
rho0=1, alpha=1.6. `ADMMSVM` is cold-started, fixed-rho, alpha=1 by default.
`PAADMMSVM().fit(X, y, C_per_sample, K=0, adaptive_rho=False, alpha=1)`
exactly reproduces cold ADMM; hyperparameters belong in `fit`, not the constructor.

WP1's standalone `src/svm_pegasos.py` is currently absent. WP3 therefore owns a
private weighted Pegasos warm-start stage; it neither supplies a registered
`pegasos` baseline nor modifies WP1 files. Its objective scaling is explicit.
Let S = sum(C_i), lambda = 1/S, and a_i = C_i/mean(C). Then

\[
 P/S=\tfrac\lambda2\lVert w\rVert^2+
 \tfrac1n\sum_i a_i[1-y_i(x_i^Tw+b)]_+.
\]

Use `eta_t=1/(lambda*t)` and the mean weighted hinge subgradient over each
mini-batch. The default batch size is 1; epochs traverse seeded random
permutations. The bias has no shrinkage. Project w to radius sqrt(2S), a safe
bound from P(0,0)=S. This is a weighted/bias adaptation of
[Pegasos](https://home.ttic.edu/~nati/Publications/PegasosMPB.pdf), not a new
convergence claim for the intercept. No class weights are applied twice.

Keep the best objective checkpoint among the initial zero model and completed
or interrupted epochs, because stochastic iterates can overshoot. Initialize
ADMM with that theta, z=A theta, q=0. K is requested epochs;
`warm_start_epochs_completed` and `warm_start_updates` report actual work.
`warm_start_batch_size` is configurable and recorded in the harness config.
Any change from the single-example default must be recorded explicitly;
results with different batch sizes do not describe the same warm-start setting.

All timers begin **before** warm start and Gram construction. Warm-start
objective checkpoints and elapsed times are included in the canonical
objective/time histories. `stage_history` distinguishes initial, pegasos,
admm, and best_iterate checkpoints. `rho_history` is aligned with these
checkpoints; residual and dual-bound histories contain ADMM iterations only.

## Reproduction

Run from the repository root; the virtual environment and raw data stay ignored:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python experiments/check_solver.py --method admm
.venv/bin/python experiments/check_solver.py --method pa_admm

# Final team experiment: unchanged 90-minute-per-fit protocol, not the pilot.
.venv/bin/python experiments/run_ablation.py --config configs/ablation.yaml --dry-run
.venv/bin/python experiments/run_ablation.py --config configs/ablation.yaml
```

The harness does not replace existing fit records by default; it refreshes
their common-reference fields after comparison. The existing contract checker
can regenerate its diagnostic report. The numerical tests compare against an
independent LIBSVM solution using the same rows and supplied C_i; ordinary WP4
harness runs produce auditable JSONs. `requirements-wp3.txt` optionally adds
CVXPY/ECOS for independent numerical reference work; these packages are not
needed by the solver or its LIBSVM-based tests.

The standalone WP3 validation/plotting scripts, short-budget pilot config,
generated CSV/PDF/PNG outputs, results record, and presentation notes have been
removed from this contribution. The common team runner and final config remain
available. The report draft retains explicitly labelled historical numbers;
their original reference/pilot workflow is not bundled here.

## Scope and remaining WP3 deliverables

The code provides both registered solvers, component switches, objective/time
histories, and small-data independent-reference/integration tests. This covers
WP3's implementation and handoff work. The proposal also assigns WP3 a component
ablation study and runtime figures: the final 90-minute-budget study and its
committed aggregate results/figures remain outstanding. The earlier short-budget
pilot does not replace them. The existing ablation runner can execute the WP3
variants without the missing WP1/WP2 solvers; those missing solvers affect the
separate all-five comparison. `EXPERIMENTS.md` and the existing configs remain
authoritative. No speedup or warm-start benefit is assumed in advance.

### WP4 handoff: match C before comparing training objectives

`run_one` selects C separately for each method from the shared grid. The current
`_assign_reference_objectives` grouping includes dataset, cost ratios, seed,
and training size, but **does not include the selected C**. If selected C values
differ, the canonical objectives describe different QPs and cannot share an
objective-gap reference. Before final main experiments, coordinate with WP4
to use a common C for an objective comparison or group references by the exact
training penalties/C, leaving insufficient comparison groups unranked. A
shared C grid alone is not sufficient. WP3's independent checks and pilot
both fix C=1, so their objective comparisons are unaffected. This handoff
records the issue without changing WP4's owned evaluation module.
