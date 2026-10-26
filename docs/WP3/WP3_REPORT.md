# Report contribution — ADMM and PA-ADMM

Member 3's draft for integration by WP1. Historical numerical verification and
the full-data pilot limitations are reported below. The final component
ablation and runtime figures remain WP3 deliverables; the final five-method
benchmark remains a team deliverable.

The pilot and independent-check numbers below are historical summaries. Their
standalone scripts, short-budget config, CSV tables, figures, and separate
results record are excluded from this implementation contribution; these
numbers are not a bundled reproducible final benchmark. Numerical tests and
the common team experiment runner remain available.

## Fixed-penalty ADMM

We solve the same cost-sensitive linear SVM objective as the other methods,

\[
\min_{w,b}\; P(w,b)=\tfrac12\|w\|_2^2+
\sum_{i=1}^{n}C_i\max\{0,1-y_i(x_i^Tw+b)\}.
\]

The penalties \(C_i\) are supplied by the common evaluation pipeline and are
not reweighted inside the solver. The bias is unregularized. With
\(\theta=(w,b)\), \(A=\operatorname{diag}(y)[X,\mathbf1]\), and
\(R=\operatorname{diag}(I_d,0)\), introduce \(z=A\theta\). The split problem
separates the quadratic regularizer from the samplewise hinge terms.

For the constraint \(z-A\theta=0\), let \(q=u/\rho\) denote the scaled dual
variable. One fixed-penalty iteration is

\[
\begin{aligned}
(A^TA+R/\rho)\theta^{k+1}&=A^T(z^k+q^k),\\
a&=A\theta^{k+1}-q^k,\\
z_i^{k+1}&=\begin{cases}
a_i,&a_i\ge1,\\
\min\{1,a_i+C_i/\rho\},&a_i<1,
\end{cases}\\
q^{k+1}&=q^k+z^{k+1}-A\theta^{k+1}.
\end{aligned}
\]

The hinge update is separable and requires no iterative inner solver. We cache
\(A^TA=[X,\mathbf1]^T[X,\mathbf1]\) and use Cholesky solves rather than an
explicit inverse. The fixed-penalty baseline starts from zero, uses rho=1 and
alpha=1 in the common experiment configuration, and reuses its factorization.

## Pegasos-warmed adaptive ADMM

PA-ADMM combines an initial weighted stochastic subgradient stage with the
same ADMM solver. Let \(S=\sum_i C_i\), \(\lambda=1/S\), and
\(a_i=C_i/\operatorname{mean}(C)\). Dividing the canonical objective by S gives

\[
P/S=\frac\lambda2\|w\|^2+
\frac1n\sum_i a_i\max\{0,1-y_i(x_i^Tw+b)\}.
\]

This scaling supplies a weighted Pegasos update with
\(\eta_t=1/(\lambda t)\); a mini-batch uses the mean sampled hinge
subgradient. The bias receives the hinge subgradient without regularization.
The best recorded primal checkpoint over K shuffled passes and the zero
initialization supplies \(\theta^0\), followed by \(z^0=A\theta^0\) and
\(q^0=0\). The total fit timer includes the stochastic stage, Gram construction,
factorizations, and ADMM iterations. This stage is an implementation adaptation
of [Pegasos](https://home.ttic.edu/~nati/Publications/PegasosMPB.pdf); no separate
convergence theorem for the weighted, unregularized-bias variant is claimed.

The second stage adapts rho using the ratio of primal and dual residual norms.
If one exceeds the other by a factor of ten, rho is respectively doubled or
halved. When rho changes, q is rescaled inversely and the system is refactored.
This preserves the unscaled multiplier. We also form
\(\widehat m=\alpha A\theta^{k+1}+(1-\alpha)z^k\), using this relaxed margin in
both the hinge and multiplier updates. These follow the penalty and relaxation
extensions described by [Boyd et al., §3.4](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf).
The experiment defaults are K=10, rho0=1, and alpha=1.6. Each component can be
disabled independently for ablation.

## Stopping, diagnostics, and computational cost

We require small primal and dual residuals together with a small relative
primal–dual objective gap. To obtain a computable lower bound, clip the
unscaled multiplier to \([0,C_i]\), then shrink the class with the greater
total multiplier mass so that \(\sum_i\lambda_i y_i=0\). The resulting feasible
dual vector gives

\[
D(\lambda)=\mathbf1^T\lambda-
\tfrac12\|X^T(y\odot\lambda)\|^2\le P^*.
\]

The best observed feasible primal value and dual lower bound provide a gap
certificate without access to a validation or test label. Objective
stagnation alone is not considered convergence. On a time or iteration limit,
the solver returns its best recorded primal iterate with converged=false.
Canonical objective and elapsed-time histories remain consistent with the
returned model. The experiment pipeline additionally computes its own
time-to-tolerance against a common cross-method reference; this reference is
not necessarily globally optimal.

Gram construction costs O(nd²); each subsequent iteration costs O(nd+d²), with
an additional O(d³) factorization whenever rho changes. Storage is
O(nd+n+d²), including the design and samplewise state. The ULB protocol excludes
Time and uses 29 model features, so the augmented system has dimension 30.
This differs from the raw-schema dimension quoted in the proposal.

## Numerical verification

We verify both implementations against independent CVXPY/ECOS and LIBSVM
solutions on identical training rows and per-sample penalties. A seeded
synthetic check uses 1400 training points; the real-data check uses a
stratified 10,000-row ULB subset with 7000 training points and 29 features.
For the real check, the ECOS objective is 5.083073735. Fixed ADMM with rho=0.01
returns 5.083577628, a relative error of 9.913e-5, while PA-ADMM initialized at
rho=1 returns 5.083187534, an error of 2.239e-5. Both agree with both reference
models on every training prediction. The synthetic check also passes the
1e-4 objective target and at least 99% prediction agreement.

The correctness check exposes penalty sensitivity: with rho=1 and a 10,000
iteration cap, cold ADMM retains an approximately 52% objective error on the
real subset despite nearly identical predicted labels. An additional numerical
sensitivity check fixes rho=0.01 and verifies the same training QP. It does not
select predictive hyperparameters or change the rho=1 benchmark configuration;
final predictive tuning follows the validation-only protocol. These
correctness runs use different penalty settings and do not establish a general
runtime ranking.

## Component study and interpretation limits

The implemented component matrix includes K in {0,5,10,20}, adaptation on/off,
alpha in {1,1.5,1.8}, and initial rho in {0.1,1,10}, with three shared ULB split
seeds. The historical pilot summarized below uses full training rows, C=1,
a 20:1 cost ratio, and the same 30-second fit
budget for every variant. Stochastic mini-batches contain 64 examples.

The pilot provides evidence about behavior under this short budget. It must
be distinguished from the final 5400-second team protocol and from the
single-example warm-start default. A component that spends time without
improving the starting objective may hurt short-budget performance. Likewise,
a smaller weighted training objective need not minimize validation-selected
test cost. Final claims about speedup, robustness, and the five-method ranking
require the corresponding common-protocol runs.

All 33 pilot fits returned successfully within 30 seconds, although none met
the full residual-and-certificate stopping conditions. Nine fits reached the
1e-3 target relative to the common best-observed reference. The variant with
adaptation disabled and alpha=1.6 provided the lowest observed objective in
each seed and reached that reference target in a mean of 10.563 seconds;
cold ADMM reached it in 17.663 seconds. These are observed-reference crossing
times, not times to a certified optimum.

In all 27 warm-started fits, no recorded stochastic checkpoint improved on the
zero model, so ADMM used the zero initialization. This exposes a limitation
of the tested warm-start settings and prevents attributing an advantage to
Pegasos initialization. Mean test cost was 0.006336 ± 0.001801 for the
adaptation-disabled variant and 0.006571 ± 0.001916 for cold ADMM. Seed
variation exceeds many between-variant differences; these three-seed results
do not establish a statistically significant generalization improvement or a
universal solver ranking.
