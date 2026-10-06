# Design Decisions

## D1: optax `GradientTransformation`, not a custom optimizer

**Status:** accepted

**Context.** The BLR could ship as its own optimizer class or as an optax
transform.

**Decision.** An optax `GradientTransformation`. It composes with
`optax.chain`, schedules, clipping and `inject_hyperparams`.

**Consequences.** Users call `optax.apply_updates` as usual. Each `update`
returns $m_{t+1} - m_t$, so the params track the variational mean.

## D2: natural-parameter state

**Status:** accepted

**Context.** The state could hold moments $(m, v)$ or natural parameters
$(\eta, s)$.

**Decision.** Natural parameters. The BLR is a convex combination in natural
parameters, so storing them keeps each step a direct transcription.

**Consequences.** `get_posterior_*` converts state to mean and (co)variance.
The low-rank variant still stores $\eta$, but takes its mean step in mean form
(D7).

## D3: log-likelihood convention, with `*_for_loss` wrappers

**Status:** accepted

**Context.** The BLR is written with $\nabla_\theta \log p(\mathcal{D} \mid
\theta)$; ML code minimises a loss $L = -\log p$.

**Decision.** The core transforms take log-likelihood gradients. The
`*_for_loss` wrappers negate loss gradients, and negate callable loss
Hessians.

**Consequences.** Two entry points per variant. Most users want `*_for_loss`.
`ivon` uses the loss convention only.

## D4: gaussx for structured precision, as an optional extra

**Status:** accepted

**Context.** The full-rank variant needs $\Lambda^{-1}\eta$; the low-rank one
needs Woodbury solves.

**Decision.** Delegate to gaussx (`gaussx.solve`, `gaussx.inv`,
`gaussx.low_rank_plus_diag`), imported lazily so the core does not depend on
it.

**Consequences.** `pip install "optax_bayes[gaussx]"` for the full-rank and
low-rank transforms. Without it they raise an informative `ImportError`.

## D5: diagonal BLR is the primary entry point

**Status:** accepted

**Decision.** `blr_diagonal_for_loss` leads the docs and examples: it has
Adam's memory footprint and works on any parameter pytree.

## D6: precisions are tagged positive semi-definite

**Status:** accepted

**Context.** gaussx and lineax pick a solver from operator tags. An untagged
dense precision is solved by LU, and `lx.Cholesky()` rejects it.

**Decision.** The full-rank and low-rank transforms tag their precision PSD,
which the BLR guarantees when $H_t \preceq 0$. `newton` does not, because an
exact Hessian can be indefinite away from a maximum.

## D7: SLANG update for the low-rank family

**Status:** accepted

**Context.** The first low-rank update formed the dense Hessian estimate and
eigendecomposed it every step, at $O(d^3)$.

**Decision.** Use SLANG (Mishkin et al., 2018). Curvature arrives in factor
form $-H = GG^\top$, $[\sqrt{1-\rho}\,U, \sqrt{\rho}\,G]$ is truncated by a
thin SVD, and the discarded diagonal is folded into $D$. The mean takes the
mean-form step $m \leftarrow m + \rho\Lambda^{-1}(g - D_0(m - m_0))$, which
equals the natural-mean recursion when nothing is truncated and stays stable
when something is.

**Consequences.** $O(dr^2)$ per step with the built-in estimators. A callable
estimator still returns a dense Hessian and costs $O(d^3)$ to factor.

## D8: Gaussian families only

**Status:** accepted

**Context.** The BLR applies to any exponential family. The research notes
considered efax as a general backend.

**Decision.** Gaussian families only, with gaussx as the backend.

## Resolved questions

| Question | Resolution |
|---|---|
| Optimizer protocol | optax `GradientTransformation` (D1) |
| State parameterisation | Natural parameters (D2) |
| Gradient convention | Log-likelihood, plus `*_for_loss` wrappers (D3) |
| Structured precision backend | gaussx, optional (D4) |
| Default variant | Diagonal (D5) |
| Solver selection | PSD tags; `newton` excepted (D6) |
| Low-rank update | SLANG (D7) |
| Non-Gaussian families | Out of scope (D8) |
| Per-parameter priors | Open: see [Scope](boundaries.md#open-questions) |
