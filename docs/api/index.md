# API Reference

optax_bayes implements the Bayesian Learning Rule (Khan & Rue, 2023) as drop-in
[optax](https://optax.readthedocs.io/) `GradientTransformation`s: train with a
normal optax loop, then read an approximate Gaussian posterior straight out of
the optimizer state. The reference is organised by variational family:

| Section | Family | What's inside |
|---------|--------|---------------|
| [Diagonal BLR](diagonal.md) | $\mathcal{N}(m, \operatorname{diag}(1/s))$ | `blr_diagonal` / `blr_diagonal_for_loss`, schedules, posterior extraction and sampling — works on arbitrary parameter pytrees |
| [Full-Rank BLR](full_rank.md) | $\mathcal{N}(m, \Lambda^{-1})$ | `blr_full_rank` / `blr_full_rank_for_loss`, Newton's method presets, dense posterior extraction and sampling |
| [Low-Rank BLR](low_rank.md) | $\Lambda = \operatorname{diag}(D) + UU^\top$ | `blr_low_rank` / `blr_low_rank_for_loss`, Woodbury-based posterior extraction and sampling |
| [IVON](ivon.md) | diagonal, deep-learning flavoured | `ivon`, posterior sampling for MC training, posterior extraction |

## Conventions

A few patterns hold across the whole package:

- **Two gradient conventions.** Every BLR variant comes in two flavours: the
  bare transform (`blr_diagonal`, `blr_full_rank`, `blr_low_rank`) expects
  **log-likelihood** gradients and is the right layer for custom inference
  loops, while the `*_for_loss` wrapper accepts standard **loss** gradients
  (same convention as `optax.adam`) and negates internally,
  $g_{\text{loglik}} = -g_{\text{loss}}$. `ivon` always uses the loss
  convention.

- **Natural-parameter state.** Optimizer state stores the precision and the
  natural mean $\eta = \Lambda m$, not the moment parameters. The state
  initialises its mean at the params passed to `opt.init` (standard optax
  drop-in semantics — start from pretrained weights and the variational mean
  starts there too), while `prior_mean` / `prior_precision` anchor every
  update through $\eta_0$, $\Lambda_0$.

- **True optax citizens.** Every transform composes with stock combinators
  (`optax.chain`, `optax.inject_hyperparams`), runs under `jax.jit` and
  `jax.lax.scan`, and its state round-trips as a pytree for checkpointing.

- **gaussx is optional.** The diagonal and IVON surface depends only on
  `jax` / `optax` / `jaxtyping` / `lineax`. The full-rank and low-rank paths
  use [gaussx](https://github.com/jejjohnson/gaussx) structured operators for
  Woodbury-dispatched solves and raise an informative `ImportError` unless you
  install the extra: `pip install "optax_bayes[gaussx]"`.

## Mathematical Background

The optimizer stores natural parameters of a Gaussian $q(\theta)$ and updates
them via

$$
\begin{aligned}
\Lambda_{t+1} &= (1 - \rho)\, \Lambda_t + \rho\,(\Lambda_0 - H_t) \\
\eta_{t+1}    &= (1 - \rho)\, \eta_t    + \rho\,(\eta_0 + g_t - H_t\, m_t) \\
m_{t+1}       &= \Lambda_{t+1}^{-1}\, \eta_{t+1}
\end{aligned}
$$

where $g_t$ is the log-likelihood gradient, $H_t \preceq 0$ is the Hessian
estimate, and $\Lambda_0, \eta_0$ are the prior natural parameters. The
$-H_t m_t$ correction is the Bonnet–Price identity term; with the exact
Hessian the fixed point is the exact Gaussian posterior of a conjugate model.

### Variants

| Variant | Precision $\Lambda$ | Cost per step | Storage |
|---|---|---|---|
| **Diagonal** | $\operatorname{diag}(s)$, elementwise | $O(d)$ | $O(d)$ |
| **Full-rank** | Dense $(d \times d)$ matrix | $O(d^3)$ solve | $O(d^2)$ |
| **Low-rank** | $\operatorname{diag}(D) + UU^\top$, $U \in \mathbb{R}^{d \times r}$ | $O(dr^2 + r^3)$ Woodbury | $O(dr)$ |

### Hessian estimators

**Diagonal** (`hessian_estimator` parameter, string selector):

| String | Formula | Behaviour |
|---|---|---|
| `"ggn_diag"` | $h = -g^2$ | Adam-like curvature |
| `"identity"` | $h = 0$ | Fixes precision to prior; mean converges to the MAP |

**Full-rank / low-rank** (`hessian_estimator` parameter, string or callable):

| Value | Formula | Behaviour |
|---|---|---|
| `"ggn"` | $H = -g\,g^\top$ | Rank-1 outer product, always NSD |
| `"identity"` | $H = 0$ | Fixes precision to prior; mean converges to the MAP |
| `callable(mean, grads)` | User-provided $(d \times d)$ matrix | Custom Hessian (e.g. exact, Fisher, GGN) |

For the `*_for_loss` wrappers, callable estimators return the **loss** Hessian
and are negated internally; string selectors are sign-invariant.

### Posterior extraction

| Function | Returns |
|---|---|
| [`get_posterior_diagonal`][optax_bayes.get_posterior_diagonal] | `(mean, variance)` pytrees: $m = \eta/s$, $v = 1/s$ |
| [`get_posterior_full_rank`][optax_bayes.get_posterior_full_rank] | `(mean, covariance)`: $m = \Lambda^{-1}\eta$, $\Sigma = \Lambda^{-1}$ |
| [`get_posterior_low_rank`][optax_bayes.get_posterior_low_rank] | `(mean, covariance)`: Woodbury inverse of $\operatorname{diag}(D) + UU^\top$ |
| [`get_posterior_ivon`][optax_bayes.get_posterior_ivon] | `(mean, variance)` pytrees: $v = 1/(h + \lambda)$ |

Every public function carries a Google-style docstring; the matching
reparameterised samplers (`sample_posterior_*`, `sample_ivon`) are documented
alongside each family.

## References

- Khan & Rue (2023), [*The Bayesian Learning Rule*](https://arxiv.org/abs/2107.04562), JMLR.
- Shen et al. (2024), [*Variational Learning is Effective for Large Deep Networks*](https://arxiv.org/abs/2402.17641), ICML.
