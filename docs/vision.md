# Vision

> **optax_bayes** implements the Bayesian Learning Rule (Khan & Rue, 2023) as
> optax `GradientTransformation`s: optimizers you drop in where you would use
> `optax.adam`, which also leave an approximate Gaussian posterior over the
> parameters in their state.

## Motivation

The Bayesian Learning Rule (BLR) puts optimisation, inference and learning
under one update. Gradient descent, Newton's method, RMSprop- and Adam-like
methods, the Laplace approximation, Kalman filtering, variational inference
and continual learning all come out of natural-gradient descent on a
variational objective, for particular choices of variational family, Hessian
approximation, prior and step size (see [Mathematical Background](math.md)).

The standard way to run the BLR is an exponential moving average of natural
parameters: the precision and the natural mean. That is exactly optax's
`init → update → state` protocol. The optimizer state holds the natural
parameters $(\eta, s)$ or $(\eta, \Lambda)$, each step is the BLR's convex
combination, and the posterior mean and (co)variance can be read from the
state at any point.

## User stories

**ML researcher.** "I want Adam-like training with per-parameter uncertainty
at the end, without a separate inference pass after training."

**Bayesian practitioner.** "I want natural-gradient variational inference as
an optax optimizer that composes with `optax.chain`, clipping and schedules."

**Continual-learning researcher.** "I want the posterior after task 1 to be
the prior for task 2, the way the BLR derives EWC-style regularisation."

## Design principles

1. **Optax-native.** Every variant is an `optax.GradientTransformation`. It
   composes with `optax.chain` and `optax.inject_hyperparams`, and runs under
   `jax.jit` and `jax.lax.scan`.
2. **Natural-parameter state.** The state stores natural parameters, not
   moments, so each step is the BLR's convex combination as written.
3. **The posterior is always available.** `get_posterior_*` and
   `sample_posterior_*` read the current state; nothing extra runs during
   training.
4. **Structured linear algebra from gaussx.** The full-rank and low-rank
   variants delegate solves and Woodbury identities to
   [gaussx](https://github.com/jejjohnson/gaussx), an optional dependency.
   The diagonal and IVON variants need only `jax` and `optax`.
5. **Log-likelihood convention, with loss wrappers.** The core transforms
   take $\nabla_\theta \log p(\mathcal{D} \mid \theta)$; the `*_for_loss`
   wrappers take loss gradients, like any other optax optimizer.

## What optax_bayes is

- `blr_diagonal`: a drop-in Adam replacement with per-parameter uncertainty,
  on arbitrary parameter pytrees.
- `blr_full_rank`: natural-gradient VI with a dense posterior covariance, for
  small flat parameter vectors.
- `blr_low_rank`: $\Lambda = \operatorname{diag}(D) + UU^\top$, with
  $O(dr)$ storage and $O(dr^2)$ work per step.
- `ivon`: an IVON-style diagonal optimizer for deep learning.
- `newton`: Newton's method as the full-rank BLR with $\rho = 1$.

## What optax_bayes is not

| Not this | Use instead |
|---|---|
| A general inference engine (MCMC, SVI) | NumPyro |
| BLR for non-Gaussian exponential families | Out of scope; optax_bayes is Gaussian-only |
| A neural-network library | Equinox, Flax |
| A training loop | Your own code: optax_bayes is the optimizer |

## Related methods

| Method | Relation |
|---|---|
| IVON (Shen et al., 2024) | A diagonal BLR variant, shipped as `ivon` |
| VOGN (Osawa et al., 2019) | Diagonal BLR with a Gauss–Newton curvature estimate |
| Vadam (Khan et al., 2018) | Diagonal BLR as weight-perturbed Adam |
| SLANG (Mishkin et al., 2018) | The low-rank-plus-diagonal update used by `blr_low_rank` |
| Adam + post-hoc Laplace | The BLR gives the uncertainty during training rather than after it |

## References

- Khan, M. E. & Rue, H. (2023). [The Bayesian Learning Rule](https://arxiv.org/abs/2107.04562). *JMLR* 24(281):1–46.
- Shen, Y. et al. (2024). [Variational Learning is Effective for Large Deep Networks](https://arxiv.org/abs/2402.17641). *ICML*.
- Osawa, K. et al. (2019). [Practical Deep Learning with Bayesian Principles](https://arxiv.org/abs/1906.02506). *NeurIPS*.
- Khan, M. E. et al. (2018). [Fast and Scalable Bayesian Deep Learning by Weight-Perturbation in Adam](https://arxiv.org/abs/1806.04854). *ICML*.
- Mishkin, A. et al. (2018). [SLANG: Fast Structured Covariance Approximations for Bayesian Deep Learning with Natural Gradient](https://arxiv.org/abs/1811.04504). *NeurIPS*.
