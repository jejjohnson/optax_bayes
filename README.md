# optax_bayes

[![Tests](https://github.com/jejjohnson/optax_bayes/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/optax_bayes/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/optax_bayes/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/optax_bayes/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/optax_bayes/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/optax_bayes/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/optax_bayes/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/optax_bayes/actions/workflows/pages.yml)
[![codecov](https://codecov.io/gh/jejjohnson/optax_bayes/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/optax_bayes)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

**The Bayesian Learning Rule as optax `GradientTransformation`s.**

Train with a normal [optax](https://github.com/google-deepmind/optax) loop,
then read an approximate Gaussian posterior straight out of the optimizer
state. Implements the Bayesian Learning Rule (Khan & Rue, 2023) in diagonal,
full-rank, and low-rank variants, plus IVON (Shen et al., 2024) and Newton's
method. Structured solves are backed by
[gaussx](https://github.com/jejjohnson/gaussx).

**Documentation**: [jejjohnson.github.io/optax_bayes](https://jejjohnson.github.io/optax_bayes)

## Installation

```bash
pip install optax_bayes              # slim core: diagonal BLR + IVON
pip install "optax_bayes[gaussx]"    # + full-rank / low-rank structured backend
pip install "optax_bayes[all]"       # everything
```

The core package depends only on `jax`, `optax`, `jaxtyping`, and `lineax`.
The full-rank and low-rank transforms (and their posterior/sampling helpers)
need the optional `gaussx` extra and raise an informative `ImportError`
pointing at it otherwise.

## Quick Start

```python
import jax
import optax
from optax_bayes import blr_diagonal_for_loss, get_posterior_diagonal

opt = blr_diagonal_for_loss(learning_rate=1e-3, prior_precision=1e-4)
state = opt.init(params)

for batch in dataloader:
    grads = jax.grad(loss_fn)(params, batch)
    updates, state = opt.update(grads, state)
    params = optax.apply_updates(params, updates)

mean, variance = get_posterior_diagonal(state)   # q(theta) = N(mean, diag(variance))
```

Everything composes like a normal optax transform:

```python
opt = optax.chain(
    optax.clip_by_global_norm(1.0),
    blr_diagonal_for_loss(learning_rate=1e-3),
)
```

Posterior sampling for Monte Carlo prediction:

```python
from optax_bayes import sample_posterior_diagonal

theta = sample_posterior_diagonal(state, jax.random.key(0))
```

## What's Inside

### Variational families

| Variant | Precision | Cost / step | Storage | Transforms |
|---------|-----------|-------------|---------|------------|
| **Diagonal** | `diag(s)` | O(d) | O(d) | `blr_diagonal`, `blr_diagonal_for_loss`, `blr_with_schedule` |
| **Full-rank** | dense (d, d) | O(d^3) | O(d^2) | `blr_full_rank`, `blr_full_rank_for_loss` |
| **Low-rank** | `diag(D) + U U^T` | O(dr^2 + r^3) | O(dr) | `blr_low_rank`, `blr_low_rank_for_loss` |

### Deep-learning variant

`ivon` — Improved Variational Online Newton (Shen et al., 2024): separate EMA
rates for mean and precision, weight decay as the prior, natural-space
gradient clipping, and `sample_ivon` for proper Bayesian MC training.

### Classic special case

`newton` / `newton_for_loss` — damped Newton's method as full-rank BLR with
rho = 1, a near-flat prior, and an exact Hessian.

### Posterior extraction & sampling

Every family pairs a `get_posterior_*` helper (mean + variance/covariance
straight from the optimizer state) with a reparameterised `sample_posterior_*`
sampler. Low-rank sampling never densifies the covariance — it uses a
two-step Woodbury trick at O(dr^2 + r^3) cost.

## API Notes

- The `*_for_loss` transforms accept standard **loss** gradients (like
  `optax.adam`). The bare `blr_*` transforms expect **log-likelihood**
  gradients and are the right layer for custom inference loops.
- Optimizer state is stored in natural-parameter form
  (`eta = s * m`, `s = 1 / v`); state initialises its mean at the params
  passed to `opt.init`, while `prior_mean` / `prior_precision` anchor
  every update.
- `hessian_estimator` accepts string selectors (`"ggn_diag"` / `"identity"`
  for diagonal; `"ggn"` / `"identity"` for full- and low-rank) or, for the
  full- and low-rank variants, a callable `fn(mean, grads) -> (d, d)` —
  pass the exact Hessian to recover the exact posterior on conjugate models.

## Development

```bash
git clone https://github.com/jejjohnson/optax_bayes.git
cd optax_bayes
make install      # uv sync --all-groups + pre-commit hooks
make test-fast    # fast unit tests (matches PR CI)
make test         # entire suite, in parallel
make test-slow    # only the slow + integration tests
make format       # ruff format + autofix
make typecheck    # ty check src/optax_bayes
```

Tests are split into speed tiers: unmarked fast unit tests run on every PR;
`@pytest.mark.slow` / `@pytest.mark.integration` tests (analytic posterior
recovery, Monte Carlo moment checks) run on demand via the
"Extended Tests" workflow (`gh workflow run tests-extended.yml`).

## References

- Khan & Rue (2023), [*The Bayesian Learning Rule*](https://arxiv.org/abs/2107.04562), JMLR.
- Shen et al. (2024), [*Variational Learning is Effective for Large Deep Networks*](https://arxiv.org/abs/2402.17641), ICML.

## License

MIT — see [LICENSE](LICENSE).
