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

`optax_bayes` implements the Bayesian Learning Rule (Khan & Rue, 2023) as a
family of drop-in [optax](https://github.com/google-deepmind/optax) optimizers.
Train with a normal optax loop, then read an approximate Gaussian posterior
straight out of the optimizer state.

- **Diagonal BLR** — `blr_diagonal_for_loss`: Adam-like cost, per-parameter
  uncertainty. Works on arbitrary parameter pytrees.
- **Full-rank BLR** — `blr_full_rank_for_loss`: exact Gaussian posterior
  recovery on conjugate problems; supports exact-Hessian callables.
- **Low-rank BLR** — `blr_low_rank_for_loss`: `Lambda = diag(D) + U U^T`
  precision with O(dr) storage, Woodbury solves via
  [gaussx](https://github.com/jejjohnson/gaussx) structured operators.
- **IVON** — `ivon`: Improved Variational Online Newton (Shen et al., 2024)
  for deep learning.
- **Newton's method** — `newton_for_loss`: classic damped Newton as the
  rho=1 special case of full-rank BLR.

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

## Quickstart

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

### Conventions

- `*_for_loss` transforms accept standard **loss** gradients (like
  `optax.adam`). The bare `blr_*` transforms expect **log-likelihood**
  gradients and are the right layer for custom inference loops.
- Optimizer state is stored in natural-parameter form
  (`eta = s * m`, `s = 1 / v`); state initialises its mean at the params
  passed to `opt.init`, while `prior_mean` / `prior_precision` anchor
  every update.

## Development

```bash
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
