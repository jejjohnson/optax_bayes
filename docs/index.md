# optax_bayes

> The Bayesian Learning Rule as optax `GradientTransformation`s.

Train with a normal [optax](https://optax.readthedocs.io/) loop, then read an
approximate Gaussian posterior straight out of the optimizer state. Implements
the Bayesian Learning Rule (Khan & Rue, 2023) in diagonal, full-rank, and
low-rank variants, plus IVON (Shen et al., 2024) and Newton's method.

**New here?** Start with the
[Diagonal BLR Walkthrough](notebooks/diagonal_blr_walkthrough.ipynb), then
browse the [API Reference](api/index.md) for the full surface and the
mathematical background.

## Installation

```bash
pip install optax_bayes              # slim core: diagonal BLR + IVON
pip install "optax_bayes[gaussx]"    # + full-rank / low-rank structured backend
pip install "optax_bayes[all]"       # everything
```

Or with `uv`:

```bash
uv add optax_bayes
```

The core package depends only on `jax`, `optax`, `jaxtyping`, and `lineax`;
the full-rank and low-rank transforms need the optional
[gaussx](https://github.com/jejjohnson/gaussx) extra.

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

## API Notes

A few conventions worth calling out up front:

- The `*_for_loss` transforms accept standard **loss** gradients (like
  `optax.adam`); the bare `blr_*` transforms expect **log-likelihood**
  gradients.
- Optimizer state is stored in natural-parameter form and initialises its
  mean at the params passed to `opt.init`; `prior_mean` / `prior_precision`
  anchor every update.
- Full-rank and low-rank entry points raise an informative `ImportError`
  unless the `gaussx` extra is installed.

## Links

- [API Reference](api/index.md)
- [Changelog](CHANGELOG.md)
- [GitHub](https://github.com/jejjohnson/optax_bayes)
