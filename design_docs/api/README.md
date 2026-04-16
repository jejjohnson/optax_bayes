---
status: draft
version: 0.1.0
---

# API Overview

## Surface Inventory

### Layer 0 — Primitives (`optax_bayes._src`)

Pure JAX functions. The BLR update equations as stateless math.

| Function | Description |
|---|---|
| `blr_diag_update_step(eta, s, g, h, m, eta0, s0, rho)` | One diagonal BLR step: $\eta, s \leftarrow (1-\rho)(\eta,s) + \rho(\eta_0 + g - hm,\; s_0 - h)$ |
| `blr_full_update_step(eta, Lambda, g, H, m, eta0, Lambda0, rho)` | One full-rank BLR step |
| `natural_to_mean_diag(eta, s) → (m, v)` | $m = \eta/s, \; v = 1/s$ |
| `mean_to_natural_diag(m, v) → (eta, s)` | $\eta = m/v, \; s = 1/v$ |
| `ggn_diag(grads) → diag_H` | GGN diagonal Hessian: $H \approx -g^2$ |
| `hutchinson_diag(hvp_fn, key, n_probes) → diag_H` | Hutchinson diagonal estimator |

See: [primitives.md](primitives.md)

### Layer 1 — Components (optax GradientTransformations)

| Export | Signature | Description |
|---|---|---|
| `blr_diagonal(lr, prior_precision, ...)` | `→ GradientTransformation` | Diagonal BLR (log-likelihood convention) |
| `blr_full_rank(lr, prior_precision, ...)` | `→ GradientTransformation` | Full-rank BLR |
| `blr_low_rank(lr, rank, ...)` | `→ GradientTransformation` | Low-rank BLR (planned) |
| `BLRDiagState` | `NamedTuple(precision, nat_mean, count)` | Diagonal optimizer state |
| `BLRFullRankState` | `NamedTuple(precision, nat_mean, count)` | Full-rank optimizer state |

See: [components.md](components.md)

### Layer 2 — Models (convenience wrappers)

| Export | Description |
|---|---|
| `blr_diagonal_for_loss(...)` | Negates gradients — drop-in for `optax.adam(...)` |
| `blr_full_rank_for_loss(...)` | Negates gradients — full-rank version |
| `blr_with_schedule(schedule_fn, ...)` | Compose with optax learning rate schedule |
| `get_posterior_diagonal(state) → (mean, variance)` | Extract posterior from diagonal state |
| `get_posterior_full_rank(state) → (mean, covariance)` | Extract posterior from full-rank state |

See: [models.md](models.md)

---

## Notation

| Symbol | Meaning |
|---|---|
| $\eta$ | Natural mean: $\eta = s \odot m$ (diagonal) or $\eta = \Lambda m$ (full-rank) |
| $s$ | Diagonal precision: $s = 1/v$ |
| $\Lambda$ | Precision matrix: $\Lambda = \Sigma^{-1}$ |
| $g$ | Gradient of log-likelihood: $g = \mathbb{E}_q[\nabla_\theta \ell]$ |
| $H$ | Hessian of log-likelihood: $H = \mathbb{E}_q[\nabla^2_\theta \ell]$ (NSD) |
| $\rho$ | Learning rate (step size) |
| $\lambda_0$ | Prior natural parameters: $(\eta_0, s_0)$ or $(\eta_0, \Lambda_0)$ |

---

## Import Conventions

```python
# Layer 1 — optax transforms (most users start here)
from optax_bayes import blr_diagonal_for_loss, blr_full_rank_for_loss
from optax_bayes import get_posterior_diagonal, get_posterior_full_rank

# Layer 1 — log-likelihood convention
from optax_bayes import blr_diagonal, blr_full_rank

# Layer 0 — pure functions (advanced users)
from optax_bayes._src.primitives import blr_diag_update_step
from optax_bayes._src.hessians import ggn_diag
```

---

## Detail Files

| File | Covers |
|---|---|
| [primitives.md](primitives.md) | Layer 0 — BLR update functions, Hessian estimators, parameter conversions |
| [components.md](components.md) | Layer 1 — optax GradientTransformations, state types |
| [models.md](models.md) | Layer 2 — loss wrappers, posterior extraction, schedule composition |

---

*For full mathematical treatment, see [../research/overview.md](../research/overview.md).*
*For usage patterns, see [../examples/](../examples/).*
