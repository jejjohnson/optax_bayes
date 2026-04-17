# API Reference

The `optax_bayes` public surface exposes the Bayesian Learning Rule (Khan & Rue, 2023) in three variants — diagonal, full-rank, and low-rank — as Optax gradient transformations.

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

mean, variance = get_posterior_diagonal(state)
```

---

## Convention Split

All three variants come in two flavours:

| Function | Expects | Use case |
|---|---|---|
| `blr_diagonal` / `blr_full_rank` / `blr_low_rank` | Log-likelihood gradients | Advanced users, custom likelihoods |
| `blr_diagonal_for_loss` / `blr_full_rank_for_loss` / `blr_low_rank_for_loss` | Standard loss gradients | Drop-in for `optax.adam` |

The `_for_loss` wrappers negate gradients internally: $g_{\text{loglik}} = -g_{\text{loss}}$.

---

## Mathematical Background

### The BLR Update

The optimizer stores natural parameters of a Gaussian $q(\theta)$ and updates them via:

$$\Lambda_{t+1} = (1 - \rho)\, \Lambda_t + \rho\,(\Lambda_0 - H_t)$$

$$\eta_{t+1} = (1 - \rho)\, \eta_t + \rho\,(\eta_0 + g_t - H_t\, m_t)$$

$$m_{t+1} = \Lambda_{t+1}^{-1}\, \eta_{t+1}$$

where $g_t$ is the log-likelihood gradient, $H_t$ is the Hessian estimate, and $\Lambda_0, \eta_0$ are prior natural parameters.

### Variants

| Variant | Precision $\Lambda$ | Cost per step | Storage |
|---|---|---|---|
| **Diagonal** | $\operatorname{diag}(s)$, elementwise | $O(d)$ | $O(d)$ |
| **Full-rank** | Dense $(d \times d)$ matrix | $O(d^3)$ solve | $O(d^2)$ |
| **Low-rank** | $\operatorname{diag}(D) + UU^T$, $U \in \mathbb{R}^{d \times r}$ | $O(dr^2 + r^3)$ Woodbury | $O(dr)$ |

### Hessian Estimators

**Diagonal** (`hessian_estimator` parameter):

| String | Formula | Behaviour |
|---|---|---|
| `"ggn_diag"` | $h = -g^2$ | Adam-like curvature |
| `"identity"` | $h = 0$ | Fixes precision to prior |

**Full-rank / Low-rank** (`hessian_estimator` parameter — Option C: string or callable):

| Value | Formula | Behaviour |
|---|---|---|
| `"ggn"` | $H = -g\,g^T$ | Rank-1 outer product, always NSD |
| `"identity"` | $H = 0$ | Fixes precision to prior |
| `callable(mean, grads)` | User-provided $(d \times d)$ matrix | Custom Hessian (e.g., exact, Fisher, GGN) |

### Posterior Extraction

| Function | Returns |
|---|---|
| `get_posterior_diagonal(state)` | `(mean, variance)` pytrees: $m = \eta/s$, $v = 1/s$ |
| `get_posterior_full_rank(state)` | `(mean, covariance)`: $m = \Lambda^{-1}\eta$, $\Sigma = \Lambda^{-1}$ |
| `get_posterior_low_rank(state)` | `(mean, covariance)`: Woodbury inverse of $D + UU^T$ |

---

## Diagonal BLR

### `blr_diagonal_for_loss`

::: optax_bayes._src.wrappers.blr_diagonal_for_loss

### `blr_diagonal`

::: optax_bayes._src.diagonal.blr_diagonal

### `get_posterior_diagonal`

::: optax_bayes._src.posterior.get_posterior_diagonal

### `BLRDiagState`

::: optax_bayes._src.types.BLRDiagState

---

## Full-Rank BLR

### `blr_full_rank_for_loss`

::: optax_bayes._src.wrappers.blr_full_rank_for_loss

### `blr_full_rank`

::: optax_bayes._src.full_rank.blr_full_rank

### `get_posterior_full_rank`

::: optax_bayes._src.posterior.get_posterior_full_rank

### `BLRFullRankState`

::: optax_bayes._src.types.BLRFullRankState

---

## Low-Rank BLR

### `blr_low_rank_for_loss`

::: optax_bayes._src.wrappers.blr_low_rank_for_loss

### `blr_low_rank`

::: optax_bayes._src.low_rank.blr_low_rank

### `get_posterior_low_rank`

::: optax_bayes._src.posterior.get_posterior_low_rank

### `BLRLowRankState`

::: optax_bayes._src.types.BLRLowRankState
