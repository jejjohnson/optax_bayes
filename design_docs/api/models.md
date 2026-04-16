---
status: draft
version: 0.1.0
---

# Layer 2 — Models

Convenience wrappers and posterior extraction.

---

## Loss Wrappers

### `blr_diagonal_for_loss`

Drop-in replacement for `optax.adam(...)`. Negates gradients internally (standard loss convention).

```python
def blr_diagonal_for_loss(
    learning_rate=1e-3,
    prior_precision=1e-4,
    hessian_estimator="ggn_diag",
    damping=1e-6,
) -> optax.GradientTransformation:
    """BLR-Diagonal for loss minimisation (not log-likelihood)."""
```

### `blr_full_rank_for_loss`

Same for full-rank variant.

---

## Schedule Composition

### `blr_with_schedule`

```python
def blr_with_schedule(
    schedule_fn: optax.Schedule,
    prior_precision=1e-4,
    damping=1e-6,
) -> optax.GradientTransformation:
    """BLR with time-varying learning rate from an optax schedule."""
```

---

## Posterior Extraction

### `get_posterior_diagonal(state) → (mean, variance)`

$$m = \eta / s, \quad v = 1/s$$

Extract the approximate posterior $q(\theta) = \mathcal{N}(m, \operatorname{diag}(v))$ from the optimizer state at any point during training.

### `get_posterior_full_rank(state) → (mean, covariance)`

$$m = \Lambda^{-1}\eta, \quad \Sigma = \Lambda^{-1}$$

Uses `gaussx.ops.solve` for structured precision when available.

---

*For the full mathematical treatment and model zoo, see [../research/overview.md](../research/overview.md).*
