---
status: draft
version: 0.1.0
---

# Layer 0 — Primitive Examples

The BLR update step by step — pure JAX, no optax.

---

## Diagonal BLR Update (Manual)

```python
from optax_bayes._src.primitives import blr_diag_update_step
from optax_bayes._src.hessians import ggn_diag

# One BLR step — pure function
g = jax.grad(log_likelihood)(theta)
h = ggn_diag(g)  # H ≈ -g², always NSD

eta_new, s_new, m_new = blr_diag_update_step(
    eta=eta, s=s, g=g, diag_h=h, m=m,
    eta0=eta_prior, s0=s_prior, rho=0.01,
)
```

---

## Natural ↔ Mean Conversion

```python
from optax_bayes._src.conversions import natural_to_mean_diag, mean_to_natural_diag

# Convert for interpretation
m, v = natural_to_mean_diag(eta, s)  # m = η/s, v = 1/s

# Convert back for storage
eta, s = mean_to_natural_diag(m, v)  # η = m/v, s = 1/v
```
