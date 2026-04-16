---
status: draft
version: 0.1.0
---

# Layer 3 — Integration Examples

Ecosystem composition: gaussx, pyrox-gp, ekalmX, xtremax.

---

## With gaussx (Structured Precision for Full-Rank BLR)

```python
from gaussx.expfam import GaussianExpFam
from gaussx.operators import low_rank_plus_diag
import gaussx.ops as gops

# Low-rank precision: Λ = diag(d) + UUᵀ
Lambda_op = low_rank_plus_diag(W=U, d=d)

# Recover mean via Woodbury — O(dr² + r³) not O(d³)
m = gops.solve(Lambda_op, eta)

# Log-normalizer via matrix determinant lemma
log_det = gops.logdet(Lambda_op)
```

---

## With pyrox-gp (GP Classification via Natural Gradient VI)

```python
from optax_bayes import blr_full_rank, get_posterior_full_rank

# GP prior: Λ₀ = K⁻¹
K = rbf_kernel(X, X) + 1e-6 * jnp.eye(N)
K_inv = jnp.linalg.inv(K)

opt = blr_full_rank(learning_rate=0.2, prior_precision=K_inv)
state = opt.init(jnp.zeros(N))

for _ in range(100):
    g = y - jax.nn.sigmoid(f)       # Bernoulli log-lik gradient
    H = -jnp.diag(p * (1 - p))      # Bernoulli Hessian
    updates, state = opt.update(g, state)
    f = f + updates

mean, cov = get_posterior_full_rank(state)  # approximate GP posterior
```

---

## Parallel with ekalmX

Both optax-bayes and ekalmX implement natural gradient updates — one gradient-based, one derivative-free:

| | optax-bayes (BLR) | ekalmX (EKP) |
|---|---|---|
| Gradient source | $\nabla_\theta \ell$ (analytic) | $C^{\theta G}$ (ensemble, derivative-free) |
| Hessian source | $\nabla^2_\theta \ell$ or GGN | $C^{GG}$ (ensemble covariance) |
| Scaling in $d$ | $O(d)$ diagonal, $O(d^2)$ full | $O(Jd)$ where $J$ = ensemble size |
| Best for | Differentiable models | Black-box simulators |
| optax interface | `blr_diagonal(...)` | `ekalmx.optax.eki(...)` |

Both fit the optax `init → update → state` pattern and compose with `optax.chain`.

---

## With xtremax (GEV Parameter Estimation with Uncertainty)

```python
from optax_bayes import blr_diagonal_for_loss, get_posterior_diagonal
from xtremax._src.gev import gev_log_prob

def gev_nll(params, data):
    loc, log_scale, shape = params
    return -jnp.sum(gev_log_prob(data, loc, jnp.exp(log_scale), shape))

opt = blr_diagonal_for_loss(learning_rate=1e-2, prior_precision=1e-2)
state = opt.init(jnp.array([30.0, jnp.log(2.5), 0.1]))

for _ in range(500):
    grads = jax.grad(gev_nll)(params, annual_maxima)
    updates, state = opt.update(grads, state)
    params = optax.apply_updates(params, updates)

# Posterior uncertainty on GEV shape parameter — critical for return levels
mean, var = get_posterior_diagonal(state)
shape_std = jnp.sqrt(var[2])  # uncertainty on ξ
```

---

## Composition Patterns

| Pattern | Components | Use Case |
|---|---|---|
| Drop-in uncertainty | `blr_diagonal_for_loss` | Adam replacement with free posterior |
| Continual learning | Task-1 posterior → task-2 prior | EWC without extra code |
| GP classification | `blr_full_rank` + GP prior precision | Natural-gradient VI |
| Structured uncertainty | `blr_low_rank` + gaussx Woodbury | Scalable posterior covariance |
| GEV with uncertainty | `blr_diagonal` + xtremax L0 functions | Tail parameter uncertainty |
| Hybrid with ekalmX | BLR (differentiable) + EKP (black-box) | Combined gradient + ensemble |
