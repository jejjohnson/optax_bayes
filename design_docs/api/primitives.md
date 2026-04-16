---
status: draft
version: 0.1.0
---

# Layer 0 — Primitives

Pure JAX functions implementing the BLR update equations. No optax, no gaussx.

For the full mathematical derivation, see [../research/overview.md](../research/overview.md) §1.

---

## BLR Update Steps

### `blr_diag_update_step(eta, s, g, diag_h, m, eta0, s0, rho)`

**Mathematical definition (Diagonal BLR — Khan & Rue 2023 §3):**

$$s_{t+1} = (1 - \rho)\,s_t + \rho\,(s_0 - \operatorname{diag}(H_t))$$

$$\eta_{t+1} = (1 - \rho)\,\eta_t + \rho\,(\eta_0 + g_t - \operatorname{diag}(H_t) \odot m_t)$$

$$m_{t+1} = \eta_{t+1} / s_{t+1}$$

where $g_t = \mathbb{E}_q[\nabla_\theta \ell]$, $H_t = \mathbb{E}_q[\nabla^2_\theta \ell] \preceq 0$.

The $-H_t m_t$ correction in $\eta$ is the Bonnet-Price identity term — omitting it yields incorrect fixed points.

### `blr_full_update_step(eta, Lambda, g, H, m, eta0, Lambda0, rho)`

**Mathematical definition (Full-rank BLR):**

$$\Lambda_{t+1} = (1 - \rho)\,\Lambda_t + \rho\,(\Lambda_0 - H_t)$$

$$\eta_{t+1} = (1 - \rho)\,\eta_t + \rho\,(\eta_0 + g_t - H_t m_t)$$

$$m_{t+1} = \Lambda_{t+1}^{-1}\,\eta_{t+1}$$

---

## Hessian Estimators

### `ggn_diag(grads)`

$$\operatorname{diag}(\hat{H}) = -g \odot g$$

GGN diagonal approximation. Always NSD. Same cost as Adam's second moment. Vanishes at optima.

### `hutchinson_diag(hvp_fn, key, n_probes)`

$$\operatorname{diag}(H) \approx \frac{1}{n} \sum_{i=1}^n v_i \odot (Hv_i), \quad v_i \sim \text{Rademacher}$$

Unbiased estimator via Hessian-vector products. $O(d)$ per HVP.

### `identity_hessian(grads)`

$H = 0$. Fixes precision to prior. Reduces BLR to preconditioned SGD.

---

## Parameter Conversions

### `natural_to_mean_diag(eta, s) → (m, v)`

$$m = \eta / s, \quad v = 1/s$$

### `mean_to_natural_diag(m, v) → (eta, s)`

$$\eta = m / v, \quad s = 1/v$$
