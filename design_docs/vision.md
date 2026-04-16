---
status: draft
version: 0.1.0
---

# Vision

## One-Liner

> **optax-bayes** implements the Bayesian Learning Rule (Khan & Rue 2023) as optax `GradientTransformation`s — drop-in replacements for Adam that give posterior uncertainty for free.

---

## Motivation

The Bayesian Learning Rule (BLR) unifies optimisation, inference, and learning under one update rule. SGD, Adam, Newton's method, Kalman filtering, variational inference, and continual learning are all special cases of natural gradient descent on the ELBO with different choices of variational family, Hessian approximation, prior, and learning rate.

The standard implementation pattern — natural-parameter EMA updates on the precision and natural mean — maps perfectly to optax's `init → update → state` protocol. The optimizer state stores the natural parameters $(\eta, s)$ or $(\eta, \Lambda)$; the update step is the BLR convex combination; posterior extraction recovers $(m, \Sigma)$ at any time.

optax-bayes provides this as a library: plug in `blr_diagonal_for_loss()` where you'd use `optax.adam()`, and get calibrated parameter uncertainty at convergence with zero extra code.

---

## User Stories

**ML researcher** — "I want Adam-like training speed but with per-parameter uncertainty at convergence. I shouldn't need to run a separate inference step after training."

**Bayesian practitioner** — "I want natural-gradient VI as an optax optimizer that composes with `optax.chain`, gradient clipping, and learning rate schedules."

**Continual learning researcher** — "I want EWC-style regularisation that falls out naturally from the BLR framework — the posterior from task 1 becomes the prior for task 2."

**GP researcher** — "I want natural-gradient variational inference for GP classification via full-rank BLR, with gaussx handling the structured precision matrix."

---

## Design Principles

1. **Optax-native** — Every BLR variant is an `optax.GradientTransformation` with `init(params) → state` and `update(grads, state) → (updates, state)`. Composes with `optax.chain`, schedules, clipping.

2. **Natural-parameter state** — Optimizer state stores natural parameters $(\eta, s)$ or $(\eta, \Lambda)$, not moments $(m, v)$. This avoids catastrophic cancellation and makes the EMA structure explicit.

3. **Posterior extraction** — `get_posterior_diagonal(state) → (mean, variance)` at any time during training. The posterior is always available — not an afterthought.

4. **gaussx-backed** — Full-rank and low-rank BLR variants delegate structured precision operations (solve, logdet, Woodbury) to gaussx. Diagonal BLR is self-contained.

5. **Log-likelihood convention** — Core functions expect $\nabla_\theta \log p(\mathcal{D}|\theta)$. A `_for_loss` wrapper negates gradients for standard loss minimisation convention.

---

## Identity

### What optax-bayes IS

- BLR-Diagonal: drop-in Adam replacement with uncertainty (same memory overhead)
- BLR-FullRank: natural-gradient VI with exact posterior covariance
- BLR-LowRank: scalable uncertainty via $\Lambda = D + UU^T$ (Woodbury)
- Posterior extraction from optimizer state at any time
- Composable with optax ecosystem (chain, schedules, clipping)
- gaussx-backed for structured precision (full-rank, low-rank)

### What optax-bayes is NOT

| Not this | Use instead |
|----------|-------------|
| Custom inference engine (MCMC, SVI) | NumPyro |
| General exponential family BLR | efax (if non-Gaussian families needed) |
| Neural network architecture | Equinox |
| GP inference | pyrox-gp (uses optax-bayes for natural-gradient VI) |
| Ensemble Kalman methods | ekalmX |
| Training loops | User code (optax-bayes provides the optimizer, not the loop) |

---

## Migration Context

### External

| Tool | What optax-bayes provides instead |
|------|-----------------------------------|
| IVON (Shen et al. 2024) | Same BLR-Diagonal algorithm, but as a composable optax transform |
| VOGN (Osawa et al. 2019) | Same structured BLR, but JAX-native with gaussx backend |
| Adam + Laplace post-hoc | BLR gives uncertainty *during* training, not as an afterthought |
| Bayes by Backprop | BLR with GGN diagonal ≈ same algorithm, cleaner derivation |

### Key References

- Khan, M.E. & Rue, H. (2023). "The Bayesian Learning Rule." *JMLR* 24(281):1–46.
- Shen, R. et al. (2024). "Variational Learning is Effective for Large Deep Networks." *ICML* (IVON).
- Osawa, K. et al. (2019). "Practical Deep Learning with Bayesian Principles." *NeurIPS* (VOGN).
- Möllenhoff, T. & Khan, M.E. (2023). "SAM as an Optimal Relaxation of Bayes." *ICLR*.

---

## Connection to Ecosystem

```
                    ┌──────────────┐
                    │   optax      │  Optimizer protocol (init → update → state)
                    └──────┬───────┘
                           │ extends
                    ┌──────▼───────┐
                    │ optax-bayes  │  BLR as GradientTransformation
                    │              │  (diagonal, full-rank, low-rank)
                    └──┬───────┬──┘
                       │       │
            ┌──────────┘       └──────────┐
     ┌──────▼──────┐              ┌──────▼──────┐
     │   gaussx    │              │   user code │
     │ (expfam,    │              │ (training   │
     │  precision  │              │  loops)     │
     │  operators) │              │             │
     └─────────────┘              └─────────────┘

Consumed by:
    pyrox-gp (natural-gradient VI for GP classification)
    ekalmX (connection: EKP ↔ BLR — both are natural gradient updates)
    xtremax (BLR for GEV parameter estimation with uncertainty)

Parallel to ekalmX:
    ekalmX = derivative-free (ensemble) natural gradient
    optax-bayes = gradient-based (analytic) natural gradient
    Both fit the optax init → update → state pattern.
```
