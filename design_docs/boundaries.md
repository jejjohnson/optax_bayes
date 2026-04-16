---
status: draft
version: 0.1.0
---

# Boundaries and Ecosystem

## Overview

optax-bayes owns the BLR update rule as optax GradientTransformations. It delegates structured linear algebra to gaussx, optimizer protocol to optax, and inference to downstream consumers.

---

## Ownership Map

| Concern | Owner | Notes |
|---------|-------|-------|
| BLR update rule (diagonal, full-rank, low-rank) | **optax-bayes** | Core competency |
| optax GradientTransformation wrappers | **optax-bayes** | L1 |
| Posterior extraction from optimizer state | **optax-bayes** | L2 |
| Hessian estimators (GGN diag, Hutchinson, identity) | **optax-bayes** | L0 |
| Loss convention wrappers (_for_loss) | **optax-bayes** | L2 |
| Structured precision operators | **gaussx** | Full-rank/low-rank BLR delegates |
| Gaussian exponential family | **gaussx** | `GaussianExpFam` for natural params |
| optax protocol + composition | **optax** | chain, schedules, clipping |
| Training loops | **user** | optax-bayes provides the optimizer |
| GP inference | **pyrox-gp** | Uses optax-bayes for natural-gradient VI |
| Ensemble Kalman methods | **ekalmX** | Parallel: derivative-free natural gradient |

---

## Decision Table

| Scenario | Recommendation |
|----------|---------------|
| Drop-in Adam replacement with uncertainty | `blr_diagonal_for_loss(lr=1e-3, prior_precision=1e-4)` |
| Full posterior covariance (small d) | `blr_full_rank(lr=0.1, hessian_fn=ggn)` |
| Scalable uncertainty (large d, low-rank) | `blr_low_rank(lr=1e-2, rank=10)` |
| Compose with gradient clipping | `optax.chain(optax.clip_by_global_norm(1.0), blr_diagonal(...))` |
| Continual learning (EWC-style) | Use task-1 posterior as task-2 prior |
| GP classification (natural gradient VI) | `blr_full_rank` with GP prior precision $\Lambda_0 = K^{-1}$ |
| Kalman filtering (ρ=1, sequential) | `blr_full_rank` with ρ=1 and observation Hessian |

---

## Scope

### In Scope

- BLR-Diagonal (Adam-like, per-parameter uncertainty)
- BLR-FullRank (natural-gradient VI, exact posterior covariance)
- BLR-LowRank ($\Lambda = D + UU^T$, Woodbury-based)
- Hessian estimators (GGN diagonal, Hutchinson, identity, custom)
- Posterior extraction (mean, variance/covariance) from state
- Loss convention wrappers and schedule composition
- gaussx backend for structured precision

### Out of Scope

- Non-Gaussian exponential families — efax (if needed)
- Custom MCMC/SVI inference — NumPyro
- Training loops — user code
- Neural network architecture — Equinox

---

## Testing Strategy

| Category | What it tests | Example |
|----------|---------------|---------|
| **Update correctness** | BLR step matches hand-derived formula | Diagonal update on quadratic recovers exact posterior |
| **Adam recovery** | BLR-Diagonal with identity Hessian ≈ SGD | Same loss trajectory as `optax.sgd` |
| **Kalman recovery** | BLR-FullRank with ρ=1 = Kalman information filter | Matches `pyrox_gp.kalman_update` |
| **PSD preservation** | Precision stays positive definite | Damping prevents collapse on adversarial Hessians |
| **optax composition** | Works with `optax.chain`, schedules, clipping | Compose with `clip_by_global_norm` |
| **Posterior extraction** | `get_posterior_*` recovers correct (mean, var) | Check against manual $m = \eta/s$ |

### Test Priorities

1. **Update correctness** — BLR step matches the math on simple problems
2. **optax protocol** — init/update/apply_updates work with standard optax patterns
3. **Special case recovery** — diagonal BLR recovers Adam/SGD, full-rank recovers Kalman
4. **PSD preservation** — precision never goes non-positive

---

## Roadmap

| Phase | Focus | Depends on |
|-------|-------|------------|
| v0.1 | BLR-Diagonal + `_for_loss` wrapper + posterior extraction | optax |
| v0.2 | BLR-FullRank with gaussx backend | v0.1 + gaussx |
| v0.3 | BLR-LowRank ($D + UU^T$) via gaussx Woodbury | v0.2 |
| v0.4+ | Schedule composition, Hutchinson Hessian, K-FAC blocks | v0.1 |

---

## Open Questions

1. **Per-parameter prior** — For continual learning, the prior precision should be a pytree matching the parameter tree. Should `prior_precision` accept pytrees natively, or require flattening?

2. **Hessian interface** — Should the Hessian estimator be a function `(params, grads) → diag_H`, a string selector, or a protocol class?

3. **gaussx vs efax** — gaussx for Gaussian-only BLR, efax for general exponential families. Should optax-bayes support both backends, or commit to gaussx?
