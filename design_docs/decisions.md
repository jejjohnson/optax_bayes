---
status: draft
version: 0.1.0
---

# Design Decisions

---

## D1: optax GradientTransformation, not custom optimizer

**Status:** accepted

**Context:** BLR could be implemented as a standalone optimizer class or as an optax GradientTransformation.

**Decision:** optax GradientTransformation. This gives free composition with `optax.chain`, schedules, clipping, `inject_hyperparams`, and the entire optax ecosystem.

**Consequences:** Users call `optax.apply_updates` as usual. The BLR update is just another transform in the chain.

---

## D2: Natural-parameter state, not moment-parameter

**Status:** accepted

**Context:** The optimizer state could store $(m, v)$ (moment parameters) or $(\eta, s)$ (natural parameters).

**Decision:** Natural parameters. The BLR update is a convex combination in natural-parameter space. Storing moments would require converting back and forth, introducing cancellation errors.

**Consequences:** `get_posterior_*` functions needed to convert state → (mean, variance) for user consumption.

---

## D3: Log-likelihood convention with `_for_loss` wrappers

**Status:** accepted

**Context:** The BLR math is naturally expressed in terms of $\nabla_\theta \log p(\mathcal{D}|\theta)$ (gradients of log-likelihood). Standard ML convention minimizes a loss $L = -\log p$.

**Decision:** Core functions (`blr_diagonal`, `blr_full_rank`) expect log-likelihood gradients. `_for_loss` wrappers negate gradients for the standard loss minimization convention.

**Consequences:** Two entry points per variant. The `_for_loss` version is the one most users reach for.

---

## D4: gaussx for structured precision (not hand-rolled)

**Status:** accepted

**Context:** Full-rank BLR needs $\Lambda^{-1}\eta$ (precision solve) and $\log|\Lambda|$ (log-normalizer). Low-rank needs Woodbury.

**Decision:** Delegate to gaussx. `gaussx.ops.solve`, `gaussx.ops.logdet`, `gaussx.operators.LowRankUpdate` handle all structured precision operations.

**Consequences:** gaussx is an optional dependency (required only for full-rank and low-rank variants). Diagonal BLR is self-contained.

---

## D5: Diagonal BLR is the primary entry point

**Status:** accepted

**Context:** Which variant should be the default / most prominent?

**Decision:** BLR-Diagonal. Same memory overhead as Adam (one extra array per parameter). Drop-in replacement. Full-rank and low-rank are for users who need richer posterior structure.

**Consequences:** Most examples and docs lead with the diagonal case.

---

## Resolved Questions

| Question | Resolution |
|---|---|
| Optimizer protocol | optax GradientTransformation (D1) |
| State parameterization | Natural parameters (D2) |
| Gradient convention | Log-likelihood + `_for_loss` wrappers (D3) |
| Structured precision backend | gaussx (D4) |
| Default variant | BLR-Diagonal (D5) |
| Non-Gaussian families | Defer to efax; optax-bayes is Gaussian-only |
| Per-parameter prior | Open question — pytree vs flatten |
