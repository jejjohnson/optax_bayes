# The Bayesian Learning Rule: Design Document

**A Unified Framework for Optimisation, Inference, and Learning**

Based on Khan & Rue (2023), "The Bayesian Learning Rule," *JMLR* 24(281):1–46.

---

## Table of Contents

1. [General Mathematical Formulation](#1-general-mathematical-formulation)
2. [Numerical Requirements](#2-numerical-requirements)
3. [Model Zoo](#3-model-zoo)
4. [Optax API Design](#4-optax-api-design)
5. [Example Applications](#5-example-applications)
6. [Efax Backend Integration](#6-efax-backend-integration) (includes §6.9 GaussX as Gaussian expfam backend)
7. [References](#7-references)

---

## 1. General Mathematical Formulation

### 1.1 The Variational Objective

Instead of finding a point estimate θ\*, we maintain a variational distribution q(θ|λ) from an exponential family and minimise:

```
ℒ(λ) = −𝔼_q[log p(𝒟|θ)] + KL(q(θ|λ) ‖ p(θ))
```

where λ are the natural parameters of q, p(θ) is the prior, and ℓ(θ) = log p(𝒟|θ) is the log-likelihood.

### 1.2 Exponential Family Setup

The variational family takes the canonical form:

```
q(θ|λ) = h(θ) exp(λᵀ T(θ) − A(λ))
```

with the standard correspondences:

```
Natural parameters:      λ
Sufficient statistics:   T(θ)
Log-partition function:  A(λ)
Expectation parameters:  μ = 𝔼_q[T(θ)] = ∇_λ A(λ)
Fisher information:      F(λ) = ∇²_λ A(λ) = ∂μ/∂λ
```

The map λ ↔ μ is a diffeomorphism for minimal exponential families. The key identity relating natural and expectation gradients is:

```
F(λ)⁻¹ ∇_λ f(λ) = ∇_μ f(λ(μ))
```

### 1.3 The Bayesian Learning Rule

Starting from natural gradient ascent on the ELBO:

```
λ_{t+1} = λ_t + ρ_t F(λ_t)⁻¹ ∇_λ ELBO(λ_t)
```

and expanding ∇_λ ELBO = ∇_λ 𝔼_q[ℓ(θ)] − F(λ_t)(λ_t − λ₀), we apply F⁻¹ to obtain:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  λ_{t+1} = (1 − ρ) λ_t  +  ρ [λ₀ + ∇_μ 𝔼_q[ℓ(θ)]]     │
│                                                             │
│  where  ℓ(θ) = log p(𝒟|θ)           log-likelihood        │
│         λ₀   = prior natural params                        │
│         ρ    = learning rate (step size)                    │
│         ∇_μ  = gradient w.r.t. expectation parameters      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

This is a convex combination: at each step we interpolate between the current posterior λ_t and a "target" λ̃ = λ₀ + ∇_μ 𝔼_q[ℓ(θ)].

### 1.4 Gaussian Expectation-Parameter Gradients

For q(θ) = 𝒩(m, Σ), the expectation parameters are μ₁ = m and μ₂ = mmᵀ + Σ. Since Σ = μ₂ − μ₁μ₁ᵀ, changing μ₁ while holding μ₂ fixed implicitly changes Σ. The chain rule, combined with the Bonnet–Price identities (∂𝔼[ℓ]/∂m = 𝔼[∇_θ ℓ] and ∂𝔼[ℓ]/∂Σ = ½ 𝔼[∇²_θ ℓ]), yields:

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  ∇_{μ₁} 𝔼_q[ℓ] = g − H m       g = 𝔼_q[∇_θ ℓ(θ)]    │
│  ∇_{μ₂} 𝔼_q[ℓ] = ½ H           H = 𝔼_q[∇²_θ ℓ(θ)]   │
│                                                          │
│  ⚠ The −Hm correction in ∇_{μ₁} is essential.          │
│    Omitting it yields incorrect fixed points.            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

The derivation: ∂Σ/∂μ₁ = −2m (from Σ = μ₂ − μ₁μ₁ᵀ), so the correction is (½H)(−2m) = −Hm.

### 1.5 Diagonal Gaussian Specialisation

For q(θ) = 𝒩(m, diag(v)) with s = 1/v (diagonal precision):

```
Natural parameters:     λ₁ = s ⊙ m,    λ₂ = −s/2
Expectation parameters: μ₁ = m,         μ₂ = m² + v
```

The BLR updates become:

```
Precision:     s_{t+1} = (1−ρ) s_t  + ρ [s₀ − diag(H_t)]
Natural mean:  η_{t+1} = (1−ρ) η_t  + ρ [η₀ + g_t − diag(H_t) ⊙ m_t]
Recover mean:  m_{t+1} = η_{t+1} / s_{t+1}
```

where η₀ = s₀ ⊙ m₀ and s₀ = 1/v₀ are the prior natural parameters.

### 1.6 Full-Rank Gaussian Specialisation

For q(θ) = 𝒩(m, Σ) with Λ = Σ⁻¹ and η = Λm:

```
Precision:     Λ_{t+1} = (1−ρ) Λ_t  + ρ [Λ₀ − H_t]
Natural mean:  η_{t+1} = (1−ρ) η_t  + ρ [η₀ + g_t − H_t m_t]
Recover mean:  m_{t+1} = solve(Λ_{t+1}, η_{t+1})
```

### 1.7 Low-Rank Gaussian Specialisation

For scalable uncertainty, parameterise Σ⁻¹ = D + UUᵀ where D is diagonal (d parameters) and U is d × r (dr parameters), giving O(dr) cost instead of O(d²):

```
Precision:  Λ = D + UUᵀ
Sampling:   Use the Woodbury identity: Λ⁻¹ = D⁻¹ − D⁻¹U(I + UᵀD⁻¹U)⁻¹UᵀD⁻¹
Solve:      Via Woodbury, O(dr² + r³) per step instead of O(d³)
```

The natural parameter updates for (D, U) follow from projecting the full-rank BLR update onto the low-rank manifold.

---

## 2. Numerical Requirements

### 2.1 Positive-Definiteness of the Precision

The precision s (diagonal) or Λ (full-rank) must remain positive (definite) throughout optimisation. This can be violated when H has large positive eigenvalues (i.e., the log-likelihood is locally convex — unusual but possible in non-convex landscapes).

**Safeguards:**

```
Diagonal:    s_{t+1} ← max(s_{t+1}, ε)           where ε ≈ 1e-6
Full-rank:   Λ_{t+1} ← Λ_{t+1} + ε I             additive damping
             or: Λ_{t+1} ← project_pd(Λ_{t+1})   spectral clipping
```

### 2.2 Hessian Approximations

The exact Hessian ∇²_θ ℓ is often unavailable or expensive. Practical approximations, each with distinct trade-offs:

```
┌────────────────────────┬──────────────────────────────────────────────────┐
│ Method                 │ Properties                                       │
├────────────────────────┼──────────────────────────────────────────────────┤
│ Exact Hessian          │ H = ∇²_θ ℓ(θ).  O(d²) storage, O(d³) solve.   │
│                        │ Not guaranteed NSD for non-convex ℓ.            │
│                        │                                                  │
│ GGN diagonal (−g²)     │ H ≈ −g ⊙ g.  Always NSD.  O(d) storage.       │
│                        │ Equivalent to Adam's second moment.             │
│                        │ Vanishes at optima (g → 0), so precision        │
│                        │ collapses — use only for optimiser mode.        │
│                        │                                                  │
│ Generalised Gauss–     │ H ≈ −Jᵀ J  where J is the Jacobian of the     │
│ Newton (GGN)           │ residuals.  Always NSD.  Exact for              │
│                        │ exponential-family likelihoods.                  │
│                        │                                                  │
│ Empirical Fisher       │ H ≈ −𝔼_data[g gᵀ].  Mini-batch average of     │
│                        │ outer products.  Always NSD.  Cheap.            │
│                        │                                                  │
│ Hutchinson diagonal    │ diag(H) ≈ v ⊙ (Hv)  where v ~ Rademacher.     │
│                        │ Unbiased estimator of diagonal.  O(d) cost      │
│                        │ per HVP.  May need multiple samples.            │
│                        │                                                  │
│ Identity (H = 0)       │ Fixes precision to prior.  Reduces BLR to      │
│                        │ preconditioned SGD.                              │
└────────────────────────┴──────────────────────────────────────────────────┘
```

### 2.3 Monte Carlo Estimation

The expectations 𝔼_q[∇_θ ℓ] and 𝔼_q[∇²_θ ℓ] are estimated by sampling θ̂ = m + Lε, ε ∼ 𝒩(0, I), where Σ = LLᵀ.

**Zero-variance limit:** Setting θ̂ = m (no sampling noise) recovers deterministic gradient/Hessian evaluation at the current mean. This is standard practice when using BLR as an optimiser and is the default mode in the implementation.

**Single-sample MC:** One draw per step. High variance but unbiased. The EMA structure of BLR provides natural smoothing.

**Multi-sample MC:** K draws, average the gradients. Reduces variance by 1/K. Useful for stochastic likelihoods.

### 2.4 Numerical Precision Considerations

```
Issue                     Mitigation
─────────────────────────────────────────────────────────────────
Catastrophic cancellation Work in natural parameter space (λ₁, λ₂)
in m = η/s when both      directly, recover m only when needed.
are large

Overflow in precision     Log-space precision: store log(s) instead
for very peaked posteriors of s.  Update: log s_{new} = logsumexp(...)

Cholesky failure in       Jitter: Λ + εI before factorisation.
full-rank solve           Fall back to eigendecomposition if needed.

Underflow in variance     Clamp: v = max(1/s, v_min) with
1/s for large precision   v_min ≈ 1e-30 (float32) or 1e-300 (float64).

Stiff learning rate       Per-parameter adaptive ρ via Optax schedule
for mixed-scale params    or group-wise BLR instances.
```

### 2.5 JAX-Specific Considerations

```
Requirement               Implementation Note
─────────────────────────────────────────────────────────────────
jit-compatibility         All updates are pure functions of
                          (grads, state, params). No Python
                          side-effects or data-dependent control
                          flow in the update loop.

vmap over MC samples      The reparameterised sample θ̂ = m + Lε
                          is naturally vmappable over ε.

scan for training loops   BLRDiagState is a NamedTuple of arrays,
                          compatible with jax.lax.scan carry.

Pytree compatibility      State fields are pytrees matching the
                          parameter tree structure (diagonal case).
                          Full-rank requires a flat vector — use
                          jax.flatten_util for pytree ↔ flat.

float64                   Enable with jax.config.update(
                          "jax_enable_x64", True) when precision
                          matters (GPs, PDEs, extreme-value models).
```

---

## 3. Model Zoo

The BLR unifies a wide family of algorithms as special cases. Each arises from a specific choice of variational family, Hessian approximation, prior, learning rate, and whether precision is updated.

### 3.1 Optimisation Algorithms

```
┌─────────────────────┬───────────────┬──────────────┬────────┬───────────────┐
│ Algorithm           │ Family        │ Hessian      │ ρ      │ Prior         │
├─────────────────────┼───────────────┼──────────────┼────────┼───────────────┤
│ SGD                 │ Diag Gaussian │ Identity     │ 1      │ Flat (s₀→0)  │
│                     │ (fixed var)   │ (H = 0)     │        │               │
│                     │               │              │        │               │
│ SGD + momentum      │ Diag Gaussian │ Identity     │ 1−β    │ Flat          │
│                     │ (fixed var)   │              │        │               │
│                     │               │              │        │               │
│ Adam                │ Diag Gaussian │ −g²          │ β₁,β₂ │ Flat          │
│ (approximate)       │               │ (GGN diag)   │        │               │
│                     │               │              │        │               │
│ Newton's method     │ Full-rank     │ Exact        │ 1      │ Flat          │
│                     │ Gaussian      │ Hessian      │        │               │
│                     │               │              │        │               │
│ Natural gradient    │ Full-rank     │ Fisher       │ ρ      │ Flat          │
│ descent             │ Gaussian      │              │        │               │
│                     │               │              │        │               │
│ L-BFGS (approx)    │ Low-rank      │ Curvature    │ 1      │ Flat          │
│                     │ Gaussian      │ pairs        │        │               │
│                     │               │              │        │               │
│ AdaGrad             │ Diag Gaussian │ Cumulative   │ varies │ Flat          │
│                     │ (growing s)   │ −g²          │        │               │
│                     │               │              │        │               │
│ RMSProp             │ Diag Gaussian │ EMA of −g²  │ varies │ Flat          │
│                     │               │              │        │               │
│ Weight decay / L2   │ Any           │ Any          │ ρ      │ 𝒩(0,1/λ I)  │
│ regularisation      │               │              │        │ (s₀ = λ)     │
└─────────────────────┴───────────────┴──────────────┴────────┴───────────────┘
```

### 3.2 Inference Algorithms

```
┌─────────────────────┬───────────────┬──────────────┬────────┬───────────────┐
│ Algorithm           │ Family        │ Hessian      │ ρ      │ Prior         │
├─────────────────────┼───────────────┼──────────────┼────────┼───────────────┤
│ Mean-field VI       │ Diag Gaussian │ Exact diag   │ ρ_t    │ Specified     │
│ (MFVI)              │               │              │        │               │
│                     │               │              │        │               │
│ Full-rank VI        │ Full-rank     │ Exact / GGN  │ ρ_t    │ Specified     │
│                     │ Gaussian      │              │        │               │
│                     │               │              │        │               │
│ Laplace approx      │ Full-rank     │ Exact        │ 1      │ Specified     │
│                     │ Gaussian      │ Hessian      │ (one   │               │
│                     │               │              │  step) │               │
│                     │               │              │        │               │
│ Kalman filter       │ Full-rank     │ Observation  │ 1      │ Prediction    │
│                     │ Gaussian      │ model        │        │ prior         │
│                     │               │              │        │               │
│ Extended Kalman     │ Full-rank     │ Linearised   │ 1      │ Prediction    │
│ filter (EKF)        │ Gaussian      │ obs. model   │        │ prior         │
│                     │               │              │        │               │
│ Assumed-density     │ Diag or full  │ Moment-      │ 1      │ Sequential    │
│ filtering (ADF)     │ Gaussian      │ matched      │        │ update        │
│                     │               │              │        │               │
│ Expectation         │ Exponential   │ Natural      │ 1      │ Specified     │
│ propagation (EP)    │ family        │ gradient     │        │               │
│                     │               │              │        │               │
│ Stochastic VI       │ Any exp.      │ MC estimate  │ ρ_t    │ Specified     │
│ (SVI / BBVI)        │ family        │              │        │               │
└─────────────────────┴───────────────┴──────────────┴────────┴───────────────┘
```

### 3.3 Learning Algorithms

```
┌─────────────────────┬───────────────┬──────────────┬────────┬───────────────┐
│ Algorithm           │ Family        │ Hessian      │ ρ      │ Prior         │
├─────────────────────┼───────────────┼──────────────┼────────┼───────────────┤
│ Online learning     │ Any           │ Any          │ 1/t    │ Previous      │
│                     │               │              │        │ posterior     │
│                     │               │              │        │               │
│ Continual learning  │ Diag or full  │ Fisher       │ ρ      │ Previous task │
│ (EWC-style)         │ Gaussian      │              │        │ posterior     │
│                     │               │              │        │               │
│ Bayesian neural     │ Diag Gaussian │ GGN / −g²   │ ρ      │ 𝒩(0, σ²I)   │
│ network (BNN)       │               │              │        │               │
│                     │               │              │        │               │
│ Natural-gradient    │ Low-rank or   │ K-FAC / GGN  │ ρ      │ Specified     │
│ VI for deep models  │ structured    │              │        │               │
│ (VOGN, IVON)        │ Gaussian      │              │        │               │
└─────────────────────┴───────────────┴──────────────┴────────┴───────────────┘
```

### 3.4 Pseudocode

**Algorithm: BLR-Diagonal**

```
Input:  ℓ(θ), m₀, v₀, ρ, T
Init:   s ← 1/v₀,  η ← s ⊙ m₀,  m ← m₀

for t = 1, …, T:
    ε  ∼ 𝒩(0, I)
    θ̂  ← m + ε / √s                      reparameterised sample
    g  ← ∇_θ ℓ(θ̂)                        gradient of log-likelihood
    h  ← diag(∇²_θ ℓ(θ̂))                 diagonal Hessian (≤ 0)

    s   ← (1−ρ) s  + ρ (1/v₀ − h)        precision update
    η   ← (1−ρ) η  + ρ (m₀/v₀ + g − h⊙m) natural mean update
    m   ← η / s                            recover mean

return m, 1/s                              posterior mean & variance
```

**Algorithm: BLR-FullRank**

```
Input:  ℓ(θ), m₀, Λ₀, ρ, T
Init:   Λ ← Λ₀,  η ← Λ₀ m₀,  m ← m₀

for t = 1, …, T:
    θ̂  ← sample from 𝒩(m, Λ⁻¹)
    g  ← ∇_θ ℓ(θ̂)
    H  ← ∇²_θ ℓ(θ̂)                       full Hessian or GGN

    Λ  ← (1−ρ) Λ  + ρ (Λ₀ − H)           precision update
    η  ← (1−ρ) η  + ρ (Λ₀ m₀ + g − H m)  natural mean update
    m  ← solve(Λ, η)                       recover mean

return m, Λ⁻¹                              posterior mean & covariance
```

---

## 4. Optax API Design

### 4.1 Design Principles

The implementation follows these principles:

1. **Optax-native.** Every BLR variant returns a `GradientTransformation` with `init(params) → state` and `update(grads, state, params) → (updates, state)`. Users call `optax.apply_updates` as usual.

2. **Natural-parameter state.** The optimizer state stores the natural parameters (η, s) or (η, Λ), not the moment parameters (m, v). This avoids catastrophic cancellation and makes the EMA structure explicit.

3. **Log-likelihood convention.** The core functions expect ∇_θ log p(𝒟|θ). A `_for_loss` wrapper negates gradients for standard loss minimisation, serving as a drop-in replacement for `optax.adam`.

4. **Posterior extraction.** Dedicated functions `get_posterior_diagonal` and `get_posterior_full_rank` recover (mean, variance/covariance) from the optimizer state at any time.

5. **Composability.** BLR optimizers compose with `optax.chain`, `optax.inject_hyperparams`, gradient clipping, and learning rate schedules.

### 4.2 Public API Surface

```python
# ── Core optimisers (expect log-likelihood gradients) ──────────────
blr_diagonal(
    learning_rate=1e-2,
    prior_precision=1e-4,          # s₀ = 1/v₀ (scalar, broadcast)
    prior_mean=0.0,                # m₀ (scalar, broadcast)
    hessian_estimator="ggn_diag",  # "ggn_diag" | "identity"
    damping=1e-6,                  # precision floor
) → GradientTransformation

blr_full_rank(
    learning_rate=1e-2,
    prior_precision=1e-4,          # scalar × I
    prior_mean=None,               # array or None (→ zeros)
    damping=1e-6,
    hessian_fn=None,               # θ → ∇²ℓ(θ), or None → rank-1 GGN
) → GradientTransformation

blr_low_rank(                      # planned
    learning_rate=1e-2,
    rank=10,
    prior_precision=1e-4,
    damping=1e-6,
) → GradientTransformation

# ── Loss wrappers (accept ∇_θ Loss, standard Optax convention) ────
blr_diagonal_for_loss(...)  → GradientTransformation
blr_full_rank_for_loss(...) → GradientTransformation

# ── Schedule composition ──────────────────────────────────────────
blr_with_schedule(
    schedule_fn,                   # optax.Schedule
    prior_precision=1e-4,
    damping=1e-6,
) → GradientTransformation

# ── Posterior extraction ──────────────────────────────────────────
get_posterior_diagonal(state: BLRDiagState) → (mean, variance)
get_posterior_full_rank(state: BLRFullRankState) → (mean, covariance)

# ── State types ───────────────────────────────────────────────────
class BLRDiagState(NamedTuple):
    precision: Params   # s = 1/v, pytree
    nat_mean:  Params   # η = s⊙m, pytree
    count:     Array    # step counter

class BLRFullRankState(NamedTuple):
    precision: Array    # Λ = Σ⁻¹, shape (d, d)
    nat_mean:  Array    # η = Λm,  shape (d,)
    count:     Array
```

### 4.3 Usage Examples

**Drop-in replacement for Adam:**

```python
import optax
from bayesian_learning_rule import blr_diagonal_for_loss, get_posterior_diagonal

opt = blr_diagonal_for_loss(learning_rate=1e-3, prior_precision=1e-4)
state = opt.init(params)

for batch in dataloader:
    grads = jax.grad(loss_fn)(params, batch)
    updates, state = opt.update(grads, state)
    params = optax.apply_updates(params, updates)

# Extract uncertainty at any time
mean, variance = get_posterior_diagonal(state)
```

**With learning rate schedule:**

```python
from bayesian_learning_rule import blr_with_schedule

schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-4, peak_value=1e-2,
    warmup_steps=1000, decay_steps=50000,
)
opt = blr_with_schedule(schedule, prior_precision=1e-3)
```

**Chain with gradient clipping:**

```python
opt = optax.chain(
    optax.clip_by_global_norm(1.0),
    blr_diagonal_for_loss(learning_rate=1e-3),
)
```

**Full-rank with custom Hessian (e.g., GGN for a neural net):**

```python
from bayesian_learning_rule import blr_full_rank

def ggn_hessian(theta):
    """Gauss–Newton: H = −Jᵀ diag(p(1−p)) J for logistic regression."""
    J = jax.jacobian(model.apply)(theta, X)       # (N, d)
    p = jax.nn.sigmoid(model.apply(theta, X))      # (N,)
    W = p * (1 - p)                                 # (N,)
    return -(J.T * W) @ J                           # (d, d), NSD

opt = blr_full_rank(learning_rate=0.1, hessian_fn=ggn_hessian)
```

---

## 5. Example Applications

### 5.1 Parameter Estimation (MLE / MAP)

**Setting:** Estimate parameters θ of a statistical model from data 𝒟.

**BLR framing:** Use `blr_diagonal_for_loss` as a drop-in optimizer. The posterior variance gives calibrated uncertainty on each parameter "for free."

```python
# Logistic regression: θ = (w, b)
def nll(theta, X, y):
    logits = X @ theta[:-1] + theta[-1]
    return -jnp.sum(y * jax.nn.log_sigmoid(logits)
                    + (1 - y) * jax.nn.log_sigmoid(-logits))

opt = blr_diagonal_for_loss(learning_rate=0.01, prior_precision=1e-2)
# prior_precision=1e-2 ≡ 𝒩(0, 100I) prior ≡ L2 regularisation with λ=0.01
```

**What you gain over Adam:** At convergence, `get_posterior_diagonal(state)` gives approximate marginal variances for each parameter, enabling Wald-type confidence intervals without any additional computation.

### 5.2 State Estimation (Filtering)

**Setting:** Online estimation of a latent state x_t from sequential observations y_t with a state-space model x_{t+1} = f(x_t) + w_t, y_t = h(x_t) + v_t.

**BLR framing:** At each timestep, the "prior" is the predictive distribution from the previous step, and ρ = 1 (full update from the single observation). This recovers the (extended) Kalman filter.

```python
def kalman_step(state, y_t, H_obs, R_obs):
    """One BLR step = one Kalman update.

    state:  BLRFullRankState carrying (Λ_pred, η_pred)
    y_t:    observation
    H_obs:  observation matrix (or Jacobian for EKF)
    R_obs:  observation noise covariance
    """
    # Log-likelihood of y_t given x: ℓ(x) = −½(y−Hx)ᵀR⁻¹(y−Hx)
    # ∇_x ℓ = Hᵀ R⁻¹ (y − Hx)
    # ∇²_x ℓ = −Hᵀ R⁻¹ H  (constant, NSD)
    #
    # BLR with ρ=1:
    #   Λ_{t|t} = Λ_{t|t-1} + Hᵀ R⁻¹ H     (precision update)
    #   η_{t|t} = η_{t|t-1} + Hᵀ R⁻¹ y      (information filter form)
    #
    # This IS the Kalman information filter update.
    ...
```

**What you gain:** A unified code path for Kalman filtering and parameter learning — both are BLR steps with different priors and ρ.

### 5.3 Bilevel Optimisation

**Setting:** min_φ F(θ*(φ), φ) where θ*(φ) = argmin_θ f(θ, φ). The inner problem estimates θ; the outer problem adjusts φ.

**BLR framing:** Run BLR on the inner problem (θ given φ). The posterior q(θ|λ) is a differentiable function of φ through the BLR updates, enabling hypergradient computation.

```python
def inner_loop(phi, data, n_steps=100):
    """Inner BLR loop — differentiable w.r.t. phi."""
    theta = init_theta(phi)
    opt = blr_diagonal(learning_rate=0.01, prior_precision=1e-3)
    state = opt.init(theta)

    def step(carry, _):
        theta, state = carry
        g = jax.grad(lambda t: -inner_loss(t, phi, data))(theta)
        updates, state = opt.update(g, state)
        theta = optax.apply_updates(theta, updates)
        return (theta, state), None

    (theta_star, final_state), _ = jax.lax.scan(step, (theta, state), None, n_steps)
    return theta_star, final_state

def outer_loss(phi, data_train, data_val):
    theta_star, state = inner_loop(phi, data_train)
    return val_loss(theta_star, phi, data_val)

# Hypergradient via autodiff through BLR
dphi = jax.grad(outer_loss)(phi, data_train, data_val)
```

**What you gain:** The BLR inner loop is scan-compatible and fully differentiable. The posterior uncertainty from the inner problem can regularise the outer objective (e.g., penalise high-uncertainty solutions).

### 5.4 Neural Network Training

**Setting:** Train a deep network f_θ(x) with millions of parameters.

**BLR framing:** `blr_diagonal_for_loss` with `hessian_estimator="ggn_diag"` behaves like Adam with a Bayesian interpretation. At convergence, the diagonal precision s gives per-parameter uncertainty.

```python
import equinox as eqx

model = MyNetwork(key=jr.PRNGKey(0))
opt = blr_diagonal_for_loss(
    learning_rate=1e-3,
    prior_precision=1e-4,      # weak prior ≈ large weight variance
    hessian_estimator="ggn_diag",
)
state = opt.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def step(model, state, batch):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)
    updates, state = opt.update(grads, state)
    model = eqx.apply_updates(model, updates)
    return model, state, loss
```

**Practical considerations for large models:**

- The diagonal BLR adds exactly one extra array per parameter (the precision s), identical memory overhead to Adam's second moment.
- For structured uncertainty (e.g., K-FAC-style), use a block-diagonal variant where each block is a small full-rank BLR over one layer's parameters. This is the VOGN/IVON approach.
- The posterior variance enables cheap predictive uncertainty: sample K parameter vectors θ_k ∼ 𝒩(m, diag(1/s)), average predictions.

### 5.5 PDE Solvers (Physics-Informed Learning)

**Setting:** Solve a PDE Lu(x) = f(x) by parameterising u_θ(x) with a neural network and minimising the physics residual.

**BLR framing:** The residual loss is the (negative) log-likelihood. The posterior uncertainty on θ propagates to uncertainty on the solution u(x), which quantifies where the PDE is poorly resolved.

```python
def pde_residual_loss(theta, collocation_pts):
    """Residual for ∂²u/∂x² = f(x) on [0,1]."""
    def u(x):
        return model.apply(theta, x)

    u_xx = jax.vmap(jax.grad(jax.grad(u)))(collocation_pts)
    f_vals = source_fn(collocation_pts)
    return jnp.mean((u_xx - f_vals) ** 2)

opt = blr_diagonal_for_loss(learning_rate=1e-3, prior_precision=1e-3)

# After training:
mean, var = get_posterior_diagonal(state)

# Predictive uncertainty on u(x):
def predictive_std(x, mean_params, var_params, n_samples=50):
    keys = jr.split(jr.PRNGKey(0), n_samples)
    def sample_predict(key):
        eps = jax.tree.map(lambda m, v: m + jr.normal(key, m.shape) * jnp.sqrt(v),
                           mean_params, var_params)
        return model.apply(eps, x)
    preds = jax.vmap(sample_predict)(keys)
    return jnp.std(preds, axis=0)
```

**What you gain:** Pointwise uncertainty estimates on the PDE solution that indicate where to add collocation points (adaptive refinement), without any additional inference step.

### 5.6 Gaussian Processes

**Setting:** GP regression/classification with n data points and kernel K(x, x').

**BLR framing:** For GP classification (non-conjugate), the posterior p(f|y) is intractable. BLR-FullRank with f as the "parameter" and the GP prior 𝒩(0, K) performs natural-gradient variational inference over the function values.

```python
def gp_classification_blr(X, y, kernel_fn, n_steps=200):
    """Variational GP classification via full-rank BLR."""
    n = X.shape[0]
    K = kernel_fn(X, X) + 1e-6 * jnp.eye(n)
    K_inv = jnp.linalg.inv(K)

    # Prior: 𝒩(0, K)  →  Λ₀ = K⁻¹, m₀ = 0
    opt = blr_full_rank(
        learning_rate=0.1,
        prior_precision=0.0,    # we override with K⁻¹ below
    )

    # Manual init to set Λ₀ = K⁻¹
    from bayesian_learning_rule import BLRFullRankState
    f = jnp.zeros(n)
    state = BLRFullRankState(
        precision=K_inv,
        nat_mean=jnp.zeros(n),
        count=jnp.zeros([], jnp.int32),
    )

    def loglik_grad(f):
        """Bernoulli log-likelihood: ∇_f ℓ = y − σ(f)."""
        return y - jax.nn.sigmoid(f)

    def loglik_hessian(f):
        """∇²_f ℓ = −diag(σ(f)(1−σ(f)))."""
        p = jax.nn.sigmoid(f)
        return -jnp.diag(p * (1 - p))

    opt_with_hess = blr_full_rank(
        learning_rate=0.2,
        prior_precision=0.0,
        hessian_fn=loglik_hessian,
    )

    for i in range(n_steps):
        g = loglik_grad(f)
        updates, state = opt_with_hess.update(g, state)
        f = f + updates

    mean, cov = get_posterior_full_rank(state)
    return mean, cov  # approximate posterior 𝒩(f | mean, cov)
```

**Connection:** This is exactly the iteratively-reweighted least squares (IRLS) algorithm for GP classification when ρ = 1, and the damped natural-gradient variant of Opper & Archambeau (2009) for ρ < 1.

### 5.7 Continual Learning

**Setting:** Train on task 1, then task 2, without forgetting task 1.

**BLR framing:** After task 1, the posterior q₁(θ) becomes the prior for task 2. The precision s₁ from task 1 acts as a per-parameter "importance weight" — parameters that were precisely determined by task 1 are strongly regularised during task 2. This is exactly Elastic Weight Consolidation (EWC), but derived naturally from the BLR framework.

```python
# Task 1
opt = blr_diagonal_for_loss(learning_rate=1e-3, prior_precision=1e-4)
state = opt.init(params)
for batch in task1_data:
    grads = jax.grad(loss_fn)(params, batch)
    updates, state = opt.update(grads, state)
    params = optax.apply_updates(params, updates)

# Extract task-1 posterior → becomes task-2 prior
mean_1, var_1 = get_posterior_diagonal(state)
precision_1 = jax.tree.map(lambda v: 1.0 / v, var_1)

# Task 2: initialise new BLR with task-1 posterior as prior
# This requires per-parameter prior — a natural extension
opt2 = blr_diagonal_for_loss(
    learning_rate=1e-3,
    prior_precision=precision_1,   # per-parameter (pytree)
    prior_mean=mean_1,             # per-parameter (pytree)
)
state2 = opt2.init(params)
for batch in task2_data:
    ...
```

### 5.8 Spatio-Temporal Models and Extreme Value Statistics

**Setting:** Hierarchical Bayesian models for geospatial data, e.g., GEV parameters varying over space with Gaussian process priors on the GEV location/scale/shape.

**BLR framing:** The hierarchical structure naturally decomposes: BLR over the GEV parameters at each site (inner), with a GP-structured prior coupling sites (outer). The diagonal BLR handles the per-site parameters; the GP prior enters through the prior precision.

```python
# Schematic for a spatial GEV model
# θ_s = (μ_s, σ_s, ξ_s) at each site s
# Prior: θ_s ~ GP(m(x_s), K(x_s, x_s'))
# Likelihood: y_st ~ GEV(μ_s, σ_s, ξ_s)

def gev_nll(theta_flat, y_data, prior_precision_matrix):
    """Negative log-posterior for spatial GEV."""
    theta = unflatten(theta_flat)
    ll = sum(gev_logpdf(y_st, mu_s, sigma_s, xi_s) for ...)
    prior = -0.5 * theta_flat @ prior_precision_matrix @ theta_flat
    return -(ll + prior)

# Use full-rank BLR for correlated spatial parameters
# or diagonal BLR if sites are treated as independent
opt = blr_diagonal_for_loss(learning_rate=1e-3, prior_precision=1e-2)
```

**What you gain:** Principled propagation of spatial uncertainty through the GEV parameters, with the GP prior enforcing spatial smoothness. The BLR posterior variance on ξ (shape parameter) directly quantifies uncertainty in tail behaviour — critical for return level estimation.

### 5.9 Application Summary

```
┌────────────────────────┬────────────┬──────────────┬─────────────────────┐
│ Application            │ BLR Variant│ Key Benefit  │ Classical Equivalent│
├────────────────────────┼────────────┼──────────────┼─────────────────────┤
│ Parameter estimation   │ Diagonal   │ Free CI's    │ Adam + bootstrap    │
│ State estimation       │ Full-rank  │ Unified code │ Kalman filter       │
│ Bilevel optimisation   │ Diagonal   │ Diff'able    │ Implicit diff.      │
│ Neural networks        │ Diagonal   │ Uncertainty  │ Adam / IVON         │
│ PDEs (PINN)            │ Diagonal   │ Error maps   │ Ensembles           │
│ Gaussian processes     │ Full-rank  │ Exact IRLS   │ Laplace approx.     │
│ Continual learning     │ Diagonal   │ Natural EWC  │ EWC / SI            │
│ Extreme value / GEV    │ Diag/Full  │ Tail uncert. │ MCMC / profile lik. │
└────────────────────────┴────────────┴──────────────┴─────────────────────┘
```

---

## 6. Efax Backend Integration

### 6.1 What Efax Provides

[Efax](https://github.com/NeilGirdhar/efax) (Exponential Families for JAX) provides a typed, JAX-native implementation of exponential family distributions with both natural and expectation parameterisations. It eliminates roughly half the manual BLR implementation work.

**What efax gives the BLR for free:**

The `MultivariateDiagonalNormalNP` dataclass stores exactly the natural parameters the BLR needs: `mean_times_precision` (our η = s⊙m) and `negative_half_precision` (−s/2). The `to_exp()` method recovers (mean, second_moment), `to_variance_parametrization()` gives (mean, variance), and `sample()` provides reparameterised draws. The full-rank `MultivariateNormalNP` does the same with `SymmetricMatrixSupport` handling the upper-triangular storage.

`parameter_map(f, q, ...)` applies a function to the variable parameters of a distribution while preserving fixed parameters. The BLR's convex combination `λ_{t+1} = (1-ρ)λ_t + ρ λ̃` becomes a single `parameter_map` call.

The `NaturalParametrization` base class provides:

```
fisher_information_diagonal()     F(λ) = ∇²A(λ), diagonal elements
_fisher_information_matrix()      Full Fisher matrix via jacfwd(grad(log_normalizer))
kl_divergence(q)                  Bregman divergence in natural/expectation params
log_pdf(x)                        ηᵀT(x) − A(η) + h(x)
```

The log-normalizer has a custom JVP (`_log_normalizer_jvp`) that uses the identity ∇A(η) = μ (expectation parameters) for numerically stable differentiation.

**What you still build on top of efax:**

The Optax `GradientTransformation` wrapper (init/update protocol), the expectation-parameter gradient correction (g − Hm from Bonnet-Price), the Hessian approximation strategy (GGN diagonal, Hutchinson, etc.), and the loss-convention adapter.

### 6.2 Correspondence Table

```
┌──────────────────────────────┬────────────────────────────────────────────┐
│ BLR Concept                  │ Efax Implementation                        │
├──────────────────────────────┼────────────────────────────────────────────┤
│ q(θ) = 𝒩(m, diag(v))       │ MultivariateDiagonalNormalNP               │
│   η = s⊙m                   │   .mean_times_precision                    │
│   λ₂ = −s/2                 │   .negative_half_precision                 │
│                              │                                            │
│ q(θ) = 𝒩(m, Σ)             │ MultivariateNormalNP                       │
│   η = Λm                    │   .mean_times_precision                    │
│   −½Λ                       │   .negative_half_precision                 │
│                              │                                            │
│ Recover (m, v)               │ q.to_variance_parametrization()            │
│ Recover (m, μ₂)             │ q.to_exp()                                 │
│ Sample θ̂ ~ q                │ q.sample(key, shape)                       │
│                              │                                            │
│ KL(q ‖ p)                   │ q.kl_divergence(p_nat)                     │
│ F(λ) diagonal               │ q.fisher_information_diagonal()            │
│ F(λ) full matrix            │ q._fisher_information_matrix()             │
│ log q(x)                    │ q.log_pdf(x)                               │
│                              │                                            │
│ λ_{t+1} = (1−ρ)λ_t + ρ λ̃  │ parameter_map(lerp, q_t, q_target)        │
│ prior evidence combination   │ parameter_map(add, prior_np, lik_np)      │
│ Average sufficient stats     │ parameter_mean(ss, axis=0)                │
│ Flatten for Optax            │ Flattener.flatten(q) / .unflatten(arr)    │
└──────────────────────────────┴────────────────────────────────────────────┘
```

### 6.3 Refactored BLR State

With efax, the optimizer state carries a full distribution object:

```python
from efax import (MultivariateDiagonalNormalNP, MultivariateDiagonalNormalVP,
                  MultivariateNormalNP, parameter_map)

class BLRDiagState(NamedTuple):
    q: MultivariateDiagonalNormalNP   # full variational distribution
    count: jnp.ndarray

class BLRFullRankState(NamedTuple):
    q: MultivariateNormalNP
    count: jnp.ndarray
```

### 6.4 Refactored Update (Diagonal)

```python
def blr_diagonal_efax(
    learning_rate: float = 1e-2,
    prior: MultivariateDiagonalNormalNP = None,  # efax distribution object
    hessian_estimator: str = "ggn_diag",
    damping: float = 1e-6,
) -> optax.GradientTransformation:
    """BLR with efax backend. Expects log-likelihood gradients."""

    def init_fn(params):
        d = sum(p.size for p in jax.tree.leaves(params))
        if prior is not None:
            q0 = prior
        else:
            q0 = MultivariateDiagonalNormalNP(
                mean_times_precision=jnp.zeros(d),
                negative_half_precision=jnp.full(d, -0.5e-4),  # s₀ = 1e-4
            )
        return BLRDiagState(q=q0, count=jnp.zeros([], jnp.int32))

    def update_fn(grads, state, params=None):
        rho = learning_rate
        q_t = state.q

        # Current mean from natural params: m = -η₁ / (2 λ₂)
        vp = q_t.to_variance_parametrization()   # efax handles the algebra
        m_t = vp.mean

        # Hessian approximation
        if hessian_estimator == "ggn_diag":
            diag_H = -(grads ** 2)   # H ≈ −g², always NSD
        else:
            diag_H = jnp.zeros_like(grads)

        # Expectation-parameter gradient (with correction)
        grad_mu1 = grads - diag_H * m_t    # ∇_{μ₁} = g − Hm

        # Build the BLR target in efax's natural parameterisation:
        #   λ̃₁ = η₀ + ∇_{μ₁}
        #   λ̃₂ = λ₂₀ − ½ diag(H)    (since λ₂ = −s/2 and target s = s₀ − H)
        prior_q = prior  # or state's stored prior
        target = MultivariateDiagonalNormalNP(
            mean_times_precision=prior_q.mean_times_precision + grad_mu1,
            negative_half_precision=prior_q.negative_half_precision + 0.5 * diag_H,
        )

        # BLR update: convex combination via parameter_map
        new_q = parameter_map(
            lambda a, b: (1 - rho) * a + rho * b,
            q_t, target,
        )

        # Damping (clamp precision away from zero)
        new_q = MultivariateDiagonalNormalNP(
            mean_times_precision=new_q.mean_times_precision,
            negative_half_precision=jnp.minimum(
                new_q.negative_half_precision, -damping / 2
            ),
        )

        # Optax update = new_mean − old_mean
        new_vp = new_q.to_variance_parametrization()
        updates = new_vp.mean - m_t

        return updates, BLRDiagState(q=new_q, count=state.count + 1)

    return optax.GradientTransformation(init_fn, update_fn)
```

### 6.5 Posterior Extraction (Efax-Native)

```python
def get_posterior_efax(state: BLRDiagState):
    """Returns the full efax distribution — then use any efax method."""
    return state.q

# Usage:
q = get_posterior_efax(state)
vp = q.to_variance_parametrization()   # (mean, variance)
samples = q.sample(key, (100,))        # 100 posterior samples
fi = q.fisher_information_diagonal()   # Fisher diagonal
kl = q.kl_divergence(another_q)        # KL between posteriors
```

### 6.6 Full-Rank with Efax

The full-rank case benefits most from efax — it handles the symmetric matrix storage, Cholesky, and `parameter_map` over `SymmetricMatrixSupport`:

```python
def blr_full_rank_efax(
    learning_rate: float = 0.1,
    prior: MultivariateNormalNP = None,
    hessian_fn=None,
    damping: float = 1e-6,
) -> optax.GradientTransformation:

    def update_fn(grads, state, params=None):
        rho = learning_rate
        q_t = state.q
        m_t = q_t.mean()  # efax method: Λ⁻¹ η

        H = hessian_fn(m_t) if hessian_fn else -jnp.outer(grads, grads)
        grad_mu1 = grads - H @ m_t

        # Target natural params via efax's matrix-aware types
        target = MultivariateNormalNP(
            mean_times_precision=prior.mean_times_precision + grad_mu1,
            negative_half_precision=prior.negative_half_precision - 0.5 * H,
        )

        new_q = parameter_map(
            lambda a, b: (1 - rho) * a + rho * b,
            q_t, target,
        )
        # ... damping, extract update delta ...
```

### 6.7 What Efax Enables Beyond the Standalone Implementation

With efax as the backend, several advanced features become straightforward:

**Beyond-Gaussian variational families.** The BLR applies to any exponential family. With efax you could implement BLR for Gamma, Beta, or Dirichlet variational families by swapping the distribution type — the `parameter_map`, `kl_divergence`, and `log_normalizer` machinery all generalise.

**Automatic Fisher preconditioning.** Use `q.fisher_information_diagonal()` or `q.apply_fisher_information(vector)` for exact natural gradient steps without manually deriving F for each family.

**Conjugate prior construction.** `HasConjugatePrior.conjugate_prior_distribution()` can automatically construct the prior natural parameters for Bayesian evidence combination.

**Numerically stable log-normalizer gradients.** The custom JVP on `log_normalizer` avoids catastrophic cancellation that arises when differentiating through `log(det(...))` and `x^T Σ^{-1} x` terms separately.

### 6.8 Dependency Note

Efax depends on `tfp-nightly` (TensorFlow Probability JAX substrate) for Bessel functions used in the von Mises-Fisher distribution. As of March 2026, `tfp-nightly` is not compatible with JAX ≥ 0.9. Pin to `jax<=0.4.38` or wait for a compatible tfp release. The core Gaussian distributions used by BLR only depend on `tjax` and `array-api-compat` — a minimal fork removing the tfp dependency would resolve this.

### 6.9 GaussX as Gaussian Exponential Family Backend

[GaussX](../gaussx/README.md) provides `gaussx.expfam.GaussianExpFam` — a Gaussian-specific exponential family implementation that is purpose-built for the BLR's Gaussian specialisations (§1.5–1.7). Unlike efax (which is general-purpose but has the tfp dependency issue), GaussX is JAX/Equinox-native with no compatibility issues.

**What GaussX gives the BLR for free:**

- `GaussianExpFam(eta1, eta2)` stores the natural parameters η₁ = Λm and η₂ = −½Λ, where `eta2` is a `lineax.AbstractLinearOperator` (can be diagonal, full-rank, or low-rank)
- `to_expectation()` / `to_natural()` conversions between natural and expectation parameters
- `log_partition()` computes A(η) = −¼ η₁ᵀ η₂⁻¹ η₁ − ½ log|−2η₂| using GaussX's structure-exploiting `solve` and `logdet`
- `fisher_info()` returns the Fisher information matrix as a structured operator
- `kl_divergence(p, q)` between two Gaussian natural parameterisations
- `sufficient_stats(x)` returns [x, xxᵀ]

**Correspondence to BLR concepts:**

```
┌──────────────────────────────┬────────────────────────────────────────────┐
│ BLR Concept                  │ GaussX Implementation                      │
├──────────────────────────────┼────────────────────────────────────────────┤
│ η = Λm (natural mean)       │ GaussianExpFam.eta1                        │
│ −½Λ (natural precision)     │ GaussianExpFam.eta2 (structured operator)  │
│ Recover (m, Σ)               │ GaussianExpFam.to_expectation()            │
│ A(η) log-partition           │ GaussianExpFam.log_partition()             │
│ F(λ) Fisher information      │ GaussianExpFam.fisher_info()               │
│ KL(q ‖ p)                   │ gaussx.expfam.kl_divergence(q, p)          │
│ Λ⁻¹η (recover mean)         │ gaussx.ops.solve(Lambda_op, eta)           │
│ log|Λ| (log-normalizer)     │ gaussx.ops.logdet(Lambda_op)               │
│ Low-rank Λ = D + UUᵀ        │ gaussx.operators.LowRankUpdate(D, U)      │
│ Woodbury solve               │ gaussx.ops.solve(LowRankUpdate(...), b)   │
└──────────────────────────────┴────────────────────────────────────────────┘
```

**Key advantage for low-rank BLR (§1.7):** The low-rank precision Λ = D + UUᵀ maps directly to `gaussx.operators.LowRankUpdate(DiagonalOperator(d), U, I, U.T)`. The Woodbury solve Λ⁻¹η and matrix determinant lemma log|Λ| are then automatic — no manual implementation needed.

**Example: Full-rank BLR with GaussX**

```python
from gaussx.expfam import GaussianExpFam
from gaussx.operators import low_rank_plus_diag
import gaussx.ops as gops

# Prior in natural form
prior = GaussianExpFam(eta1=eta0, eta2=neg_half_Lambda0)

# BLR update step
def blr_update(q_nat: GaussianExpFam, g, H, rho):
    m = gops.solve(-2 * q_nat.eta2, q_nat.eta1)   # recover mean
    grad_mu1 = g - H @ m                            # expectation gradient (§1.4)
    grad_mu2 = 0.5 * H

    target_eta1 = prior.eta1 + grad_mu1
    target_eta2 = prior.eta2 + grad_mu2

    new_eta1 = (1 - rho) * q_nat.eta1 + rho * target_eta1
    new_eta2 = (1 - rho) * q_nat.eta2 + rho * target_eta2
    return GaussianExpFam(eta1=new_eta1, eta2=new_eta2)
```

**Example: Low-rank BLR with GaussX**

```python
from gaussx.operators import low_rank_plus_diag
import gaussx.ops as gops

# Low-rank precision: Λ = diag(d) + U Uᵀ
Lambda_op = low_rank_plus_diag(W=U, d=d)

# Woodbury solve and logdet — automatic via structure dispatch
m = gops.solve(Lambda_op, eta)          # O(dr² + r³) via Woodbury
log_det = gops.logdet(Lambda_op)        # O(dr² + r³) via matrix det lemma
```

**When to use GaussX vs efax:** Use GaussX when the BLR is Gaussian-only (most common case — §1.5, 1.6, 1.7). GaussX's structured operators make the low-rank and full-rank cases efficient without manual Woodbury/Cholesky code. Use efax when extending BLR to non-Gaussian exponential families (§6.7).

---

## 7. References

1. Khan, M.E. & Rue, H. (2023). "The Bayesian Learning Rule." *JMLR* 24(281):1–46.
2. Khan, M.E. et al. (2018). "Fast and Scalable Estimation of Uncertainty using Bayesian Deep Learning." *ICML*.
3. Osawa, K. et al. (2019). "Practical Deep Learning with Bayesian Principles." *NeurIPS* (VOGN).
4. Möllenhoff, T. & Khan, M.E. (2023). "SAM as an Optimal Relaxation of Bayes." *ICLR*.
5. Lin, W. et al. (2020). "Handling the Positive-Definite Constraint in the Bayesian Learning Rule." *ICML*.
6. Kunstner, F. et al. (2019). "Limitations of the Empirical Fisher Approximation." *NeurIPS*.
7. Martens, J. (2020). "New Insights and Perspectives on the Natural Gradient Method." *JMLR*.
8. Opper, M. & Archambeau, C. (2009). "The Variational Gaussian Approximation Revisited." *Neural Computation*.
9. Kirkpatrick, J. et al. (2017). "Overcoming Catastrophic Forgetting in Neural Networks." *PNAS* (EWC).
10. Shen, R. et al. (2024). "Variational Learning is Effective for Large Deep Networks." *ICML* (IVON).
11. Girdhar, N. "Efax: Exponential Families for JAX." https://github.com/NeilGirdhar/efax
