"""
Bayesian Learning Rule (Khan & Rue, 2023) — Optax Implementation
================================================================

Mathematical Foundation
─────────────────────

The Bayesian Learning Rule (BLR) casts optimisation as inference.
Instead of finding a point estimate θ*, we maintain a variational
distribution q(θ|λ) from an exponential family and minimise:

    ℒ(λ) = −𝔼_q[log p(𝒟|θ)] + KL(q(θ|λ) ‖ p(θ))

The key result: natural-gradient descent on ℒ(λ) in natural parameter
space reduces to an exponential moving average toward a "target".

─────────────────────────────────────────────────────────────────────
§1  EXPONENTIAL FAMILY BACKGROUND
─────────────────────────────────────────────────────────────────────

  q(θ|λ) = h(θ) exp(λᵀ T(θ) − A(λ))

  Natural parameters:     λ
  Sufficient statistics:  T(θ)
  Log-partition:          A(λ)
  Expectation parameters: μ = 𝔼_q[T(θ)] = ∇_λ A(λ)
  Fisher information:     F(λ) = ∇²_λ A(λ) = ∂μ/∂λ

  The map λ ↔ μ is a diffeomorphism (for minimal exponential families).
  Key identity: F⁻¹ ∇_λ f(λ) = ∇_μ f(λ(μ))   (natural ↔ expectation)

─────────────────────────────────────────────────────────────────────
§2  THE BAYESIAN LEARNING RULE
─────────────────────────────────────────────────────────────────────

  Starting from natural gradient ascent on the ELBO:

    λ_{t+1} = λ_t + ρ_t F(λ_t)⁻¹ ∇_λ ELBO(λ_t)

  Expanding the ELBO gradient for exponential families:

    ∇_λ ELBO = ∇_λ 𝔼_q[ℓ(θ)] − F(λ_t − λ₀)

  and applying F⁻¹, we obtain:

  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  λ_{t+1} = (1 − ρ) λ_t + ρ [λ₀ + ∇_μ 𝔼_q[ℓ(θ)]]       │
  │                                                             │
  │  where  ℓ(θ) = log p(𝒟|θ)                                 │
  │         λ₀   = natural parameters of the prior p(θ)        │
  │         ρ     = learning rate                               │
  │         ∇_μ   = gradient w.r.t. expectation parameters      │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘

  This is a convex combination interpolating between the current
  λ_t and a "target" λ̃ = λ₀ + ∇_μ 𝔼_q[ℓ(θ)].

─────────────────────────────────────────────────────────────────────
§3  GAUSSIAN EXPECTATION-PARAMETER GRADIENTS
─────────────────────────────────────────────────────────────────────

  For q(θ) = 𝒩(m, Σ), the expectation parameters are:

    μ₁ = m          μ₂ = m mᵀ + Σ

  Since Σ = μ₂ − μ₁ μ₁ᵀ, changing μ₁ while holding μ₂ fixed
  also changes Σ.  The chain rule gives:

    ∇_{μ₁} 𝔼_q[ℓ] = ∂𝔼[ℓ]/∂m + (∂𝔼[ℓ]/∂Σ)(∂Σ/∂μ₁)

  Using the Bonnet–Price identities (∂𝔼[ℓ]/∂m = 𝔼[∇_θ ℓ],
  ∂𝔼[ℓ]/∂Σ = ½ 𝔼[∇²_θ ℓ]) and ∂Σ/∂μ₁ = −2m:

  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │  ∇_{μ₁} 𝔼_q[ℓ] = g − H m     (g = 𝔼[∇_θ ℓ])         │
  │  ∇_{μ₂} 𝔼_q[ℓ] = ½ H         (H = 𝔼[∇²_θ ℓ])        │
  │                                                          │
  │  ⚠ The −Hm term in ∇_{μ₁} is essential!  Omitting it   │
  │    (i.e. using just g) yields wrong fixed points.        │
  │                                                          │
  └──────────────────────────────────────────────────────────┘

  NOTE: g and H can be approximated by single-sample MC estimates
  using the reparameterisation trick θ̂ = m + L ε,  ε ∼ 𝒩(0, I),
  where Σ = L Lᵀ.  In the zero-variance limit (θ̂ = m), this
  reduces to standard gradient/Hessian evaluations at the mean.

─────────────────────────────────────────────────────────────────────
§4  DIAGONAL GAUSSIAN SPECIALISATION
─────────────────────────────────────────────────────────────────────

  q(θ) = 𝒩(m, diag(v))    where v = (v₁, …, v_d), vᵢ > 0

  Natural parameters:
    λ₁ = m / v   =  s ⊙ m     (s = 1/v = diagonal precision)
    λ₂ = −1/(2v) = −s/2

  BLR updates with ∇_{μ₁} = g − diag(H) ⊙ m  and  ∇_{μ₂} = ½ diag(H):

    ┌──────────────────────────────────────────────────────────┐
    │  Precision update (from λ₂):                             │
    │                                                          │
    │    s_{t+1} = (1−ρ) s_t  +  ρ [s₀ − diag(H_t)]         │
    │                                                          │
    │  Natural mean update (from λ₁):                          │
    │                                                          │
    │    η_{t+1} = (1−ρ) η_t                                  │
    │             + ρ [η₀ + g_t − diag(H_t) ⊙ m_t]           │
    │                                                          │
    │  Recover mean:                                           │
    │    m_{t+1} = η_{t+1} / s_{t+1}                          │
    └──────────────────────────────────────────────────────────┘

  where  g_t = ∇_θ ℓ(θ̂_t)           — stochastic gradient at sample
         H_t = diag(∇²_θ ℓ(θ̂_t))    — diagonal Hessian
         η₀  = s₀ ⊙ m₀              — prior natural mean
         s₀  = 1/v₀                  — prior precision

─────────────────────────────────────────────────────────────────────
§5  FULL-RANK GAUSSIAN SPECIALISATION
─────────────────────────────────────────────────────────────────────

  q(θ) = 𝒩(m, Σ)     (d × d full covariance)

  Natural parameters:  η = Λ m,   Λ = Σ⁻¹

  BLR updates:

    Λ_{t+1} = (1−ρ) Λ_t        + ρ [Λ₀ − H_t]
    η_{t+1} = (1−ρ) η_t        + ρ [η₀ + g_t − H_t m_t]
    m_{t+1} = solve(Λ_{t+1}, η_{t+1})

─────────────────────────────────────────────────────────────────────
§6  CONNECTIONS TO KNOWN ALGORITHMS
─────────────────────────────────────────────────────────────────────

  ● SGD:  Fix v (don't update λ₂), flat prior (s₀ → 0), ρ = 1.
          Then m_{t+1} = m_t + v ⊙ g_t.

  ● Adam (approx):  Diagonal Gaussian BLR with EMA smoothing.
          Adam's second moment v̂_t ≈ −diag(H_t) when H ≈ −g².

  ● Newton:  ρ = 1, flat prior, full exact Hessian for H.

  ● Natural gradient VI:  Full-rank Gaussian BLR with exact Fisher.

  ● Kalman filter:  Sequential BLR, one data point per step, ρ = 1.

─────────────────────────────────────────────────────────────────────
§7  PSEUDOCODE
─────────────────────────────────────────────────────────────────────

  ─── Algorithm: BLR-Diagonal ───────────────────────────────────
  Input:  ℓ(θ), m₀, v₀, ρ, T

  Initialise:  s ← 1/v₀,  η ← s ⊙ m₀,  m ← m₀

  for t = 1, …, T:
      ε  ∼ 𝒩(0, I)
      θ̂  ← m + ε / √s                    # reparameterised sample
      g  ← ∇_θ ℓ(θ̂)
      h  ← diag(∇²_θ ℓ(θ̂))               # diagonal Hessian (≤ 0)

      s   ← (1−ρ) s   + ρ (1/v₀ − h)
      η   ← (1−ρ) η   + ρ (m₀/v₀ + g − h ⊙ m)
      m   ← η / s

  return m, 1/s
  ──────────────────────────────────────────────────────────────

  ─── Algorithm: BLR-FullRank ──────────────────────────────────
  Input:  ℓ(θ), m₀, Λ₀, ρ, T

  Initialise:  Λ ← Λ₀,  η ← Λ₀ m₀,  m ← m₀

  for t = 1, …, T:
      θ̂  ← sample from 𝒩(m, Λ⁻¹)
      g  ← ∇_θ ℓ(θ̂)
      H  ← ∇²_θ ℓ(θ̂)

      Λ   ← (1−ρ) Λ   + ρ (Λ₀ − H)
      η   ← (1−ρ) η   + ρ (Λ₀ m₀ + g − H m)
      m   ← solve(Λ, η)

  return m, Λ⁻¹
  ──────────────────────────────────────────────────────────────
"""

# ═══════════════════════════════════════════════════════════════
# IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════

from __future__ import annotations
from typing import NamedTuple, Callable, Optional

import jax
import jax.numpy as jnp
import optax


# ───────────────────────────────────────────────────────────────
# State containers
# ───────────────────────────────────────────────────────────────

class BLRDiagState(NamedTuple):
    """Natural-parameter state for q(θ) = 𝒩(m, diag(1/s))."""
    precision: optax.Params     # s = 1/v
    nat_mean: optax.Params      # η = s ⊙ m
    count: jnp.ndarray


class BLRFullRankState(NamedTuple):
    """Natural-parameter state for q(θ) = 𝒩(m, Λ⁻¹), θ ∈ ℝᵈ."""
    precision: jnp.ndarray      # Λ (d, d)
    nat_mean: jnp.ndarray       # η = Λ m (d,)
    count: jnp.ndarray


# ───────────────────────────────────────────────────────────────
# Diagonal Gaussian BLR
# ───────────────────────────────────────────────────────────────

def blr_diagonal(
    learning_rate: float = 1e-2,
    prior_precision: float = 1e-4,
    prior_mean: float = 0.0,
    hessian_estimator: str = "ggn_diag",
    damping: float = 1e-6,
) -> optax.GradientTransformation:
    """Diagonal-Gaussian BLR.  Expects ∇_θ log p(𝒟|θ)."""

    def init_fn(params: optax.Params) -> BLRDiagState:
        precision = jax.tree.map(
            lambda p: jnp.full_like(p, prior_precision), params
        )
        nat_mean = jax.tree.map(
            lambda p: jnp.full_like(p, prior_precision * prior_mean), params
        )
        return BLRDiagState(
            precision=precision,
            nat_mean=nat_mean,
            count=jnp.zeros([], jnp.int32),
        )

    def update_fn(
        grads: optax.Updates,
        state: BLRDiagState,
        params: Optional[optax.Params] = None,
    ) -> tuple[optax.Updates, BLRDiagState]:
        rho = learning_rate
        s0 = prior_precision
        eta0 = prior_precision * prior_mean

        # Current mean: m_t = η_t / s_t
        m_t = jax.tree.map(lambda eta, s: eta / s, state.nat_mean, state.precision)

        # ── Diagonal Hessian estimate (H ≤ 0 for concave ℓ) ──
        if hessian_estimator == "ggn_diag":
            diag_H = jax.tree.map(lambda g: -(g ** 2), grads)  # H ≈ −g²
        elif hessian_estimator == "identity":
            diag_H = jax.tree.map(jnp.zeros_like, grads)
        else:
            raise ValueError(f"Unknown hessian_estimator: {hessian_estimator}")

        # ── Expectation-parameter gradient (with correction!) ──
        # ∇_{μ₁} 𝔼_q[ℓ] = g − diag(H) ⊙ m
        grad_mu1 = jax.tree.map(lambda g, h, m: g - h * m, grads, diag_H, m_t)

        # ── Natural parameter updates ──
        neg_diag_H = jax.tree.map(lambda h: -h, diag_H)
        new_precision = jax.tree.map(
            lambda s, nh: jnp.maximum((1 - rho) * s + rho * (s0 + nh), damping),
            state.precision, neg_diag_H,
        )
        new_nat_mean = jax.tree.map(
            lambda eta, gm1: (1 - rho) * eta + rho * (eta0 + gm1),
            state.nat_mean, grad_mu1,
        )

        # ── Optax update: Δθ = m_{t+1} − m_t ──
        new_mean = jax.tree.map(lambda eta, s: eta / s, new_nat_mean, new_precision)
        updates = jax.tree.map(lambda mn, mo: mn - mo, new_mean, m_t)

        return updates, BLRDiagState(new_precision, new_nat_mean, state.count + 1)

    return optax.GradientTransformation(init_fn, update_fn)


def blr_diagonal_for_loss(
    learning_rate: float = 1e-2,
    prior_precision: float = 1e-4,
    prior_mean: float = 0.0,
    hessian_estimator: str = "ggn_diag",
    damping: float = 1e-6,
) -> optax.GradientTransformation:
    """Diagonal BLR accepting ∇_θ Loss (standard Optax convention).

    Drop-in replacement for ``optax.adam`` / ``optax.sgd``.
    Internally negates: ∇_θ ℓ = −∇_θ Loss.
    """
    inner = blr_diagonal(
        learning_rate=learning_rate,
        prior_precision=prior_precision,
        prior_mean=prior_mean,
        hessian_estimator=hessian_estimator,
        damping=damping,
    )

    def init_fn(params):
        return inner.init(params)

    def update_fn(grads, state, params=None):
        return inner.update(jax.tree.map(jnp.negative, grads), state, params)

    return optax.GradientTransformation(init_fn, update_fn)


# ───────────────────────────────────────────────────────────────
# Full-rank Gaussian BLR
# ───────────────────────────────────────────────────────────────

def blr_full_rank(
    learning_rate: float = 1e-2,
    prior_precision: float = 1e-4,
    prior_mean: Optional[jnp.ndarray] = None,
    damping: float = 1e-6,
    hessian_fn: Optional[Callable] = None,
) -> optax.GradientTransformation:
    """Full-rank Gaussian BLR for θ ∈ ℝᵈ.  Expects ∇_θ log p(𝒟|θ)."""

    def init_fn(params: jnp.ndarray) -> BLRFullRankState:
        d = params.shape[0]
        Lambda_0 = prior_precision * jnp.eye(d)
        m0 = jnp.zeros(d) if prior_mean is None else prior_mean
        return BLRFullRankState(
            precision=Lambda_0,
            nat_mean=Lambda_0 @ m0,
            count=jnp.zeros([], jnp.int32),
        )

    def update_fn(
        grads: jnp.ndarray,
        state: BLRFullRankState,
        params: Optional[jnp.ndarray] = None,
    ) -> tuple[jnp.ndarray, BLRFullRankState]:
        rho = learning_rate
        d = grads.shape[0]
        Lambda_0 = prior_precision * jnp.eye(d)
        m0 = jnp.zeros(d) if prior_mean is None else prior_mean
        eta_0 = Lambda_0 @ m0

        # Current mean: m_t = Λ_t⁻¹ η_t
        m_t = jnp.linalg.solve(state.precision, state.nat_mean)

        # Hessian
        H = hessian_fn(m_t) if hessian_fn is not None else -jnp.outer(grads, grads)

        # ∇_{μ₁} 𝔼_q[ℓ] = g − H m  (the critical correction)
        grad_mu1 = grads - H @ m_t

        # Natural parameter updates
        new_Lambda = (1 - rho) * state.precision + rho * (Lambda_0 - H)
        new_Lambda = new_Lambda + damping * jnp.eye(d)
        new_eta = (1 - rho) * state.nat_mean + rho * (eta_0 + grad_mu1)

        new_mean = jnp.linalg.solve(new_Lambda, new_eta)
        update = new_mean - m_t

        return update, BLRFullRankState(new_Lambda, new_eta, state.count + 1)

    return optax.GradientTransformation(init_fn, update_fn)


# ───────────────────────────────────────────────────────────────
# Posterior extraction
# ───────────────────────────────────────────────────────────────

def get_posterior_diagonal(state: BLRDiagState):
    """Returns (mean, variance) pytrees."""
    mean = jax.tree.map(lambda eta, s: eta / s, state.nat_mean, state.precision)
    variance = jax.tree.map(lambda s: 1.0 / s, state.precision)
    return mean, variance


def get_posterior_full_rank(state: BLRFullRankState):
    """Returns (mean [d], covariance [d,d])."""
    mean = jnp.linalg.solve(state.precision, state.nat_mean)
    covariance = jnp.linalg.inv(state.precision)
    return mean, covariance


# ───────────────────────────────────────────────────────────────
# Schedule composability
# ───────────────────────────────────────────────────────────────

def blr_with_schedule(
    schedule_fn: optax.Schedule,
    prior_precision: float = 1e-4,
    damping: float = 1e-6,
) -> optax.GradientTransformation:
    """BLR-diagonal with an Optax learning rate schedule."""
    return optax.inject_hyperparams(
        lambda learning_rate: blr_diagonal_for_loss(
            learning_rate=learning_rate,
            prior_precision=prior_precision,
            damping=damping,
        )
    )(learning_rate=schedule_fn)


# ═══════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Toy: minimise f(θ) = ½‖θ − θ*‖² ──
    # ℓ(θ) = −f(θ),   ∇_θ ℓ = θ* − θ,   ∇²_θ ℓ = −I
    # Prior: 𝒩(0, (1/s₀)I),  s₀ = 1e-4
    # Analytic posterior: m* = θ*/(1+s₀) ≈ 2.9997,  v* ≈ 1/(1+s₀) ≈ 0.9999

    d = 5
    theta_star = jnp.ones(d) * 3.0

    def loss_fn(theta):
        return 0.5 * jnp.sum((theta - theta_star) ** 2)

    loss_grad = jax.grad(loss_fn)
    s0 = 1e-4
    analytic_m = theta_star / (1 + s0)
    analytic_v = 1.0 / (1.0 + s0)
    print(f"Analytic posterior:  m*[0]={analytic_m[0]:.6f}  v*={analytic_v:.6f}\n")

    # ── 1. Diagonal BLR (GGN, Adam-like) via for_loss wrapper ──
    print("─── Diagonal BLR (GGN/Adam-like) ───")
    opt1 = blr_diagonal_for_loss(learning_rate=0.05, prior_precision=s0)
    theta = jnp.zeros(d)
    st = opt1.init(theta)
    for i in range(500):
        u, st = opt1.update(loss_grad(theta), st)
        theta = optax.apply_updates(theta, u)
        if (i + 1) % 100 == 0:
            m, v = get_posterior_diagonal(st)
            print(f"  step {i+1:4d}  m[0]={m[0]:.4f}  v[0]={v[0]:.6f}  loss={loss_fn(theta):.6f}")

    # ── 2. Full-rank BLR (exact Hessian, log-lik gradients) ──
    print("\n─── Full-rank BLR (exact Hessian) ───")
    hess_fn = lambda _: -jnp.eye(d)
    opt2 = blr_full_rank(learning_rate=0.1, prior_precision=s0, hessian_fn=hess_fn)
    theta = jnp.zeros(d)
    st = opt2.init(theta)
    for i in range(200):
        g = -loss_grad(theta)  # log-lik gradient
        u, st = opt2.update(g, st)
        theta = theta + u
        if (i + 1) % 50 == 0:
            m, C = get_posterior_full_rank(st)
            print(f"  step {i+1:4d}  m[0]={m[0]:.4f}  v[0]={C[0,0]:.6f}  loss={loss_fn(theta):.6f}")

    print(f"\n  Final θ:       {theta}")
    print(f"  Analytic m*:   {analytic_m}")
    print(f"  Analytic v*:   {analytic_v:.6f}")
