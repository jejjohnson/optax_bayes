# Mathematical Background

This page derives the updates optax_bayes implements, states the numerical
safeguards they rely on, and lists the classical algorithms they reduce to.
Each formula matches the code; the pseudocode blocks follow the `update`
functions step by step. The main reference is Khan & Rue (2023).

## 1. The Bayesian Learning Rule

### 1.1 Variational objective

Instead of a point estimate $\theta^\star$, keep a distribution
$q(\theta \mid \lambda)$ from an exponential family and minimise

$$
\mathcal{L}(\lambda) = -\mathbb{E}_{q}\big[\ell(\theta)\big]
  + \mathrm{KL}\big(q(\theta \mid \lambda)\,\|\,p(\theta)\big),
\qquad \ell(\theta) = \log p(\mathcal{D} \mid \theta),
$$

with $p(\theta)$ the prior and $\ell$ the log-likelihood.

### 1.2 Exponential families

$q(\theta \mid \lambda) = h(\theta)\exp\big(\lambda^\top T(\theta) - A(\lambda)\big)$
has natural parameters $\lambda$, sufficient statistics $T$, log-partition $A$,
expectation parameters $\mu = \mathbb{E}_q[T(\theta)] = \nabla_\lambda A$ and
Fisher information $F(\lambda) = \nabla^2_\lambda A = \partial\mu / \partial\lambda$.
For a minimal family, the natural gradient is the gradient with respect to the
expectation parameters:

$$
F(\lambda)^{-1}\, \nabla_\lambda f = \nabla_\mu f.
$$

### 1.3 The rule

Natural-gradient descent on $\mathcal{L}$, with a prior in the same family
(natural parameters $\lambda_0$), gives the BLR:

$$
\boxed{\;\lambda_{t+1} = (1 - \rho)\,\lambda_t
  + \rho\,\big(\lambda_0 + \nabla_\mu \mathbb{E}_{q_t}[\ell(\theta)]\big)\;}
$$

Each step is a convex combination of the current natural parameters and a
target, with step size $\rho \in (0, 1]$.

### 1.4 Gaussian expectation gradients

For $q = \mathcal{N}(m, \Sigma)$, $\mu_1 = m$ and
$\mu_2 = mm^\top + \Sigma$. The Bonnet and Price identities
$\nabla_m \mathbb{E}_q[\ell] = \mathbb{E}_q[\nabla_\theta \ell]$ and
$\nabla_\Sigma \mathbb{E}_q[\ell] = \tfrac12\mathbb{E}_q[\nabla^2_\theta \ell]$
(Opper & Archambeau, 2009), together with $\Sigma = \mu_2 - \mu_1\mu_1^\top$,
give

$$
\nabla_{\mu_1}\mathbb{E}_q[\ell] = g - Hm, \qquad
\nabla_{\mu_2}\mathbb{E}_q[\ell] = \tfrac12 H, \qquad
g = \mathbb{E}_q[\nabla_\theta \ell], \quad H = \mathbb{E}_q[\nabla^2_\theta \ell].
$$

The $-Hm$ term comes from $\partial\Sigma / \partial\mu_1 = -2m$. Leaving it
out gives the wrong fixed point.

With $\Lambda = \Sigma^{-1}$ and natural parameters $\eta = \Lambda m$ and
$-\tfrac12\Lambda$, the rule becomes the update every Gaussian variant builds
on:

$$
\begin{aligned}
\Lambda_{t+1} &= (1 - \rho)\,\Lambda_t + \rho\,(\Lambda_0 - H_t) \\
\eta_{t+1}    &= (1 - \rho)\,\eta_t + \rho\,(\eta_0 + g_t - H_t m_t) \\
m_{t+1}       &= \Lambda_{t+1}^{-1}\,\eta_{t+1}
\end{aligned}
$$

### 1.5 Mean form

Substituting $\eta_t = \Lambda_t m_t$ and the precision update into the
natural-mean update gives
$\eta_{t+1} = \Lambda_{t+1} m_t + \rho\big(g_t - \Lambda_0(m_t - m_0)\big)$, so

$$
m_{t+1} = m_t + \rho\,\Lambda_{t+1}^{-1}\big(g_t - \Lambda_0 (m_t - m_0)\big):
$$

a step along the gradient of the log-joint, preconditioned by the new
precision. The two forms agree exactly when $\Lambda_{t+1}$ is the precision
the rule prescribes. They differ when the stored precision is an
approximation of it (the low-rank variant, §2.3): there the mean form stays a
well-defined preconditioned step, while the natural form multiplies the
approximation error by $\Lambda_{t+1}^{-1}$.

## 2. The families

| Family | Precision | Storage | Cost per step | Transform |
|---|---|---|---|---|
| Diagonal | $\operatorname{diag}(s)$ | $O(d)$ | $O(d)$ | `blr_diagonal` |
| Full-rank | dense $\Lambda$ | $O(d^2)$ | $O(d^3)$ | `blr_full_rank` |
| Low-rank | $\operatorname{diag}(D) + UU^\top$ | $O(dr)$ | $O(dr^2 + r^3)$ | `blr_low_rank` |

In every transform the state starts at the params passed to `init`
($\eta = s_0 \odot \theta_{\text{init}}$, so $m = \theta_{\text{init}}$), and the
prior $(m_0, s_0)$ enters every step through $\eta_0$ and $\Lambda_0$. Each
`update` returns $m_{t+1} - m_t$, so after `optax.apply_updates` the params
equal the variational mean.

### 2.1 Diagonal

With $q = \mathcal{N}(m, \operatorname{diag}(1/s))$, $\eta = s \odot m$ and a
diagonal Hessian estimate $h$:

```text
blr_diagonal: update(g, state)            # g = ∇θ ℓ at the current mean
    m   ← η / s
    h   ← -g²            ("ggn_diag")  or  0  ("identity")
    s'  ← max((1-ρ) s + ρ (s₀ - h), ε)     # ε = damping
    η'  ← (1-ρ) η + ρ (s₀ m₀ + g - h ⊙ m)
    m'  ← η' / s'
    return m' - m,  (s', η')
```

$s_0$ and $m_0$ are scalars broadcast over the parameter pytree.

### 2.2 Full rank

```text
blr_full_rank: update(g, state)
    m   ← solve(Λ, η)                      # Cholesky: Λ is tagged PSD
    H   ← -g gᵀ  ("ggn"),  0  ("identity"),  or  hessian_estimator(m, g)
    Λ'  ← (1-ρ) Λ + ρ (Λ₀ - H) + ε I
    η'  ← (1-ρ) η + ρ (Λ₀ m₀ + g - H m)
    m'  ← solve(Λ', η')
    return m' - m,  (Λ', η')
```

$\Lambda_0 = s_0 I$. With the exact Hessian of a conjugate model (Bayesian
linear regression) the fixed point is the exact posterior.

### 2.3 Low rank (SLANG)

The low-rank family follows SLANG (Mishkin et al., 2018). The curvature comes
as a factor, $-H_t = G_t G_t^\top$: $G_t = g_t$ for `"ggn"`, an empty factor for
`"identity"`, and the positive eigen-part of $-H$ for a callable. The precision
the rule prescribes is then

$$
(1-\rho)\Lambda_t + \rho(\Lambda_0 - H_t)
  = \operatorname{diag}\big((1-\rho) D_t + \rho D_0\big) + \tilde U \tilde U^\top,
\qquad \tilde U = \big[\sqrt{1-\rho}\, U_t,\ \sqrt{\rho}\, G_t\big].
$$

A thin SVD keeps the top $r$ directions of $\tilde U$, and the diagonal of the
discarded part moves into $D$:

$$
\begin{aligned}
U_{t+1} &= \operatorname{top}_r(\tilde U) \\
D_{t+1} &= (1-\rho) D_t + \rho D_0
   + \operatorname{diag}\big(\tilde U \tilde U^\top - U_{t+1} U_{t+1}^\top\big) + \epsilon
\end{aligned}
$$

So $\operatorname{diag}(\Lambda_{t+1})$ is exactly what the rule prescribes,
and with $r \ge d$ nothing is discarded and the whole precision is exact. The
mean takes the mean-form step of §1.5, with $\Lambda_0 = D_0 = s_0 I$.

```text
blr_low_rank: update(g, state)
    m   ← solve(diag(D) + U Uᵀ, η)                     # Woodbury, O(d r²)
    G   ← g[:, None]  ("ggn"),  (d, 0)  ("identity"),  or  psd_factor(-H(m, g))
    Ũ   ← [√(1-ρ) U,  √ρ G]                            # (d, r + k)
    P, σ ← thin_svd(Ũ)
    U'  ← P[:, :r] σ[:r]
    D'  ← (1-ρ) D + ρ s₀ + Σ_{j>r} (P[:, j] σ_j)² + ε
    Δ   ← solve(diag(D') + U' U'ᵀ, ρ (g - s₀ (m - m₀)))
    η'  ← (diag(D') + U' U'ᵀ)(m + Δ)
    return Δ,  (D', U', η')
```

A step costs $O(d(r+k)^2)$ for a rank-$k$ factor: $O(dr^2)$ for the built-in
estimators. A callable estimator returns a dense $(d, d)$ Hessian, and
factoring it costs $O(d^3)$.

### 2.4 IVON

`ivon` is an IVON-style diagonal optimizer (Shen et al., 2024) on loss
gradients, with momentum $\bar g$, a diagonal curvature estimate $h$, and
weight decay $\lambda$ acting as the prior precision:

```text
ivon: update(g, state, params θ)                  # g = ∇θ loss
    ḡ  ← β₁ ḡ + (1-β₁) g
    h  ← β₂ h + (1-β₂) · ess · g²
    Δ  ← -α · clip((ḡ / (1-β₁ᵗ) + λ θ) / (h + λ), ±clip_radius)
    return Δ,  (ḡ, h)
```

The posterior is $\mathcal{N}(\theta, \operatorname{diag}(1/(h + \lambda)))$,
and `sample_ivon` draws from it before each gradient for Monte Carlo training.
This implementation uses the squared gradient as its curvature estimate. The
paper's IVON uses a reparameterisation-trick Hessian estimate and adds a
correction term to the $h$ update that keeps $h$ positive.

### 2.5 Newton's method

`newton` is the full-rank rule with $\rho = 1$, $\Lambda_0 = \delta I$, no
per-step damping and the exact Hessian:

$$
m_{t+1} = (\delta I - H_t)^{-1}(g_t - H_t m_t)
  \;\xrightarrow{\;\delta \to 0\;}\; m_t - H_t^{-1} g_t .
$$

Away from a maximum $H_t$ can be indefinite, so `newton` solves with LU rather
than Cholesky.

## 3. Numerical requirements

### 3.1 Keeping the precision positive

The precision must stay positive (definite). The rule guarantees it when
$H_t \preceq 0$; with an indefinite estimate it can fail.

| Family | Safeguard |
|---|---|
| Diagonal | $s \leftarrow \max(s, \epsilon)$ |
| Full rank | $\Lambda \leftarrow \Lambda + \epsilon I$ every step |
| Low rank | $D \leftarrow D + \epsilon$ every step; a callable's positive curvature is dropped when it is factored |

Additive damping accumulates: at a fixed point the extra precision is about
$\epsilon / \rho$. Lin et al. (2020) give an update that stays positive
definite without damping.

### 3.2 Hessian estimates

| Estimate | Where | Properties |
|---|---|---|
| Exact $\nabla^2_\theta \ell$ | callable (full, low rank) | $O(d^2)$ memory. Not NSD for non-concave $\ell$. Gives the exact posterior of conjugate models. |
| Squared gradient, $h = -g^2$ / $H = -gg^\top$ | `"ggn_diag"`, `"ggn"` | Always NSD and cheap. Uses the gradient it is given, so with a mini-batch gradient it is an empirical-Fisher-type estimate (Kunstner et al., 2019). It vanishes at stationary points, where the precision decays to the prior. |
| Gauss–Newton $-J^\top \nabla^2 J$ | callable | NSD. Equals the Fisher for exponential-family likelihoods with canonical links (Martens, 2020). Needs per-example Jacobians. |
| Zero, $H = 0$ | `"identity"` | Precision stays at the prior; the mean converges to the MAP estimate. |

The rank-1 `"ggn"` estimate can be unstable under a near-flat prior: the
precision is then nearly singular in every direction the latest gradients do
not span. Use an informative prior or a better curvature estimate.

### 3.3 Monte Carlo and the delta method

The transforms take whatever gradient you pass. Evaluating it at the current
mean replaces $\mathbb{E}_q[\cdot]$ by its value at $m$ (the delta method),
which is the usual optimizer mode. Then the fixed point with the exact Hessian
is the Laplace approximation, $m = \hat\theta_{\text{MAP}}$ and
$\Lambda = \Lambda_0 - \nabla^2\ell(\hat\theta)$. For Monte Carlo VI, draw
$\hat\theta$ with `sample_posterior_*` (or `sample_ivon`), and evaluate the
gradient and curvature there.

### 3.4 Floating point

| Issue | Handling |
|---|---|
| Float32 params under `jax_enable_x64` | The state, updates, posteriors and samples keep the params' dtype; inputs are cast to it |
| Float32 training | Supported. CI runs the suite with x64 off; tests that need float64 are marked `x64_only` |
| Ill-conditioned precision | Damping, an informative prior, or `solver=gaussx.CGSolver(...)` with a preconditioner |
| Cancellation in $m = \eta / s$ | The state holds $\eta$ and $s$; $m$ is formed only when needed |

### 3.5 JAX

The `update` functions are pure, with no Python control flow on traced values,
so they `jit`, and the states are `NamedTuple` pytrees, so they can be
`lax.scan` carries. The diagonal and IVON states mirror the parameter tree. The
full-rank and low-rank states need a flat vector; use
`jax.flatten_util.ravel_pytree`.

## 4. Model zoo

Each row is the BLR with one choice of family, curvature, step size and prior.
Khan & Rue (2023, Table 1) give more.

| Algorithm | Family | Curvature $H$ | $\rho$ | Prior | In optax_bayes |
|---|---|---|---|---|---|
| Gradient descent with weight decay | diagonal, fixed $s = s_0$ | $0$ | $\rho$ (step $\rho / s_0$) | $\mathcal{N}(m_0, s_0^{-1} I)$ | `blr_diagonal(hessian_estimator="identity")` |
| RMSprop-like scaling (no square root) | diagonal | $-g^2$ | $\approx 1 - \beta_2$ | weak | `blr_diagonal` |
| Newton's method | full rank | exact | $1$ | flat ($\delta \to 0$) | `newton` |
| Laplace approximation | full rank | exact, at the mean | fixed point | given | `blr_full_rank`, exact callable |
| Conjugate update / Kalman measurement step | full rank | exact | $1$ | predictive | `blr_full_rank(learning_rate=1.0)` with an exact callable; a non-isotropic predictive prior needs matrix priors (open question) |
| Online and continual learning | any | any | $\rho$ | previous posterior | read the posterior, rebuild with it as the prior |
| Vadam, VOGN | diagonal | $-g^2$ or Gauss–Newton, at MC samples | $\rho$ | $\mathcal{N}(0, s_0^{-1} I)$ | `blr_diagonal` with `sample_posterior_diagonal` |
| IVON | diagonal | $-g^2$ (this implementation) | $1 - \beta_2$ | weight decay | `ivon` |
| SLANG | low rank | Gauss–Newton factor | $\rho$ | $\mathcal{N}(0, s_0^{-1} I)$ | `blr_low_rank` |

Two worked cases:

- **Gradient descent.** With $H = 0$ the precision stays at $s_0$, and §1.5
  gives $m_{t+1} = m_t + (\rho / s_0)\big(g_t - s_0(m_t - m_0)\big)$: gradient
  ascent on the log-likelihood with step $\rho / s_0$ and weight decay towards
  $m_0$.
- **Kalman measurement update.** For a linear-Gaussian observation
  $y = A\theta + \varepsilon$ with noise precision $R^{-1}$, the exact Hessian
  is $-A^\top R^{-1} A$. One step with $\rho = 1$ from the predictive prior
  $(m_0, \Lambda_0)$ gives $\Lambda_1 = \Lambda_0 + A^\top R^{-1} A$ and
  $m_1 = \Lambda_1^{-1}(\Lambda_0 m_0 + A^\top R^{-1} y)$: the
  information-form Kalman update.

## 5. Mapping to the API

| Symbol | Argument |
|---|---|
| $\rho$ | `learning_rate` |
| $s_0$, $\Lambda_0 = s_0 I$, $D_0 = s_0 I$ | `prior_precision` |
| $m_0$ | `prior_mean` |
| $H_t$ or its factor | `hessian_estimator` |
| $\epsilon$ | `damping` |
| $r$ | `rank` |
| $g_t$ | the gradient passed to `update` (log-likelihood for `blr_*`, loss for `*_for_loss`, `newton_for_loss` and `ivon`) |

## 6. Example applications

- **MAP and regularised estimation.** `hessian_estimator="identity"` turns
  the prior into L2 regularisation: `prior_precision=1e-2` is
  $\mathcal{N}(0, 100 I)$, or weight decay $0.01$.
- **Bayesian neural networks.** `blr_diagonal_for_loss` or `ivon`, sampling
  weights before each gradient.
- **Sequential estimation.** Full rank with $\rho = 1$ and an exact Hessian is
  the Kalman measurement update (§4). Chaining steps, with each posterior as
  the next prior, needs matrix-valued priors, which the transforms do not take
  yet.
- **Continual learning.** The posterior after task $k$ is the prior for task
  $k+1$: the BLR form of EWC (Kirkpatrick et al., 2017). Today the prior is a
  scalar, so this means rebuilding the transform with the old posterior's
  scale (see [Scope](boundaries.md#open-questions)).

## References

- Khan, M. E. & Rue, H. (2023). [The Bayesian Learning Rule](https://arxiv.org/abs/2107.04562). *JMLR* 24(281):1–46.
- Khan, M. E., Nielsen, D., Tangkaratt, V., Lin, W., Gal, Y. & Srivastava, A. (2018). [Fast and Scalable Bayesian Deep Learning by Weight-Perturbation in Adam](https://arxiv.org/abs/1806.04854). *ICML*.
- Mishkin, A., Kunstner, F., Nielsen, D., Schmidt, M. & Khan, M. E. (2018). [SLANG: Fast Structured Covariance Approximations for Bayesian Deep Learning with Natural Gradient](https://arxiv.org/abs/1811.04504). *NeurIPS*.
- Osawa, K. et al. (2019). [Practical Deep Learning with Bayesian Principles](https://arxiv.org/abs/1906.02506). *NeurIPS*.
- Shen, Y. et al. (2024). [Variational Learning is Effective for Large Deep Networks](https://arxiv.org/abs/2402.17641). *ICML*.
- Lin, W., Schmidt, M. & Khan, M. E. (2020). [Handling the Positive-Definite Constraint in the Bayesian Learning Rule](https://arxiv.org/abs/2002.10060). *ICML*.
- Kunstner, F., Balles, L. & Hennig, P. (2019). [Limitations of the Empirical Fisher Approximation for Natural Gradient Descent](https://arxiv.org/abs/1905.12558). *NeurIPS*.
- Martens, J. (2020). [New Insights and Perspectives on the Natural Gradient Method](https://arxiv.org/abs/1412.1193). *JMLR* 21(146):1–76.
- Opper, M. & Archambeau, C. (2009). The Variational Gaussian Approximation Revisited. *Neural Computation* 21(3):786–792.
- Kirkpatrick, J. et al. (2017). [Overcoming Catastrophic Forgetting in Neural Networks](https://arxiv.org/abs/1612.00796). *PNAS* 114(13):3521–3526.
