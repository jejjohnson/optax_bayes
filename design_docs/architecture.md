---
status: draft
version: 0.1.0
---

# Architecture

## Three-Layer Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 2 — Loss Wrappers & Utilities                                    │
│  blr_diagonal_for_loss, blr_full_rank_for_loss, blr_with_schedule       │
│  get_posterior_diagonal, get_posterior_full_rank                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 1 — optax GradientTransformations                                │
│  blr_diagonal, blr_full_rank, blr_low_rank                             │
│  BLRDiagState, BLRFullRankState                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 0 — Primitives                                                   │
│  Pure JAX functions: blr_diag_update_step, blr_full_update_step         │
│  Hessian estimators: ggn_diag, hutchinson_diag, identity_hessian        │
│  Natural ↔ expectation parameter conversions                            │
└─────────────────────────────────────────────────────────────────────────┘

Foundation (not owned by optax-bayes):
┌──────────┐  ┌──────────┐  ┌──────────┐
│  optax   │  │  gaussx  │  │   jax    │
│ (protocol│  │ (expfam, │  │ (autodiff│
│  + chain)│  │  solve,  │  │  vmap,   │
│          │  │  logdet) │  │  jit)    │
└──────────┘  └──────────┘  └──────────┘
```

**Layer 0** is pure JAX math. The BLR update equations from Khan & Rue (2023) as stateless functions:

$$\eta_{t+1} = (1 - \rho)\eta_t + \rho[\eta_0 + g_t - H_t m_t], \quad s_{t+1} = (1 - \rho)s_t + \rho[s_0 - \operatorname{diag}(H_t)]$$

No optax, no gaussx — just arrays.

**Layer 1** wraps L0 functions into `optax.GradientTransformation` with proper `init` and `update`. State types (`BLRDiagState`, `BLRFullRankState`) store natural parameters.

**Layer 2** provides convenience: `_for_loss` wrappers (negate gradients), `blr_with_schedule` (compose with optax schedules), and `get_posterior_*` (extract mean + variance/covariance from state).

---

## gaussx Integration

Full-rank and low-rank BLR variants use gaussx for structured precision operations:

| optax-bayes operation | gaussx backend | Benefit |
|---|---|---|
| Recover mean $m = \Lambda^{-1}\eta$ | `gaussx.ops.solve(Lambda_op, eta)` | Woodbury when low-rank |
| Log-normalizer $\log|\Lambda|$ | `gaussx.ops.logdet(Lambda_op)` | Matrix determinant lemma |
| Low-rank precision $\Lambda = D + UU^T$ | `gaussx.operators.LowRankUpdate` | $O(dr^2)$ not $O(d^3)$ |
| Natural ↔ expectation conversion | `gaussx.expfam.GaussianExpFam` | Structure-preserving |
| KL divergence | `gaussx.expfam.kl_divergence` | Bregman divergence |

---

## Package Layout

```
optax_bayes/
├── __init__.py
├── _src/
│   ├── primitives.py         # L0: blr_diag_update_step, blr_full_update_step
│   ├── hessians.py           # L0: ggn_diag, hutchinson_diag, identity_hessian
│   ├── conversions.py        # L0: natural_to_mean, mean_to_natural
│   ├── diagonal.py           # L1: blr_diagonal → GradientTransformation
│   ├── full_rank.py          # L1: blr_full_rank → GradientTransformation
│   ├── low_rank.py           # L1: blr_low_rank → GradientTransformation
│   ├── wrappers.py           # L2: _for_loss, blr_with_schedule
│   ├── posterior.py           # L2: get_posterior_diagonal, get_posterior_full_rank
│   └── types.py              # BLRDiagState, BLRFullRankState
```

---

## Dependencies

### Required

| Package | Role |
|---|---|
| `jax` | Array computation, autodiff, JIT, vmap |
| `optax` | GradientTransformation protocol, apply_updates, schedules |

### Optional

| Package | Role | Used by |
|---|---|---|
| `gaussx` | Structured precision operators, expfam | Full-rank and low-rank BLR |
| `equinox` | Module system for Equinox model training | Examples |

---

## CI / Quality Gates

| Check | Command | Scope |
|-------|---------|-------|
| Tests | `uv run pytest tests -x` | Full suite |
| Lint | `uv run ruff check .` | Entire repo |
| Format | `uv run ruff format --check .` | Entire repo |
| Typecheck | `uv run ty check optax_bayes` | Package only |

All four must pass before merge. GitHub Actions on push/PR.
Conventional commits required (`feat:`, `fix:`, `docs:`, `test:`, etc.).

**Build system:** hatchling (PEP 621)
**Python:** >= 3.12, < 3.14
**License:** MIT
