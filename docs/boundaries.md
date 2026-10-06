# Scope

optax_bayes owns the BLR update rule as optax transforms. It leaves structured
linear algebra to gaussx, the optimizer protocol to optax, and training loops
to you.

## Ownership

| Concern | Owner |
|---|---|
| BLR updates (diagonal, full-rank, low-rank) and IVON | optax_bayes |
| Hessian estimators and loss-convention wrappers | optax_bayes |
| Posterior extraction and sampling from optimizer state | optax_bayes |
| Structured precision operators, Woodbury solves | gaussx |
| The optimizer protocol, chaining, schedules, clipping | optax |
| Training loops, data, models | your code |

## Which transform?

| Situation | Start with |
|---|---|
| Adam replacement with uncertainty | `blr_diagonal_for_loss(learning_rate=1e-3, prior_precision=1e-4)` |
| Deep network, IVON-style training | `ivon(...)` with `sample_ivon` before each gradient |
| Small model, full posterior covariance | `blr_full_rank_for_loss(hessian_estimator=...)` |
| Correlated uncertainty for large $d$ | `blr_low_rank_for_loss(rank=r)` |
| Exact posterior of a conjugate model | full-rank or low-rank ($r \ge d$) with the exact Hessian as a callable |
| Newton's method | `newton_for_loss(loss_hessian_fn)` |
| Gradient clipping | `optax.chain(optax.clip_by_global_norm(1.0), blr_diagonal_for_loss(...))` |
| Learning-rate schedule (diagonal) | `blr_with_schedule(schedule_fn)` |
| Continual learning | Read task 1's posterior and use it as task 2's prior |

## In scope

- Gaussian variational families: diagonal, full-rank, low-rank plus diagonal.
- Hessian estimators: squared-gradient (`"ggn_diag"`, `"ggn"`), identity, or
  a user-supplied callable (full-rank and low-rank).
- Posterior extraction and reparameterised sampling from optimizer state.
- Loss-convention wrappers and schedule composition.

## Out of scope

- Non-Gaussian exponential families.
- MCMC and general-purpose SVI: use NumPyro.
- Model definitions and training loops.

## Testing strategy

| Category | What it checks | Where |
|---|---|---|
| Update correctness | One step matches the hand-derived formula | `test_primitives.py`, `test_diagonal.py` |
| Posterior recovery | Exact-Hessian BLR reaches the analytic posterior of Bayesian linear regression | `test_recovery.py` (slow) |
| Low-rank structure | $r \ge d$ is exact and matches the full-rank BLR; $r < d$ keeps the diagonal exact; no $(d, d)$ arrays for `"ggn"` | `test_low_rank.py` |
| optax protocol | `jit`, `scan`, `chain`, pytree round-trips; schedules via `inject_hyperparams` | `test_protocol.py`, `test_extras.py` |
| Linear algebra | PSD tags, every `solver=` kind, Newton on an indefinite Hessian | `test_linalg.py` |
| Dtypes | float32 params stay float32 under x64; the suite also runs with x64 off | `test_dtypes.py`, `make test-no-x64` |
| Optional dependency | Diagonal and IVON import and run without gaussx | `test_optional_deps.py` |

## Open questions

1. **Per-parameter priors.** `prior_precision` and `prior_mean` are scalars.
   Continual learning wants pytree priors that match the parameter tree.
2. **Pytree support beyond diagonal.** The full-rank and low-rank transforms
   take flat vectors; users flatten with `jax.flatten_util.ravel_pytree`.
3. **Per-example curvature.** `"ggn"` uses the gradient it is given. A true
   Gauss–Newton or Fisher estimate needs per-example Jacobians, currently
   only through a callable, which is $O(d^2)$ memory.
