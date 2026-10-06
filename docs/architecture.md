# Architecture

## Three layers

```text
┌──────────────────────────────────────────────────────────────────────┐
│ Layer 2: wrappers and utilities                                      │
│   *_for_loss wrappers, blr_with_schedule, newton / newton_for_loss   │
│   get_posterior_*, sample_posterior_*                                │
├──────────────────────────────────────────────────────────────────────┤
│ Layer 1: optax GradientTransformations                               │
│   blr_diagonal, blr_full_rank, blr_low_rank, ivon                    │
│   BLRDiagState, BLRFullRankState, BLRLowRankState, IVONState         │
├──────────────────────────────────────────────────────────────────────┤
│ Layer 0: primitives                                                  │
│   blr_diag_update_step, natural <-> mean conversions                 │
│   Hessian estimators: ggn_diag, ggn_outer, low-rank factors          │
│   PSD-tagged precision operators and solves (linalg)                 │
└──────────────────────────────────────────────────────────────────────┘

Foundations (not owned by optax_bayes):
  optax (protocol, combinators) · jax · lineax · gaussx (optional)
```

**Layer 0** is stateless maths on arrays: the diagonal update step, the
parameter conversions, the Hessian estimators, and the helpers in `linalg`
that wrap a precision as a PSD-tagged operator and solve with it.

**Layer 1** turns those into `optax.GradientTransformation`s with `init` and
`update`. Each state is a `NamedTuple` of natural parameters, so it is a
pytree: it can be a `jax.lax.scan` carry and it checkpoints like any other
optax state. Each `update` returns $m_{t+1} - m_t$, so after
`optax.apply_updates` the params equal the variational mean.

**Layer 2** adds conveniences: the `*_for_loss` wrappers negate loss gradients
(and callable loss Hessians), `blr_with_schedule` drives $\rho$ from an optax
schedule, `newton` is the full-rank BLR with $\rho = 1$, and the
`get_posterior_*` / `sample_posterior_*` helpers read the state.

## gaussx integration

The full-rank and low-rank variants use
[gaussx](https://github.com/jejjohnson/gaussx) for their linear algebra:

| Operation | gaussx call | Notes |
|---|---|---|
| Mean $m = \Lambda^{-1}\eta$ (dense) | `gaussx.solve` on a PSD-tagged `lx.MatrixLinearOperator` | Cholesky; `newton` leaves the tag off, since its Hessian may be indefinite |
| Low-rank precision $\operatorname{diag}(D) + UU^\top$ | `gaussx.low_rank_plus_diag(D, U, psd=True)` | A `LowRankUpdate` operator |
| Low-rank solve | `gaussx.solve` | Woodbury, $O(dr^2 + r^3)$ |
| Posterior covariance | `gaussx.inv(...).as_matrix()` | Structured inverse, then densified |

Every `solver=` argument takes either a `lineax` solver (passed through to
`gaussx.solve`) or a gaussx solver strategy such as `gaussx.DenseSolver()` or
`gaussx.CGSolver()`.

gaussx is imported lazily through `optax_bayes._src._optional.require_gaussx`,
so `import optax_bayes` and the diagonal and IVON transforms work without it.

## Package layout

```text
src/optax_bayes/
├── __init__.py        # public API
└── _src/
    ├── primitives.py  # L0: blr_diag_update_step
    ├── conversions.py # L0: natural <-> mean parameters (diagonal)
    ├── hessians.py    # L0: diagonal, dense and low-rank-factor estimators
    ├── linalg.py      # L0: PSD-tagged precision operators, solves, inverses
    ├── _optional.py   # lazy gaussx import
    ├── types.py       # L1: state NamedTuples
    ├── diagonal.py    # L1: blr_diagonal
    ├── full_rank.py   # L1: blr_full_rank
    ├── low_rank.py    # L1: blr_low_rank
    ├── ivon.py        # L1: ivon, sample_ivon, get_posterior_ivon
    ├── wrappers.py    # L2: *_for_loss
    ├── schedules.py   # L2: blr_with_schedule
    ├── presets.py     # L2: newton, newton_for_loss
    ├── posterior.py   # L2: get_posterior_*
    └── sampling.py    # L2: sample_posterior_*
```

## Dependencies

| Package | Required | Role |
|---|---|---|
| `jax` | yes | Arrays, autodiff, `jit`, `vmap`, `scan` |
| `optax` | yes | The `GradientTransformation` protocol and combinators |
| `jaxtyping` | yes | Array annotations |
| `lineax` | yes | Linear-operator types and tags |
| `gaussx` | `[gaussx]` extra | Structured solves for the full-rank and low-rank variants |
| `equinox` | `[equinox]` extra | Examples only |

## Quality gates

| Check | Command | Runs on |
|---|---|---|
| Fast tests | `make test-fast` | Every PR |
| Float32 tests | `make test-no-x64` | Every PR |
| Tests, Python 3.12 and 3.13, with coverage | `uv run pytest -m "not slow and not integration" --cov` | Pushes to `main` |
| Slow and integration tests | `make test-slow` | On demand (`tests-extended.yml`) |
| Lint and format | `ruff check .`, `ruff format --check .` | Every PR |
| Type check | `ty check src/optax_bayes` | Every PR |

Commits follow Conventional Commits, and release-please cuts releases from
them.
