---
status: draft
version: 0.1.0
---

# Layer 1 — Components

optax `GradientTransformation` wrappers around Layer 0 primitives.

---

## `blr_diagonal`

```python
def blr_diagonal(
    learning_rate: float = 1e-2,
    prior_precision: float = 1e-4,
    prior_mean: float = 0.0,
    hessian_estimator: str = "ggn_diag",
    damping: float = 1e-6,
) -> optax.GradientTransformation:
    """BLR-Diagonal as optax transform. Expects log-likelihood gradients."""
```

**State:** `BLRDiagState(precision, nat_mean, count)` — natural parameters as pytrees matching the parameter tree.

**Memory:** Same as Adam — one extra array per parameter (the precision $s$).

---

## `blr_full_rank`

```python
def blr_full_rank(
    learning_rate: float = 1e-2,
    prior_precision: float = 1e-4,
    damping: float = 1e-6,
    hessian_fn: Callable | None = None,
) -> optax.GradientTransformation:
    """BLR-FullRank as optax transform. Expects log-likelihood gradients.

    Uses gaussx for structured precision solve when available.
    """
```

**State:** `BLRFullRankState(precision, nat_mean, count)` — precision is $d \times d$ matrix (or gaussx operator for low-rank).

---

## `blr_low_rank` (planned)

```python
def blr_low_rank(
    learning_rate: float = 1e-2,
    rank: int = 10,
    prior_precision: float = 1e-4,
    damping: float = 1e-6,
) -> optax.GradientTransformation:
    """BLR-LowRank: Λ = D + UUᵀ via gaussx Woodbury. O(dr²) per step."""
```

---

## Model Zoo (special cases)

The BLR unifies classical algorithms. See [../research/overview.md](../research/overview.md) §3 for the full table.

| Algorithm | BLR configuration |
|---|---|
| SGD | Diagonal, $H = 0$, $\rho = 1$, flat prior |
| Adam (approx) | Diagonal, $H = -g^2$, EMA rates $\beta_1, \beta_2$ |
| Newton's method | Full-rank, exact Hessian, $\rho = 1$, flat prior |
| Kalman filter | Full-rank, $\rho = 1$, observation Hessian, sequential prior |
| Mean-field VI | Diagonal, exact diagonal Hessian, $\rho_t \to 0$ |
| Full-rank VI | Full-rank, GGN Hessian, $\rho_t \to 0$ |
| EWC (continual learning) | Diagonal, task-1 posterior = task-2 prior |
