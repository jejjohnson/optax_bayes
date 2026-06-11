"""Low-rank BLR as an optax GradientTransformation.

Parameterises the precision as Lambda = diag(D) + U U^T where
D is (d,) and U is (d, r), giving O(dr) storage instead of O(d^2).

Uses ``gaussx.LowRankUpdate`` as the structured operator and
``gaussx.solve`` for the Woodbury-dispatched solve.

Expects log-likelihood gradients.  Most users should use
``blr_low_rank_for_loss`` from the wrappers module instead.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import lineax as lx
import optax

from optax_bayes._src._optional import require_gaussx
from optax_bayes._src.hessians import resolve_hessian_estimator_full
from optax_bayes._src.types import BLRLowRankState


def _build_low_rank_operator(
    d_diag: jnp.ndarray, u: jnp.ndarray
) -> lx.AbstractLinearOperator:
    """Build a gaussx LowRankUpdate operator for diag(D) + U U^T.

    Args:
        d_diag: Diagonal entries, shape (d,).
        u: Low-rank factor, shape (d, r).

    Returns:
        A ``gaussx.LowRankUpdate`` operator.
    """
    gaussx = require_gaussx("low-rank BLR operators")
    return gaussx.low_rank_plus_diag(d_diag, u)


def _low_rank_solve(
    d_diag: jnp.ndarray,
    u: jnp.ndarray,
    b: jnp.ndarray,
    solver: lx.AbstractLinearSolver | None = None,
) -> jnp.ndarray:
    """Solve (diag(D) + U U^T) x = b via gaussx.

    Args:
        d_diag: Diagonal entries, shape (d,).
        u: Low-rank factor, shape (d, r).
        b: Right-hand side, shape (d,).
        solver: Optional gaussx solver strategy.

    Returns:
        Solution x, shape (d,).
    """
    gaussx = require_gaussx("low-rank BLR solves")
    op = _build_low_rank_operator(d_diag, u)
    return gaussx.solve(op, b, solver=solver)


def _truncate_to_rank(u: jnp.ndarray, rank: int) -> jnp.ndarray:
    """Truncate U from (d, r+k) to (d, rank) via SVD.

    Keeps the top-``rank`` singular vectors scaled by singular values,
    so that U_trunc @ U_trunc^T approximates U @ U^T.

    Args:
        u: Factor matrix, shape (d, r+k) where r+k > rank.
        rank: Target rank.

    Returns:
        Truncated factor, shape (d, rank).
    """
    p, s, _qt = jnp.linalg.svd(u, full_matrices=False)  # ty: ignore[invalid-argument-type, unknown-argument, not-iterable]
    return p[:, :rank] * s[None, :rank]


def blr_low_rank(
    learning_rate: float = 1e-2,
    rank: int = 10,
    prior_precision: float = 1e-4,
    prior_mean: jnp.ndarray | None = None,
    hessian_estimator: str | Callable = "ggn",
    damping: float = 1e-6,
    solver: lx.AbstractLinearSolver | None = None,
) -> optax.GradientTransformation:
    r"""Low-rank Gaussian BLR as an optax transform.

    Parameterises the precision as
    $\Lambda = \operatorname{diag}(D) + U U^\top$ with $D \in
    \mathbb{R}^d$ and $U \in \mathbb{R}^{d \times r}$, giving
    $O(dr)$ storage and $O(dr^2 + r^3)$ solves via the Woodbury
    identity through ``gaussx`` structured operators (requires the
    optional ``gaussx`` extra).

    Each step splits the Hessian estimate $-H_t$ into its diagonal
    (absorbed into $D$) and the positive eigen-part of its
    off-diagonal remainder (appended to $U$); the augmented factor is
    truncated back to rank $r$ via SVD so that
    $U_{t+1} U_{t+1}^\top \approx (1-\rho)\, U_t U_t^\top + \rho\,
    \text{(new curvature)}$. The natural mean follows the standard BLR
    update. The state initialises its mean at the params passed to
    ``init``; ``prior_mean`` and ``prior_precision`` anchor every
    update.

    **This API expects log-likelihood gradients.**  For standard loss
    minimisation, use
    [`blr_low_rank_for_loss`][optax_bayes.blr_low_rank_for_loss] instead.

    Args:
        learning_rate: Step size rho in (0, 1].
        rank: Target rank r of the low-rank factor U.
        prior_precision: Scalar diagonal prior precision D_0 = s0 * I.
        prior_mean: Prior mean vector (d,), or None for zeros.
        hessian_estimator: ``"ggn"`` (outer product ``-g g^T``),
            ``"identity"`` (zero), or a callable
            ``fn(mean, grads) -> (d, d)``.
        damping: Additive damping on the diagonal after each update.
        solver: A ``gaussx`` solver strategy (e.g.
            ``gaussx.DenseSolver()``, ``gaussx.CGSolver()``).
            ``None`` uses the default ``gaussx.solve`` dispatch.

    Returns:
        An ``optax.GradientTransformation``.

    Raises:
        ImportError: If the optional ``gaussx`` dependency is not
            installed.
    """
    require_gaussx("blr_low_rank")
    _hessian_fn = resolve_hessian_estimator_full(hessian_estimator)

    def init_fn(params: jnp.ndarray) -> BLRLowRankState:
        d = params.shape[0]
        # Clamp rank to d: SVD of a (d, k) matrix returns at most d singular
        # vectors, so a rank > d request would be silently truncated later
        # and produce a shape mismatch.
        effective_rank = min(rank, d)
        d0 = jnp.full(d, prior_precision)
        u0 = jnp.zeros((d, effective_rank))
        # The variational mean starts at the user's params (standard optax
        # drop-in semantics).  The prior mean still anchors every update
        # through eta_0 inside update_fn.
        eta_0 = d0 * params
        return BLRLowRankState(
            diag_precision=d0,
            low_rank_factor=u0,
            nat_mean=eta_0,
            count=jnp.zeros([], jnp.int32),
        )

    def update_fn(
        grads: jnp.ndarray,
        state: BLRLowRankState,
        params: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, BLRLowRankState]:
        rho = learning_rate
        d = grads.shape[0]
        d0 = jnp.full(d, prior_precision)
        m0 = jnp.zeros(d) if prior_mean is None else prior_mean
        eta_0 = d0 * m0

        # Current mean via gaussx structured solve
        m_t = _low_rank_solve(
            state.diag_precision,
            state.low_rank_factor,
            state.nat_mean,
            solver=solver,
        )

        # Decompose -H as diag(-H) + off_diag(-H).  We put the diagonal
        # into D (well-conditioned) and the off-diagonal into U (via the
        # positive eigenvectors).  This O(d^3) eigendecomposition is a
        # scalability bottleneck for large d; a rank-1 GGN-specific fast
        # path could be added later but requires different numerics.
        h = _hessian_fn(m_t, grads)
        h_m_t = h @ m_t

        # Diagonal precision update
        new_diag = (1 - rho) * state.diag_precision + rho * (d0 - jnp.diag(h))
        new_diag = jnp.maximum(new_diag, damping)

        # Low-rank factor update: extract positive eigenvectors of -H's
        # off-diagonal part.
        neg_h = -h
        neg_h_offdiag = neg_h - jnp.diag(jnp.diag(neg_h))
        eigvals, eigvecs = jnp.linalg.eigh(neg_h_offdiag)
        pos_mask = eigvals > 0
        h_factor = eigvecs * jnp.sqrt(jnp.maximum(eigvals, 0.0))[None, :]
        h_factor = jnp.where(pos_mask[None, :], h_factor, 0.0)

        # Low-rank factor update:
        # U_{new} U_{new}^T ≈ (1-rho) U U^T + rho * h_factor @ h_factor^T
        u_scaled = jnp.sqrt(jnp.maximum(1 - rho, 0.0)) * state.low_rank_factor
        h_scaled = jnp.sqrt(rho) * h_factor
        u_aug = jnp.concatenate([u_scaled, h_scaled], axis=1)

        new_u = _truncate_to_rank(u_aug, min(rank, d))

        # Natural mean update
        grad_mu1 = grads - h_m_t
        new_nat_mean = (1 - rho) * state.nat_mean + rho * (eta_0 + grad_mu1)

        # Recover new mean via gaussx structured solve
        new_mean = _low_rank_solve(new_diag, new_u, new_nat_mean, solver=solver)
        updates = new_mean - m_t

        new_state = BLRLowRankState(
            diag_precision=new_diag,
            low_rank_factor=new_u,
            nat_mean=new_nat_mean,
            count=state.count + 1,
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)  # ty: ignore[invalid-argument-type]
