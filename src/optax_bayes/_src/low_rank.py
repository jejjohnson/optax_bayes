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
from typing import TYPE_CHECKING

import jax.numpy as jnp
import optax

from optax_bayes._src._optional import require_gaussx
from optax_bayes._src.hessians import resolve_hessian_estimator_full
from optax_bayes._src.linalg import low_rank_precision_operator, solve
from optax_bayes._src.types import BLRLowRankState


if TYPE_CHECKING:
    from optax_bayes._src.linalg import Solver


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
    solver: Solver | None = None,
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
        solver: A ``lineax`` solver (e.g. ``lx.Cholesky()``) or a
            ``gaussx`` solver strategy (e.g. ``gaussx.DenseSolver()``,
            ``gaussx.CGSolver()``). ``None`` uses ``gaussx.solve``'s
            structural dispatch.

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
        d0 = jnp.full(d, prior_precision, params.dtype)
        u0 = jnp.zeros((d, effective_rank), params.dtype)
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
        # The state's dtype (set from the params at init) is the working
        # dtype: inputs are cast to it so nothing promotes the state.
        dtype = state.nat_mean.dtype
        grads = jnp.asarray(grads, dtype)
        d = grads.shape[0]
        d0 = jnp.full(d, prior_precision, dtype)
        m0 = (
            jnp.zeros(d, dtype)
            if prior_mean is None
            else jnp.asarray(prior_mean, dtype)
        )
        eta_0 = d0 * m0

        # Current mean via gaussx structured solve
        m_t = solve(
            low_rank_precision_operator(state.diag_precision, state.low_rank_factor),
            state.nat_mean,
            solver,
        )

        # Decompose -H as diag(-H) + off_diag(-H).  We put the diagonal
        # into D (well-conditioned) and the off-diagonal into U (via the
        # positive eigenvectors).  This O(d^3) eigendecomposition is a
        # scalability bottleneck for large d; a rank-1 GGN-specific fast
        # path could be added later but requires different numerics.
        h = jnp.asarray(_hessian_fn(m_t, grads), dtype)
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
        new_mean = solve(
            low_rank_precision_operator(new_diag, new_u), new_nat_mean, solver
        )
        updates = new_mean - m_t

        new_state = BLRLowRankState(
            diag_precision=new_diag,
            low_rank_factor=new_u,
            nat_mean=new_nat_mean,
            count=state.count + 1,
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)  # ty: ignore[invalid-argument-type]
