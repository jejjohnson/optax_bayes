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
from optax_bayes._src.hessians import resolve_hessian_factor_low_rank
from optax_bayes._src.linalg import low_rank_precision_operator, solve
from optax_bayes._src.types import BLRLowRankState


if TYPE_CHECKING:
    from optax_bayes._src.linalg import Solver


def _truncate_to_rank(u: jnp.ndarray, rank: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Truncate U from (d, k) to (d, rank) via a thin SVD.

    Keeps the top-``rank`` singular vectors scaled by singular values,
    the best rank-``rank`` approximation of U U^T. Costs O(d k^2).

    Args:
        u: Factor matrix, shape (d, k).
        rank: Target rank, at most ``min(d, k)``.

    Returns:
        Tuple ``(u_trunc, residual_diag)``: the (d, rank) factor and
        ``diag(U U^T - U_trunc U_trunc^T)`` (non-negative), the diagonal
        of the discarded part.
    """
    p, s, _qt = jnp.linalg.svd(u, full_matrices=False)  # ty: ignore[invalid-argument-type, unknown-argument, not-iterable]
    scaled = p * s[None, :]
    residual_diag = jnp.sum(scaled[:, rank:] ** 2, axis=1)
    return scaled[:, :rank], residual_diag


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

    The update is SLANG's (Mishkin et al., 2018). With the curvature in
    factor form $-H_t = G_t G_t^\top$ ($G_t = g_t$ for ``"ggn"``), the
    target precision $(1-\rho)\Lambda_t + \rho(\Lambda_0 - H_t)$ is
    $\operatorname{diag}((1-\rho) D_t + \rho D_0) + \tilde U \tilde U^\top$
    with $\tilde U = [\sqrt{1-\rho}\, U_t,\ \sqrt{\rho}\, G_t]$. A thin
    SVD keeps the top $r$ directions of $\tilde U$, and the diagonal of
    the discarded part is added to $D$:

    $$
    \begin{aligned}
    U_{t+1} &= \operatorname{top}_r(\tilde U) \\
    D_{t+1} &= (1-\rho) D_t + \rho D_0
        + \operatorname{diag}(\tilde U \tilde U^\top - U_{t+1} U_{t+1}^\top)
        + \epsilon
    \end{aligned}
    $$

    so the diagonal of the precision is updated exactly, and the whole
    precision is exact when $r \ge d$. A step costs $O(d (r + k)^2)$ for
    a rank-$k$ factor: $O(d r^2)$ for ``"ggn"``. The mean takes the
    mean-form Gaussian BLR step

    $$
    m_{t+1} = m_t + \rho\, \Lambda_{t+1}^{-1} \big(g_t - D_0 (m_t - m_0)\big),
    $$

    which matches the natural-mean recursion of ``blr_full_rank`` whenever
    the precision is exact and stays stable when it is truncated. The
    state initialises its mean at the params passed to ``init``;
    ``prior_mean`` and ``prior_precision`` anchor every update.

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
            ``fn(mean, grads) -> (d, d)``. A callable's dense Hessian is
            factored by eigendecomposition, $O(d^3)$ per step, keeping
            only its negative-definite part.
        damping: Additive damping epsilon on the diagonal after each
            update, as in ``blr_full_rank``.
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
    _factor_fn = resolve_hessian_factor_low_rank(hessian_estimator)

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

        # Current mean via gaussx structured solve
        m_t = solve(
            low_rank_precision_operator(state.diag_precision, state.low_rank_factor),
            state.nat_mean,
            solver,
        )

        # Curvature in factor form, -H = G G^T, with G of shape (d, k).
        g_factor = jnp.asarray(_factor_fn(m_t, grads), dtype)

        # SLANG precision update: stack the decayed factor with the new
        # curvature, keep the top-r directions, and fold the diagonal of
        # what was discarded into D.
        u_aug = jnp.concatenate(
            [
                jnp.sqrt(jnp.maximum(1 - rho, 0.0)) * state.low_rank_factor,
                jnp.sqrt(rho) * g_factor,
            ],
            axis=1,
        )
        new_u, residual_diag = _truncate_to_rank(u_aug, min(rank, d))
        new_diag = (1 - rho) * state.diag_precision + rho * d0 + residual_diag + damping

        # Mean update in mean form, m += rho * Lambda^{-1} (g - D_0 (m - m_0)).
        # It equals the natural-mean recursion when the precision is exact,
        # but stays a well-defined preconditioned step after truncation;
        # the recursion would instead amplify the truncation error by
        # Lambda^{-1} along weakly curved directions.
        new_op = low_rank_precision_operator(new_diag, new_u)
        updates = solve(new_op, rho * (grads - d0 * (m_t - m0)), solver)
        new_mean = m_t + updates
        new_nat_mean = new_op.mv(new_mean)

        new_state = BLRLowRankState(
            diag_precision=new_diag,
            low_rank_factor=new_u,
            nat_mean=new_nat_mean,
            count=state.count + 1,
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)  # ty: ignore[invalid-argument-type]
