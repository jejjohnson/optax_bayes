"""Full-rank BLR as an optax GradientTransformation.

Operates on flat parameter vectors theta in R^d.  The precision
Lambda is a (d, d) matrix, and the natural mean eta = Lambda @ m
is a (d,) vector.

Uses ``gaussx.solve`` for the precision solve, which dispatches
to structure-aware solvers via lineax.

Expects log-likelihood gradients.  Most users should use
``blr_full_rank_for_loss`` from the wrappers module instead.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import lineax as lx
import optax

from optax_bayes._src._optional import require_gaussx
from optax_bayes._src.hessians import resolve_hessian_estimator_full
from optax_bayes._src.types import BLRFullRankState


def blr_full_rank(
    learning_rate: float = 1e-2,
    prior_precision: float = 1e-4,
    prior_mean: jnp.ndarray | None = None,
    hessian_estimator: str | Callable = "ggn",
    damping: float = 1e-6,
    solver: lx.AbstractLinearSolver | None = None,
) -> optax.GradientTransformation:
    r"""Full-rank Gaussian BLR as an optax transform.

    Implements the Bayesian Learning Rule (Khan & Rue, 2023) with a
    full-rank Gaussian variational family
    $q(\theta) = \mathcal{N}(m, \Lambda^{-1})$ over flat parameter
    vectors $\theta \in \mathbb{R}^d$. Optimizer state stores the
    precision matrix $\Lambda$ (``(d, d)``) and the natural mean
    $\eta = \Lambda m$ (``(d,)``), updated as

    $$
    \begin{aligned}
    \Lambda_{t+1} &= (1 - \rho)\, \Lambda_t + \rho\, (\Lambda_0 - H_t) \\
    \eta_{t+1}    &= (1 - \rho)\, \eta_t    + \rho\, (\eta_0 + g_t - H_t m_t)
    \end{aligned}
    $$

    with $g_t$ the log-likelihood gradient and $H_t \preceq 0$ the
    Hessian estimate. With the exact Hessian, the fixed point is the
    exact Gaussian posterior of a conjugate model. The state initialises
    its mean at the params passed to ``init``; ``prior_mean`` and
    ``prior_precision`` anchor every update.

    Uses ``gaussx.solve`` for the precision solve at each step (requires
    the optional ``gaussx`` extra).

    **This API expects log-likelihood gradients.**  For standard loss
    minimisation, use
    [`blr_full_rank_for_loss`][optax_bayes.blr_full_rank_for_loss] instead.

    Args:
        learning_rate: Step size rho in (0, 1].
        prior_precision: Scalar multiplied by I to form Lambda_0.
        prior_mean: Prior mean vector (d,), or None for zeros.
        hessian_estimator: ``"ggn"`` (outer product ``-g g^T``),
            ``"identity"`` (zero), or a callable
            ``fn(mean, grads) -> (d, d)``.
        damping: Additive damping epsilon * I after each precision
            update.
        solver: A ``gaussx`` solver strategy (e.g.
            ``gaussx.DenseSolver()``, ``gaussx.CGSolver()``).
            ``None`` uses the default ``gaussx.solve`` dispatch.

    Returns:
        An ``optax.GradientTransformation``.

    Raises:
        ImportError: If the optional ``gaussx`` dependency is not
            installed.
    """
    gaussx = require_gaussx("blr_full_rank")
    _hessian_fn = resolve_hessian_estimator_full(hessian_estimator)

    def init_fn(params: jnp.ndarray) -> BLRFullRankState:
        d = params.shape[0]
        lambda_0 = prior_precision * jnp.eye(d)
        # The variational mean starts at the user's params (standard optax
        # drop-in semantics).  The prior mean still anchors every update
        # through eta_0 inside update_fn.
        return BLRFullRankState(
            precision=lambda_0,
            nat_mean=lambda_0 @ params,
            count=jnp.zeros([], jnp.int32),
        )

    def update_fn(
        grads: jnp.ndarray,
        state: BLRFullRankState,
        params: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, BLRFullRankState]:
        rho = learning_rate
        d = grads.shape[0]
        lambda_0 = prior_precision * jnp.eye(d)
        m0 = jnp.zeros(d) if prior_mean is None else prior_mean
        eta_0 = lambda_0 @ m0

        # Current mean: m_t = Lambda_t^{-1} eta_t
        op: lx.AbstractLinearOperator = lx.MatrixLinearOperator(state.precision)
        m_t = gaussx.solve(op, state.nat_mean, solver=solver)

        # Hessian estimate
        h = _hessian_fn(m_t, grads)

        # Precision update
        new_precision = (1 - rho) * state.precision + rho * (lambda_0 - h)
        new_precision = new_precision + damping * jnp.eye(d)

        # Natural mean update
        grad_mu1 = grads - h @ m_t
        new_nat_mean = (1 - rho) * state.nat_mean + rho * (eta_0 + grad_mu1)

        # Recover mean and compute update
        new_op: lx.AbstractLinearOperator = lx.MatrixLinearOperator(new_precision)
        new_mean = gaussx.solve(new_op, new_nat_mean, solver=solver)
        updates = new_mean - m_t

        new_state = BLRFullRankState(
            precision=new_precision,
            nat_mean=new_nat_mean,
            count=state.count + 1,
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)  # ty: ignore[invalid-argument-type]
