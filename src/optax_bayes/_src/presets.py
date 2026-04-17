"""Named algorithm presets as BLR special cases."""

from __future__ import annotations

from collections.abc import Callable

import lineax as lx
import optax

from optax_bayes._src.full_rank import blr_full_rank
from optax_bayes._src.wrappers import _wrap_for_loss


def newton(
    hessian_fn: Callable,
    damping: float = 1e-6,
    solver: lx.AbstractLinearSolver | None = None,
) -> optax.GradientTransformation:
    """Newton's method as full-rank BLR.

    Equivalent to BLR with rho=1, near-flat prior, and exact Hessian.
    Recovers the classic update: m_{t+1} = m_t - H^{-1} g.

    **Expects log-likelihood gradients.**

    Args:
        hessian_fn: Callable ``fn(mean, grads) -> (d, d)`` returning
            the Hessian of the log-likelihood.
        damping: Additive damping epsilon * I for numerical safety.
        solver: Optional ``lineax`` solver.

    Returns:
        An ``optax.GradientTransformation``.
    """
    # Damped Newton: a weak damping * I prior ensures the initial state is
    # invertible and acts as spectral regularisation on the Hessian.  We
    # zero the BLR's per-step damping so damping is applied only once
    # (via the prior), not twice.  With rho=1 this produces the update
    #   m_{t+1} = (damping * I - H_t)^{-1} (g_t - H_t m_t)
    # which converges to classical Newton as damping -> 0.
    return blr_full_rank(
        learning_rate=1.0,
        prior_precision=damping,
        hessian_estimator=hessian_fn,
        damping=0.0,
        solver=solver,
    )


def newton_for_loss(
    loss_hessian_fn: Callable,
    damping: float = 1e-6,
    solver: lx.AbstractLinearSolver | None = None,
) -> optax.GradientTransformation:
    """Newton's method for loss minimisation.

    Accepts standard loss gradients and a loss Hessian function.
    Internally negates both to match the BLR log-likelihood convention.

    Args:
        loss_hessian_fn: Callable ``fn(mean) -> (d, d)`` returning
            the Hessian of the **loss** (not log-likelihood).
        damping: Additive damping epsilon * I for numerical safety.
        solver: Optional ``lineax`` solver.

    Returns:
        An ``optax.GradientTransformation``.
    """

    def _neg_loss_hessian(mean, grads):
        return -loss_hessian_fn(mean)

    inner = newton(
        hessian_fn=_neg_loss_hessian,
        damping=damping,
        solver=solver,
    )
    return _wrap_for_loss(inner)
