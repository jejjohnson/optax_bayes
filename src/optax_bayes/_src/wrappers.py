"""Loss-convention wrappers for BLR transforms.

Most users minimise a loss L(theta) = -log p(D|theta).  These wrappers
negate the incoming gradients so the inner transforms receive
log-likelihood gradients as expected.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import lineax as lx
import optax

from optax_bayes._src.diagonal import blr_diagonal
from optax_bayes._src.full_rank import blr_full_rank
from optax_bayes._src.low_rank import blr_low_rank


def _wrap_for_loss(inner: optax.GradientTransformation):
    """Wrap a log-likelihood transform into a loss-minimisation one."""

    def init_fn(params):
        return inner.init(params)

    def update_fn(grads, state, params=None):
        neg_grads = jax.tree.map(jnp.negative, grads)
        return inner.update(neg_grads, state, params)

    return optax.GradientTransformation(init_fn, update_fn)  # ty: ignore[invalid-argument-type]


def _negate_hessian_for_loss(
    hessian_estimator: str | Callable,
) -> str | Callable:
    """If *hessian_estimator* is a callable returning a **loss** Hessian,
    wrap it to return the log-likelihood Hessian (negated).  String
    selectors are passed through unchanged because they are computed
    from gradients internally (squared, so sign-invariant).
    """
    if callable(hessian_estimator):
        user_fn = hessian_estimator
        return lambda mean, grads: -user_fn(mean, grads)  # ty: ignore[call-top-callable]
    return hessian_estimator


def blr_diagonal_for_loss(
    learning_rate: float = 1e-2,
    prior_precision: float = 1e-4,
    prior_mean: float = 0.0,
    hessian_estimator: str = "ggn_diag",
    damping: float = 1e-6,
) -> optax.GradientTransformation:
    """Diagonal BLR accepting standard loss gradients.

    Drop-in replacement for ``optax.adam`` / ``optax.sgd``.
    Internally negates gradients: ``g_loglik = -g_loss``.

    Args:
        learning_rate: Step size rho in (0, 1].
        prior_precision: Scalar prior precision s0 = 1/v0.
        prior_mean: Scalar prior mean m0.
        hessian_estimator: One of ``"ggn_diag"`` or ``"identity"``.
        damping: Minimum precision after update.

    Returns:
        An ``optax.GradientTransformation``.
    """
    return _wrap_for_loss(
        blr_diagonal(
            learning_rate=learning_rate,
            prior_precision=prior_precision,
            prior_mean=prior_mean,
            hessian_estimator=hessian_estimator,
            damping=damping,
        )
    )


def blr_full_rank_for_loss(
    learning_rate: float = 1e-2,
    prior_precision: float = 1e-4,
    prior_mean: jnp.ndarray | None = None,
    hessian_estimator: str | Callable = "ggn",
    damping: float = 1e-6,
    solver: lx.AbstractLinearSolver | None = None,
) -> optax.GradientTransformation:
    """Full-rank BLR accepting standard loss gradients.

    Internally negates gradients (``g_loglik = -g_loss``) and, for
    callable Hessian estimators, negates the returned matrix so the
    inner BLR transform receives log-likelihood Hessians.

    Args:
        learning_rate: Step size rho in (0, 1].
        prior_precision: Scalar multiplied by I to form Lambda_0.
        prior_mean: Prior mean vector (d,), or None for zeros.
        hessian_estimator: ``"ggn"``, ``"identity"``, or a callable
            ``fn(mean, grads) -> (d, d)`` returning the **loss** Hessian.
        damping: Additive damping epsilon * I.
        solver: Optional ``lineax`` solver.

    Returns:
        An ``optax.GradientTransformation``.
    """
    return _wrap_for_loss(
        blr_full_rank(
            learning_rate=learning_rate,
            prior_precision=prior_precision,
            prior_mean=prior_mean,
            hessian_estimator=_negate_hessian_for_loss(hessian_estimator),
            damping=damping,
            solver=solver,
        )
    )


def blr_low_rank_for_loss(
    learning_rate: float = 1e-2,
    rank: int = 10,
    prior_precision: float = 1e-4,
    prior_mean: jnp.ndarray | None = None,
    hessian_estimator: str | Callable = "ggn",
    damping: float = 1e-6,
    solver: lx.AbstractLinearSolver | None = None,
) -> optax.GradientTransformation:
    """Low-rank BLR accepting standard loss gradients.

    Internally negates gradients (``g_loglik = -g_loss``) and, for
    callable Hessian estimators, negates the returned matrix so the
    inner BLR transform receives log-likelihood Hessians.

    Args:
        learning_rate: Step size rho in (0, 1].
        rank: Target rank r of the low-rank factor U.
        prior_precision: Scalar diagonal prior precision.
        prior_mean: Prior mean vector (d,), or None for zeros.
        hessian_estimator: ``"ggn"``, ``"identity"``, or a callable
            ``fn(mean, grads) -> (d, d)`` returning the **loss** Hessian.
        damping: Additive damping on the diagonal.
        solver: Optional ``lineax`` solver.

    Returns:
        An ``optax.GradientTransformation``.
    """
    return _wrap_for_loss(
        blr_low_rank(
            learning_rate=learning_rate,
            rank=rank,
            prior_precision=prior_precision,
            prior_mean=prior_mean,
            hessian_estimator=_negate_hessian_for_loss(hessian_estimator),
            damping=damping,
            solver=solver,
        )
    )
