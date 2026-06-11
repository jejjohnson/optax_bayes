"""Posterior extraction from BLR optimizer state."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import optax

from optax_bayes._src._optional import require_gaussx
from optax_bayes._src.low_rank import _build_low_rank_operator
from optax_bayes._src.types import BLRDiagState, BLRFullRankState, BLRLowRankState


def get_posterior_diagonal(
    state: BLRDiagState,
) -> tuple[optax.Params, optax.Params]:
    r"""Extract the approximate posterior from diagonal BLR state.

    Returns the mean and variance of
    $q(\theta) = \mathcal{N}(m, \operatorname{diag}(v))$ where

    $$
    m = \eta / s, \qquad v = 1 / s.
    $$

    Args:
        state: A ``BLRDiagState`` from ``blr_diagonal`` or
            ``blr_diagonal_for_loss``.

    Returns:
        Tuple ``(mean, variance)`` as pytrees matching the parameter tree.
    """
    mean = jax.tree.map(lambda eta, s: eta / s, state.nat_mean, state.precision)
    variance = jax.tree.map(lambda s: 1.0 / s, state.precision)
    return mean, variance


def get_posterior_full_rank(
    state: BLRFullRankState,
    solver: lx.AbstractLinearSolver | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Extract the approximate posterior from full-rank BLR state.

    Returns the mean and covariance of
    $q(\theta) = \mathcal{N}(\Lambda^{-1}\eta,\ \Lambda^{-1})$,
    using ``gaussx.solve`` for the mean and ``gaussx.inv`` for the
    covariance (requires the optional ``gaussx`` extra).

    Args:
        state: A ``BLRFullRankState`` from ``blr_full_rank`` or
            ``blr_full_rank_for_loss``.
        solver: Optional ``lineax`` solver (e.g. ``lx.AutoLinearSolver()``).

    Returns:
        Tuple ``(mean, covariance)`` where mean is (d,) and
        covariance is (d, d).
    """
    gaussx = require_gaussx("get_posterior_full_rank")
    op: lx.AbstractLinearOperator = lx.MatrixLinearOperator(state.precision)
    mean = gaussx.solve(op, state.nat_mean, solver=solver)
    inv_op = gaussx.inv(op, solver=solver)
    covariance = inv_op.as_matrix()
    return mean, covariance


def get_posterior_low_rank(
    state: BLRLowRankState,
    solver: lx.AbstractLinearSolver | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Extract the approximate posterior from low-rank BLR state.

    Returns the mean and covariance of
    $q(\theta) = \mathcal{N}(\Lambda^{-1}\eta,\ \Lambda^{-1})$ with
    $\Lambda = \operatorname{diag}(D) + U U^\top$, using ``gaussx``
    structured operators for the mean solve and ``gaussx.inv`` for the
    covariance (Woodbury-dispatched; requires the optional ``gaussx``
    extra).

    Args:
        state: A ``BLRLowRankState`` from ``blr_low_rank`` or
            ``blr_low_rank_for_loss``.
        solver: Optional ``gaussx`` solver strategy.

    Returns:
        Tuple ``(mean, covariance)`` where mean is (d,) and
        covariance is (d, d).
    """
    gaussx = require_gaussx("get_posterior_low_rank")
    op = _build_low_rank_operator(state.diag_precision, state.low_rank_factor)
    mean = gaussx.solve(op, state.nat_mean, solver=solver)
    inv_op = gaussx.inv(op, solver=solver)
    covariance = inv_op.as_matrix()
    return mean, covariance
