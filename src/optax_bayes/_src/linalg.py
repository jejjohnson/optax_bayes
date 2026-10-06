"""Precision-matrix linear algebra for the gaussx-backed transforms.

Every BLR precision is positive definite (a positive prior plus a
negative semi-definite Hessian estimate, plus damping), so the operators
built here carry ``lx.positive_semidefinite_tag``.  That routes dense
solves to Cholesky instead of LU, and lets a user-supplied
``lx.Cholesky()`` accept the operator at all.

The ``solver`` argument accepts either a ``lineax`` solver (forwarded to
``gaussx.solve`` / ``gaussx.inv``) or a ``gaussx`` solver strategy such as
``gaussx.DenseSolver()`` or ``gaussx.CGSolver()`` (called directly).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import lineax as lx

from optax_bayes._src._optional import require_gaussx


if TYPE_CHECKING:
    import gaussx

    type Solver = lx.AbstractLinearSolver | gaussx.AbstractSolverStrategy


def precision_operator(
    precision: jnp.ndarray, *, psd: bool = True
) -> lx.AbstractLinearOperator:
    """Wrap a dense (d, d) precision matrix as a lineax operator.

    Args:
        precision: The precision matrix.
        psd: Tag it positive semi-definite (Cholesky solves). Pass
            ``False`` only when it may be indefinite, as in ``newton``.

    Returns:
        A ``lx.MatrixLinearOperator``.
    """
    tags = lx.positive_semidefinite_tag if psd else ()
    return lx.MatrixLinearOperator(precision, tags)


def low_rank_precision_operator(
    d_diag: jnp.ndarray, u: jnp.ndarray
) -> lx.AbstractLinearOperator:
    """Build the PSD ``gaussx.LowRankUpdate`` operator diag(D) + U U^T.

    Args:
        d_diag: Diagonal entries, shape (d,), kept positive by damping.
        u: Low-rank factor, shape (d, r).

    Returns:
        A ``gaussx.LowRankUpdate`` operator.
    """
    gaussx = require_gaussx("low-rank BLR operators")
    return gaussx.low_rank_plus_diag(d_diag, u, psd=True)


def solve(
    op: lx.AbstractLinearOperator,
    b: jnp.ndarray,
    solver: Solver | None = None,
) -> jnp.ndarray:
    """Solve ``op @ x = b`` with a lineax solver or a gaussx strategy.

    Args:
        op: Precision operator.
        b: Right-hand side, shape (d,).
        solver: ``None`` (gaussx structural dispatch), a ``lineax`` solver,
            or a ``gaussx`` solver strategy.

    Returns:
        Solution x, shape (d,).
    """
    gaussx = require_gaussx("BLR precision solves")
    if solver is None or isinstance(solver, lx.AbstractLinearSolver):
        return gaussx.solve(op, b, solver=solver)
    return solver.solve(op, b)


def inverse_matrix(
    op: lx.AbstractLinearOperator,
    solver: Solver | None = None,
) -> jnp.ndarray:
    """Materialise ``op^{-1}`` as a dense (d, d) matrix.

    Args:
        op: Precision operator.
        solver: As for :func:`solve`.

    Returns:
        The dense inverse (the covariance), shape (d, d).
    """
    gaussx = require_gaussx("BLR posterior covariances")
    if solver is None or isinstance(solver, lx.AbstractLinearSolver):
        return gaussx.inv(op, solver=solver).as_matrix()
    eye = jnp.eye(op.in_size(), dtype=op.in_structure().dtype)
    return jax.vmap(lambda col: solver.solve(op, col), out_axes=1)(eye)
