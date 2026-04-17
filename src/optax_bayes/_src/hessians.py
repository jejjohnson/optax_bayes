"""Hessian estimators for BLR.

Diagonal estimators:
- ``ggn_diag``: GGN diagonal approximation, H = -g^2 (always NSD).
- ``identity_hessian``: H = 0 (fixes precision to prior).

Full-rank estimators:
- ``ggn_outer``: Rank-1 outer product, H = -g g^T (always NSD).
- ``identity_hessian_full``: H = 0 (d x d zero matrix).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def ggn_diag(grads: Float[Array, ...]) -> Float[Array, ...]:
    """GGN diagonal Hessian approximation.

    Returns ``-(grads ** 2)`` elementwise.  Always non-positive, so
    ``s0 - h`` increases precision rather than making it indefinite.

    Args:
        grads: Log-likelihood gradient leaf array.

    Returns:
        Diagonal Hessian estimate, same shape as *grads*.
    """
    return -(grads**2)


def identity_hessian(grads: Float[Array, ...]) -> Float[Array, ...]:
    """Identity (zero) Hessian estimator.

    Returns zeros matching *grads*.  Fixes precision to the prior,
    reducing BLR to preconditioned SGD.

    Args:
        grads: Log-likelihood gradient leaf array (used only for shape/dtype).

    Returns:
        Zero array, same shape as *grads*.
    """
    return jnp.zeros_like(grads)


def ggn_diag_tree(grads: dict) -> dict:
    """PyTree-aware GGN diagonal estimator."""
    return jax.tree.map(ggn_diag, grads)


def identity_hessian_tree(grads: dict) -> dict:
    """PyTree-aware identity Hessian estimator."""
    return jax.tree.map(identity_hessian, grads)


# ── Full-rank estimators ────────────────────────────────────────


def ggn_outer(
    grads: Float[Array, ...],
) -> Float[Array, ...]:
    """GGN rank-1 outer-product Hessian approximation.

    Returns ``-g g^T``.  Always negative semi-definite (rank 1).

    Args:
        grads: Log-likelihood gradient vector, shape (d,).

    Returns:
        Hessian estimate, shape (d, d).
    """
    return -jnp.outer(grads, grads)


def identity_hessian_full(
    grads: Float[Array, ...],
) -> Float[Array, ...]:
    """Zero Hessian for full-rank BLR.

    Returns a (d, d) zero matrix.

    Args:
        grads: Log-likelihood gradient vector (used only for shape/dtype).

    Returns:
        Zero matrix, shape (d, d).
    """
    d = grads.shape[0]
    return jnp.zeros((d, d), dtype=grads.dtype)


def resolve_hessian_estimator_full(hessian_estimator):
    """Resolve a full-rank hessian_estimator argument (Option C).

    Accepts a string selector or a callable:
    - ``"ggn"``: rank-1 outer product ``-g g^T``
    - ``"identity"``: zero matrix
    - callable: ``fn(mean, grads) -> (d, d)`` Hessian matrix

    Args:
        hessian_estimator: String or callable.

    Returns:
        A callable ``(mean, grads) -> (d, d)``.

    Raises:
        ValueError: If string is not recognised.
        TypeError: If not a string or callable.
    """
    if callable(hessian_estimator):
        return hessian_estimator
    if hessian_estimator == "ggn":
        return lambda mean, grads: ggn_outer(grads)
    if hessian_estimator == "identity":
        return lambda mean, grads: identity_hessian_full(grads)
    raise ValueError(
        f"Unknown hessian_estimator: {hessian_estimator!r}. "
        "Supported strings: 'ggn', 'identity', or pass a callable."
    )
