"""Hessian estimators for BLR.

Diagonal estimators:
- ``ggn_diag``: GGN diagonal approximation, H = -g^2 (always NSD).
- ``identity_hessian``: H = 0 (fixes precision to prior).

Full-rank estimators:
- ``ggn_outer``: Rank-1 outer product, H = -g g^T (always NSD).
- ``identity_hessian_full``: H = 0 (d x d zero matrix).

Low-rank estimators return a factor G (d, k) with -H = G G^T, so the
low-rank transform never forms a (d, d) matrix for the built-in choices:
- ``"ggn"``: G = g (k = 1).
- ``"identity"``: G is empty (k = 0).
- callable: the positive eigen-part of a dense -H (k = d, O(d^3)).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def ggn_diag(grads: Float[Array, ...]) -> Float[Array, ...]:
    r"""GGN diagonal Hessian approximation.

    Returns $h = -g^2$ elementwise. Always non-positive, so the BLR
    precision target $s_0 - h = s_0 + g^2$ increases precision rather
    than making it indefinite — this is why the estimator acts like an
    Adam-style curvature surrogate in the diagonal BLR path.

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
    r"""GGN rank-1 outer-product Hessian approximation.

    Returns $H = -g g^\top$. Always negative semi-definite (rank 1).

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


def resolve_hessian_estimator_full(
    hessian_estimator: str | Callable,
) -> Callable:
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
    if not isinstance(hessian_estimator, str):
        raise TypeError(
            "hessian_estimator must be a string or callable, "
            f"got {type(hessian_estimator).__name__}"
        )
    if hessian_estimator == "ggn":
        return lambda mean, grads: ggn_outer(grads)
    if hessian_estimator == "identity":
        return lambda mean, grads: identity_hessian_full(grads)
    raise ValueError(
        f"Unknown hessian_estimator: {hessian_estimator!r}. "
        "Supported strings: 'ggn', 'identity', or pass a callable."
    )


# ── Low-rank estimators (factor form) ───────────────────────────


def psd_factor(neg_hessian: Float[Array, ...]) -> Float[Array, ...]:
    r"""Factor the positive part of a symmetric matrix.

    Returns $G = V \sqrt{\max(\Lambda, 0)}$ from the eigendecomposition
    $-H = V \Lambda V^\top$, so $G G^\top$ is the closest PSD matrix to
    $-H$. Negative curvature (where $H$ is not NSD) is dropped, because it
    would make the precision indefinite.

    Args:
        neg_hessian: Symmetric matrix $-H$, shape (d, d).

    Returns:
        Factor $G$, shape (d, d).
    """
    eigvals, eigvecs = jnp.linalg.eigh(neg_hessian)
    return eigvecs * jnp.sqrt(jnp.maximum(eigvals, 0.0))[None, :]


def resolve_hessian_factor_low_rank(
    hessian_estimator: str | Callable,
) -> Callable:
    """Resolve a low-rank hessian_estimator argument to factor form.

    - ``"ggn"``: ``G = g[:, None]``, so ``-H = g g^T`` (rank 1).
    - ``"identity"``: an empty ``(d, 0)`` factor, ``H = 0``.
    - callable: ``fn(mean, grads) -> (d, d)`` Hessian, factored with
      :func:`psd_factor` (O(d^3); the built-in strings avoid it).

    Args:
        hessian_estimator: String or callable.

    Returns:
        A callable ``(mean, grads) -> G`` with ``G`` of shape (d, k).

    Raises:
        ValueError: If string is not recognised.
        TypeError: If not a string or callable.
    """
    if isinstance(hessian_estimator, str):
        if hessian_estimator == "ggn":
            return lambda mean, grads: grads[:, None]
        if hessian_estimator == "identity":
            return lambda mean, grads: jnp.zeros((grads.shape[0], 0), grads.dtype)
        raise ValueError(
            f"Unknown hessian_estimator: {hessian_estimator!r}. "
            "Supported strings: 'ggn', 'identity', or pass a callable."
        )
    if not callable(hessian_estimator):
        raise TypeError(
            "hessian_estimator must be a string or callable, "
            f"got {type(hessian_estimator).__name__}"
        )
    hessian_fn = hessian_estimator
    return lambda mean, grads: psd_factor(-hessian_fn(mean, grads))
