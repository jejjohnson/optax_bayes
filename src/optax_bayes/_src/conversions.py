"""Natural-parameter <-> mean-parameter conversions for diagonal Gaussians."""

from __future__ import annotations

import jax
import optax
from jaxtyping import Array, Float


def natural_to_mean_diag(
    eta: Float[Array, ...],
    s: Float[Array, ...],
) -> tuple[Float[Array, ...], Float[Array, ...]]:
    r"""Convert natural parameters to mean parameters (diagonal case).

    $$
    m = \eta / s, \qquad v = 1 / s.
    $$

    Args:
        eta: Natural mean, $\eta = s \odot m$.
        s: Precision (inverse variance), $s = 1 / v$.

    Returns:
        Tuple ``(m, v)``.
    """
    m = eta / s
    v = 1.0 / s
    return m, v


def mean_to_natural_diag(
    m: Float[Array, ...],
    v: Float[Array, ...],
) -> tuple[Float[Array, ...], Float[Array, ...]]:
    r"""Convert mean parameters to natural parameters (diagonal case).

    $$
    \eta = m / v, \qquad s = 1 / v.
    $$

    Args:
        m: Mean.
        v: Variance (must be positive).

    Returns:
        Tuple ``(eta, s)``.
    """
    s = 1.0 / v
    eta = m * s
    return eta, s


def natural_to_mean_diag_tree(
    eta: optax.Params,
    s: optax.Params,
) -> tuple[optax.Params, optax.Params]:
    """PyTree-aware natural-to-mean conversion.

    Args:
        eta: Natural mean pytree.
        s: Precision pytree.

    Returns:
        Tuple ``(mean_tree, variance_tree)``.
    """
    m = jax.tree.map(lambda e, p: e / p, eta, s)
    v = jax.tree.map(lambda p: 1.0 / p, s)
    return m, v
