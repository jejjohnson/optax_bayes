"""Monte Carlo sampling from BLR posterior state.

Provides reparameterised posterior samples for MC estimation:
  theta_hat = m + L * eps,  eps ~ N(0, I)

where L is the Cholesky-like square root of the covariance.
For diagonal: L = diag(1/sqrt(s)).
For full-rank: L = cholesky(Lambda^{-1}).
For low-rank: L via Woodbury.
"""

from __future__ import annotations

import gaussx
import jax
import jax.numpy as jnp
import lineax as lx
import optax

from optax_bayes._src.low_rank import _build_low_rank_operator
from optax_bayes._src.types import BLRDiagState, BLRFullRankState, BLRLowRankState


def sample_posterior_diagonal(
    state: BLRDiagState,
    key: jax.Array,
) -> optax.Params:
    """Draw one reparameterised sample from the diagonal posterior.

    Samples theta ~ N(m, diag(1/s)) via:
      theta = m + eps / sqrt(s),  eps ~ N(0, I)

    Args:
        state: A ``BLRDiagState``.
        key: PRNG key.

    Returns:
        Sampled parameter pytree.
    """
    mean = jax.tree.map(lambda eta, s: eta / s, state.nat_mean, state.precision)
    std = jax.tree.map(lambda s: 1.0 / jnp.sqrt(s), state.precision)

    leaves, treedef = jax.tree.flatten(mean)
    keys = jax.random.split(key, len(leaves))
    eps_leaves = [
        jax.random.normal(k, leaf.shape, leaf.dtype)
        for k, leaf in zip(keys, leaves, strict=True)
    ]
    eps = treedef.unflatten(eps_leaves)

    return jax.tree.map(lambda m, s, e: m + s * e, mean, std, eps)


def sample_posterior_full_rank(
    state: BLRFullRankState,
    key: jax.Array,
) -> jnp.ndarray:
    """Draw one reparameterised sample from the full-rank posterior.

    Samples theta ~ N(m, Lambda^{-1}) via Cholesky:
      theta = m + L^{-T} eps,  eps ~ N(0, I)

    where L L^T = Lambda (Cholesky of precision).

    Args:
        state: A ``BLRFullRankState``.
        key: PRNG key.

    Returns:
        Sampled parameter vector, shape (d,).
    """
    op: lx.AbstractLinearOperator = lx.MatrixLinearOperator(state.precision)  # ty: ignore[invalid-assignment]
    mean = gaussx.solve(op, state.nat_mean)

    # Cholesky of precision: Lambda = L L^T
    chol = jnp.linalg.cholesky(state.precision)

    d = mean.shape[0]
    eps = jax.random.normal(key, (d,), mean.dtype)

    # Sample: theta = m + L^{-T} eps
    # L^{-T} eps = solve(L^T, eps)
    sample = mean + jax.scipy.linalg.solve_triangular(chol.T, eps, lower=False)
    return sample


def sample_posterior_low_rank(
    state: BLRLowRankState,
    key: jax.Array,
) -> jnp.ndarray:
    """Draw one reparameterised sample from the low-rank posterior.

    Samples theta ~ N(m, Lambda^{-1}) where Lambda = diag(D) + U U^T.

    Uses a two-step Gaussian trick that avoids forming the dense
    covariance matrix:

    1. Draw z ~ N(0, Lambda) cheaply via
       ``z = sqrt(D) * eps_1 + U @ eps_2``.
    2. Solve ``Lambda y = z`` via Woodbury; then y ~ N(0, Lambda^{-1}).

    Cost: O(d r + r^3) memory / O(d r^2 + r^3) time.

    Args:
        state: A ``BLRLowRankState``.
        key: PRNG key.

    Returns:
        Sampled parameter vector, shape (d,).
    """
    op = _build_low_rank_operator(state.diag_precision, state.low_rank_factor)
    mean = gaussx.solve(op, state.nat_mean)

    d = state.diag_precision.shape[0]
    r = state.low_rank_factor.shape[1]

    k1, k2 = jax.random.split(key)
    eps_d = jax.random.normal(k1, (d,), mean.dtype)
    eps_r = jax.random.normal(k2, (r,), mean.dtype)

    # z ~ N(0, Lambda) via the factorisation
    z = jnp.sqrt(state.diag_precision) * eps_d + state.low_rank_factor @ eps_r

    # y = Lambda^{-1} z ~ N(0, Lambda^{-1}) via Woodbury (gaussx)
    y = gaussx.solve(op, z)

    return mean + y
