"""State containers for BLR transforms."""

from __future__ import annotations

from typing import NamedTuple

import jax
import optax


class BLRDiagState(NamedTuple):
    """Natural-parameter state for q(theta) = N(m, diag(1/s)).

    Attributes:
        precision: Precision pytree, s = 1/v. Mirrors the parameter tree.
        nat_mean: Natural mean pytree, eta = s * m. Mirrors the parameter tree.
        count: Scalar step counter.
    """

    precision: optax.Params
    nat_mean: optax.Params
    count: jax.Array


class BLRFullRankState(NamedTuple):
    """Natural-parameter state for q(theta) = N(m, Lambda^{-1}), theta in R^d.

    Attributes:
        precision: Precision matrix Lambda, shape (d, d).
        nat_mean: Natural mean eta = Lambda @ m, shape (d,).
        count: Scalar step counter.
    """

    precision: jax.Array  # (d, d)
    nat_mean: jax.Array  # (d,)
    count: jax.Array


class BLRLowRankState(NamedTuple):
    """Low-rank natural-parameter state: Lambda = diag(D) + U U^T.

    Attributes:
        diag_precision: Diagonal precision D, shape (d,).
        low_rank_factor: Low-rank factor U, shape (d, r).
        nat_mean: Natural mean eta = Lambda @ m, shape (d,).
        count: Scalar step counter.
    """

    diag_precision: jax.Array  # (d,)
    low_rank_factor: jax.Array  # (d, r)
    nat_mean: jax.Array  # (d,)
    count: jax.Array
