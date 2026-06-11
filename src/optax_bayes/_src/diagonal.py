"""Diagonal BLR as an optax GradientTransformation.

Expects log-likelihood gradients (not loss gradients).  Most users should
use ``blr_diagonal_for_loss`` from the wrappers module instead.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from optax_bayes._src.hessians import ggn_diag, identity_hessian
from optax_bayes._src.types import BLRDiagState


def blr_diagonal(
    learning_rate: float = 1e-2,
    prior_precision: float = 1e-4,
    prior_mean: float = 0.0,
    hessian_estimator: str = "ggn_diag",
    damping: float = 1e-6,
) -> optax.GradientTransformation:
    """Diagonal-Gaussian BLR as an optax transform.

    Implements the Bayesian Learning Rule (Khan & Rue, 2023) with a
    diagonal Gaussian variational family.  Optimizer state is stored
    in natural-parameter form.

    **This API expects log-likelihood gradients.**  For standard loss
    minimisation, use ``blr_diagonal_for_loss`` instead.

    Args:
        learning_rate: Step size rho in (0, 1].
        prior_precision: Scalar prior precision s0 = 1/v0, broadcast
            across all parameters.
        prior_mean: Scalar prior mean m0, broadcast across all parameters.
        hessian_estimator: One of ``"ggn_diag"`` or ``"identity"``.
        damping: Minimum precision after update (numerical safeguard).

    Returns:
        An ``optax.GradientTransformation``.

    Raises:
        ValueError: If *hessian_estimator* is not a recognised string.
    """
    if hessian_estimator not in ("ggn_diag", "identity"):
        raise ValueError(
            f"Unknown hessian_estimator: {hessian_estimator!r}. "
            "Wave 1 supports 'ggn_diag' and 'identity'."
        )

    _hessian_fn = ggn_diag if hessian_estimator == "ggn_diag" else identity_hessian

    def init_fn(params: optax.Params) -> BLRDiagState:
        # The variational mean starts at the user's params (standard optax
        # drop-in semantics): eta_0 = s_0 * params, so m_0 = params.  The
        # prior mean still anchors every update through eta0 below.
        precision = jax.tree.map(lambda p: jnp.full_like(p, prior_precision), params)
        nat_mean = jax.tree.map(lambda p: prior_precision * p, params)
        return BLRDiagState(
            precision=precision,
            nat_mean=nat_mean,
            count=jnp.zeros([], jnp.int32),
        )

    def update_fn(
        grads: optax.Updates,
        state: BLRDiagState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, BLRDiagState]:
        s0 = prior_precision
        eta0 = prior_precision * prior_mean
        rho = learning_rate

        # Current mean: m_t = eta_t / s_t
        m_t = jax.tree.map(lambda eta, s: eta / s, state.nat_mean, state.precision)

        # Diagonal Hessian estimate (h <= 0 for concave log-likelihood)
        diag_h = jax.tree.map(_hessian_fn, grads)

        # Precision update: s_next = (1 - rho) * s + rho * (s0 - h)
        new_precision = jax.tree.map(
            lambda s, h: jnp.maximum((1 - rho) * s + rho * (s0 - h), damping),
            state.precision,
            diag_h,
        )

        # Natural mean update: eta_next = (1-rho)*eta + rho*(eta0 + g - h*m)
        new_nat_mean = jax.tree.map(
            lambda eta, g, h, m: (1 - rho) * eta + rho * (eta0 + g - h * m),
            state.nat_mean,
            grads,
            diag_h,
            m_t,
        )

        # Recover mean and compute Optax update
        new_mean = jax.tree.map(lambda eta, s: eta / s, new_nat_mean, new_precision)
        updates = jax.tree.map(lambda mn, mo: mn - mo, new_mean, m_t)

        new_state = BLRDiagState(
            precision=new_precision,
            nat_mean=new_nat_mean,
            count=state.count + 1,
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)  # ty: ignore[invalid-argument-type]
