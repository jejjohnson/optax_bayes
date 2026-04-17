"""BLR with optax learning rate schedules."""

from __future__ import annotations

import optax

from optax_bayes._src.wrappers import blr_diagonal_for_loss


def blr_with_schedule(
    schedule_fn: optax.Schedule,
    prior_precision: float = 1e-4,
    prior_mean: float = 0.0,
    hessian_estimator: str = "ggn_diag",
    damping: float = 1e-6,
) -> optax.GradientTransformation:
    """Diagonal BLR with a time-varying learning rate.

    Wraps ``blr_diagonal_for_loss`` with ``optax.inject_hyperparams``
    so that the step size ``rho`` follows an optax schedule.

    Args:
        schedule_fn: An ``optax.Schedule`` (e.g.
            ``optax.warmup_cosine_decay_schedule(...)``).
        prior_precision: Scalar prior precision s0 = 1/v0.
        prior_mean: Scalar prior mean m0.
        hessian_estimator: One of ``"ggn_diag"`` or ``"identity"``.
        damping: Minimum precision after update.

    Returns:
        An ``optax.GradientTransformation`` with scheduled learning rate.
    """
    return optax.inject_hyperparams(
        lambda learning_rate: blr_diagonal_for_loss(
            learning_rate=learning_rate,
            prior_precision=prior_precision,
            prior_mean=prior_mean,
            hessian_estimator=hessian_estimator,
            damping=damping,
        )
    )(learning_rate=schedule_fn)
