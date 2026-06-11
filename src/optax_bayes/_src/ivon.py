"""IVON: Improved Variational Online Newton (Shen et al., 2024).

A practical diagonal BLR variant for deep learning with:
- Separate EMA rates for mean (beta1) and precision (beta2)
- Explicit weight decay (prior precision)
- Gradient clipping in natural-parameter space
- MC sampling mode for proper Bayesian inference

Accepts standard loss gradients (like ``optax.adam``).

Reference: Shen et al. (2024) "Variational Learning is Effective for
Large Deep Networks." ICML.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax


class IVONState(NamedTuple):
    """State for the IVON optimizer.

    Attributes:
        momentum: EMA of gradients (first moment), pytree.
        hess: Diagonal Hessian estimate (precision - weight_decay),
            pytree.
        count: Scalar step counter.
    """

    momentum: optax.Params
    hess: optax.Params
    count: jax.Array


def ivon(
    learning_rate: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.99999,
    weight_decay: float = 1e-4,
    hess_init: float = 1.0,
    clip_radius: float = float("inf"),
    ess: float = 1.0,
) -> optax.GradientTransformation:
    """IVON optimizer (Shen et al., 2024).

    A diagonal BLR variant designed for deep learning.  Accepts
    standard **loss** gradients (same convention as ``optax.adam``).

    The optimizer maintains a diagonal precision estimate and a
    momentum buffer.  Use ``sample_ivon`` to draw reparameterised
    posterior samples before computing gradients for proper Bayesian
    inference.

    Args:
        learning_rate: Step size alpha.
        beta1: EMA rate for gradient momentum.
        beta2: EMA rate for Hessian (precision) estimate.
        weight_decay: Prior precision / L2 regularisation strength.
            Acts as both the Bayesian prior and explicit weight decay.
        hess_init: Initial diagonal Hessian value.
        clip_radius: Maximum magnitude of the natural-gradient update
            per parameter.
        ess: Effective sample size scaling for the Hessian.  Set to
            the training set size N for calibrated uncertainty;
            set to 1.0 (default) for optimizer-only mode.

    Returns:
        An ``optax.GradientTransformation``.
    """

    def init_fn(params: optax.Params) -> IVONState:
        momentum = jax.tree.map(jnp.zeros_like, params)
        hess = jax.tree.map(lambda p: jnp.full_like(p, hess_init), params)
        return IVONState(
            momentum=momentum,
            hess=hess,
            count=jnp.zeros([], jnp.int32),
        )

    def update_fn(
        grads: optax.Updates,
        state: IVONState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, IVONState]:
        count = state.count + 1

        # 1. Update momentum: m = beta1 * m + (1 - beta1) * g
        new_momentum = jax.tree.map(
            lambda m, g: beta1 * m + (1 - beta1) * g,
            state.momentum,
            grads,
        )

        # 2. Update Hessian estimate (GGN diagonal: g^2 scaled by ess)
        #    h_new = beta2 * h + (1 - beta2) * ess * g^2
        new_hess = jax.tree.map(
            lambda h, g: beta2 * h + (1 - beta2) * ess * g**2,
            state.hess,
            grads,
        )

        # 3. Bias correction for momentum
        debias = 1.0 - beta1**count

        # 4. Compute clipped natural-gradient update
        #    update = -lr * clip((m/debias + wd*theta) / (h + wd), radius)
        #    where theta is the current params
        if params is None:
            # Without params, skip weight decay on the mean
            updates = jax.tree.map(
                lambda m, h: (
                    -learning_rate
                    * jnp.clip(
                        m / debias / (h + weight_decay),
                        -clip_radius,
                        clip_radius,
                    )
                ),
                new_momentum,
                new_hess,
            )
        else:
            updates = jax.tree.map(
                lambda m, h, p: (
                    -learning_rate
                    * jnp.clip(
                        (m / debias + weight_decay * p) / (h + weight_decay),
                        -clip_radius,
                        clip_radius,
                    )
                ),
                new_momentum,
                new_hess,
                params,
            )

        new_state = IVONState(
            momentum=new_momentum,
            hess=new_hess,
            count=count,
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)  # ty: ignore[invalid-argument-type]


def sample_ivon(
    state: IVONState,
    params: optax.Params,
    key: jax.Array,
    weight_decay: float = 1e-4,
) -> optax.Params:
    r"""Draw a reparameterised posterior sample from IVON state.

    Samples
    $\theta \sim \mathcal{N}(\text{params},\
    \operatorname{diag}(1 / (h + \lambda)))$ via

    $$
    \theta = \text{params} + \epsilon / \sqrt{h + \lambda},
    \qquad \epsilon \sim \mathcal{N}(0, I),
    $$

    where $h$ is the Hessian estimate and $\lambda$ the weight decay.
    Call this before computing gradients for proper Bayesian MC
    estimation.

    Args:
        state: An ``IVONState``.
        params: Current parameters (the posterior mean).
        key: PRNG key.
        weight_decay: Must match the ``weight_decay`` used in ``ivon()``.

    Returns:
        Sampled parameter pytree.
    """
    leaves, treedef = jax.tree.flatten(params)
    hess_leaves, _ = jax.tree.flatten(state.hess)
    keys = jax.random.split(key, len(leaves))

    sampled = []
    for k, p, h in zip(keys, leaves, hess_leaves, strict=True):
        eps = jax.random.normal(k, p.shape, p.dtype)
        sigma = 1.0 / jnp.sqrt(h + weight_decay)
        sampled.append(p + sigma * eps)

    return treedef.unflatten(sampled)


def get_posterior_ivon(
    state: IVONState,
    params: optax.Params,
    weight_decay: float = 1e-4,
) -> tuple[optax.Params, optax.Params]:
    r"""Extract the approximate posterior from IVON state.

    Returns ``(mean, variance)`` where the mean is the current iterate
    (``params``) and

    $$
    v = 1 / (h + \lambda)
    $$

    with $h$ the Hessian estimate and $\lambda$ the weight decay.

    Args:
        state: An ``IVONState``.
        params: Current parameters.
        weight_decay: Must match the ``weight_decay`` used in ``ivon()``.

    Returns:
        Tuple ``(mean, variance)`` as pytrees.
    """
    variance = jax.tree.map(
        lambda h: 1.0 / (h + weight_decay),
        state.hess,
    )
    return params, variance
