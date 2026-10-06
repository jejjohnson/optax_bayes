"""Transforms keep the params' dtype, whatever JAX's default float is."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import optax_bayes as ob


FLAT = jnp.ones(4, jnp.float32)
FLAT_GRADS = jnp.arange(4.0, dtype=jnp.float32)
# The widest float available: float64 with x64 on, float32 in the no-x64 lane.
WIDE = jax.dtypes.canonicalize_dtype(jnp.float64)
TREE = {"w": jnp.ones((2, 2), jnp.float32), "b": jnp.ones(3, jnp.float32)}
TREE_GRADS = jax.tree.map(jnp.ones_like, TREE)

TRANSFORMS = [
    pytest.param(ob.blr_diagonal(), TREE, TREE_GRADS, id="blr_diagonal"),
    pytest.param(ob.ivon(), TREE, TREE_GRADS, id="ivon"),
    # A float64 prior mean and Hessian must not promote the state either.
    pytest.param(
        ob.blr_full_rank(
            prior_mean=jnp.zeros(4, WIDE),
            hessian_estimator=lambda mean, grads: -jnp.eye(4, dtype=WIDE),
        ),
        FLAT,
        FLAT_GRADS,
        id="blr_full_rank",
    ),
    pytest.param(
        ob.blr_low_rank(rank=2, prior_mean=jnp.zeros(4, WIDE)),
        FLAT,
        FLAT_GRADS,
        id="blr_low_rank",
    ),
    pytest.param(
        ob.newton(lambda mean, grads: -jnp.eye(4, dtype=WIDE)),
        FLAT,
        FLAT_GRADS,
        id="newton",
    ),
]


def _float_dtypes(tree):
    return {
        leaf.dtype
        for leaf in jax.tree.leaves(tree)
        if jnp.issubdtype(leaf.dtype, jnp.floating)
    }


@pytest.mark.parametrize(("opt", "params", "grads"), TRANSFORMS)
def test_float32_params_stay_float32(opt, params, grads):
    state = opt.init(params)
    updates, new_state = opt.update(grads, state, params)
    assert _float_dtypes(state) == {jnp.dtype(jnp.float32)}
    assert _float_dtypes(updates) == {jnp.dtype(jnp.float32)}
    assert _float_dtypes(new_state) == {jnp.dtype(jnp.float32)}


def test_full_rank_helpers_stay_float32():
    opt = ob.blr_full_rank()
    _, state = opt.update(FLAT_GRADS, opt.init(FLAT), FLAT)
    mean, cov = ob.get_posterior_full_rank(state)
    sample = ob.sample_posterior_full_rank(state, jax.random.key(0))
    assert {mean.dtype, cov.dtype, sample.dtype} == {jnp.dtype(jnp.float32)}


def test_low_rank_helpers_stay_float32():
    opt = ob.blr_low_rank(rank=2)
    _, state = opt.update(FLAT_GRADS, opt.init(FLAT), FLAT)
    mean, cov = ob.get_posterior_low_rank(state)
    sample = ob.sample_posterior_low_rank(state, jax.random.key(0))
    assert {mean.dtype, cov.dtype, sample.dtype} == {jnp.dtype(jnp.float32)}
