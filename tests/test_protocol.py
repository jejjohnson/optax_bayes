"""Optax protocol guarantees: jit, scan, chaining, and pytree state.

These tests pin down the promise that every transform in the package
composes like a normal optax ``GradientTransformation``: state is a
pytree, updates work under ``jax.jit`` and ``jax.lax.scan``, and the
transforms chain with stock optax combinators.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest

import optax_bayes


def _quadratic_grads(params):
    """Gradient of the loss 0.5 * ||params||^2."""
    return params


class TestJitUpdate:
    """opt.update must be jittable for every transform."""

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: optax_bayes.blr_diagonal_for_loss(learning_rate=0.1),
            lambda: optax_bayes.blr_full_rank_for_loss(learning_rate=0.1),
            lambda: optax_bayes.blr_low_rank_for_loss(learning_rate=0.1, rank=2),
            lambda: optax_bayes.ivon(learning_rate=0.1),
        ],
        ids=["diagonal", "full_rank", "low_rank", "ivon"],
    )
    def test_update_under_jit(self, factory):
        params = jnp.array([1.0, -2.0, 0.5, 3.0])
        opt = factory()
        state = opt.init(params)

        @jax.jit
        def step(grads, state, params):
            return opt.update(grads, state, params)

        updates, _ = step(_quadratic_grads(params), state, params)
        assert updates.shape == params.shape
        assert jnp.all(jnp.isfinite(updates))


class TestScanTrainingLoop:
    """Whole training loops must run under lax.scan (single compile)."""

    def test_diagonal_pytree_params(self):
        params = {"w": jnp.array([1.0, -2.0]), "b": jnp.array(0.5)}
        opt = optax_bayes.blr_diagonal_for_loss(learning_rate=0.2)
        state = opt.init(params)

        def step(carry, _):
            params, state = carry
            grads = jax.tree.map(lambda p: p, params)
            updates, state = opt.update(grads, state)
            params = optax.apply_updates(params, updates)
            return (params, state), None

        (final_params, final_state), _ = jax.lax.scan(
            step, (params, state), None, length=50
        )
        assert jnp.abs(final_params["b"]) < jnp.abs(params["b"])
        assert int(final_state.count) == 50

    def test_full_rank_scan(self):
        # Exact loss Hessian (identity for 0.5*||x||^2) and an informative
        # prior: the rank-1 GGN default is not stable on this toy problem.
        params = jnp.array([2.0, -1.0, 0.5])
        opt = optax_bayes.blr_full_rank_for_loss(
            learning_rate=0.3,
            prior_precision=1.0,
            hessian_estimator=lambda mean, grads: jnp.eye(3),
        )
        state = opt.init(params)

        def step(carry, _):
            params, state = carry
            updates, state = opt.update(_quadratic_grads(params), state)
            params = optax.apply_updates(params, updates)
            return (params, state), None

        (final_params, _), _ = jax.lax.scan(step, (params, state), None, length=50)
        assert jnp.linalg.norm(final_params) < jnp.linalg.norm(params)

    def test_low_rank_scan(self):
        params = jnp.array([2.0, -1.0, 0.5, 1.5])
        opt = optax_bayes.blr_low_rank_for_loss(learning_rate=0.3, rank=2)
        state = opt.init(params)

        def step(carry, _):
            params, state = carry
            updates, state = opt.update(_quadratic_grads(params), state)
            params = optax.apply_updates(params, updates)
            return (params, state), None

        (final_params, _), _ = jax.lax.scan(step, (params, state), None, length=50)
        assert jnp.linalg.norm(final_params) < jnp.linalg.norm(params)


class TestChaining:
    """Transforms must compose with stock optax combinators."""

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: optax_bayes.blr_diagonal_for_loss(learning_rate=0.1),
            lambda: optax_bayes.blr_full_rank_for_loss(learning_rate=0.1),
            lambda: optax_bayes.blr_low_rank_for_loss(learning_rate=0.1, rank=2),
        ],
        ids=["diagonal", "full_rank", "low_rank"],
    )
    def test_chain_with_clip(self, factory):
        params = jnp.array([10.0, -20.0, 5.0])
        opt = optax.chain(optax.clip_by_global_norm(1.0), factory())
        state = opt.init(params)
        updates, state = opt.update(_quadratic_grads(params), state, params)
        params = optax.apply_updates(params, updates)
        assert jnp.all(jnp.isfinite(params))

    def test_schedule_wrapper_steps(self):
        schedule = optax.linear_schedule(0.3, 0.01, transition_steps=20)
        opt = optax_bayes.blr_with_schedule(schedule)
        params = jnp.array([1.0, -1.0])
        state = opt.init(params)
        for _ in range(5):
            updates, state = opt.update(_quadratic_grads(params), state)
            params = optax.apply_updates(params, updates)
        assert jnp.all(jnp.isfinite(params))
        assert float(state.hyperparams["learning_rate"]) < 0.3


class TestStateIsPytree:
    """Optimizer state must round-trip through pytree flatten/unflatten,
    which is what checkpointing libraries rely on."""

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: optax_bayes.blr_diagonal(learning_rate=0.1),
            lambda: optax_bayes.blr_full_rank(learning_rate=0.1),
            lambda: optax_bayes.blr_low_rank(learning_rate=0.1, rank=2),
            lambda: optax_bayes.ivon(learning_rate=0.1),
        ],
        ids=["diagonal", "full_rank", "low_rank", "ivon"],
    )
    def test_flatten_roundtrip(self, factory):
        params = jnp.array([1.0, -2.0, 0.5])
        opt = factory()
        state = opt.init(params)

        leaves, treedef = jax.tree.flatten(state)
        restored = jax.tree.unflatten(treedef, leaves)

        u1, _ = opt.update(params, state)
        u2, _ = opt.update(params, restored)
        assert jnp.allclose(u1, u2)
