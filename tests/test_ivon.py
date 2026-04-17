"""Tests for IVON optimizer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

import optax_bayes
from optax_bayes import IVONState, get_posterior_ivon, ivon, sample_ivon


class TestIVONInit:
    def test_shapes(self):
        params = {"w": jnp.zeros((3, 4)), "b": jnp.zeros(4)}
        opt = ivon()
        state = opt.init(params)
        assert isinstance(state, IVONState)
        assert state.momentum["w"].shape == (3, 4)
        assert state.hess["b"].shape == (4,)
        assert state.count == 0

    def test_hess_init(self):
        params = jnp.zeros(5)
        opt = ivon(hess_init=2.0)
        state = opt.init(params)
        assert jnp.allclose(state.hess, jnp.full(5, 2.0))


class TestIVONQuadratic:
    def test_converges(self):
        d = 5
        theta_star = jnp.ones(d) * 2.0

        def loss_fn(theta):
            return 0.5 * jnp.sum((theta - theta_star) ** 2)

        opt = ivon(learning_rate=0.1, weight_decay=1e-4, hess_init=0.1)
        theta = jnp.zeros(d)
        state = opt.init(theta)
        initial_loss = loss_fn(theta)

        @jax.jit
        def step(carry, _):
            theta, state = carry
            g = jax.grad(loss_fn)(theta)
            updates, state = opt.update(g, state, params=theta)
            theta = optax.apply_updates(theta, updates)
            return (theta, state), None

        (theta, state), _ = jax.lax.scan(step, (theta, state), None, length=50)

        assert loss_fn(theta) < initial_loss * 0.01
        assert jnp.all(jnp.isfinite(theta))

    def test_without_params(self):
        """IVON should work without passing params (no weight decay)."""
        d = 3
        theta_star = jnp.ones(d)

        def loss_fn(theta):
            return 0.5 * jnp.sum((theta - theta_star) ** 2)

        opt = ivon(learning_rate=0.1, hess_init=0.1)
        theta = jnp.zeros(d)
        state = opt.init(theta)

        @jax.jit
        def step(carry, _):
            theta, state = carry
            g = jax.grad(loss_fn)(theta)
            updates, state = opt.update(g, state)
            theta = optax.apply_updates(theta, updates)
            return (theta, state), None

        (theta, state), _ = jax.lax.scan(step, (theta, state), None, length=50)

        assert jnp.all(jnp.isfinite(theta))


class TestIVONSampling:
    def test_sample_shape(self):
        params = {"w": jnp.ones((3, 2)), "b": jnp.ones(2)}
        opt = ivon(hess_init=1.0)
        state = opt.init(params)
        key = jax.random.PRNGKey(42)
        sample = sample_ivon(state, params, key, weight_decay=1e-4)
        assert sample["w"].shape == (3, 2)
        assert sample["b"].shape == (2,)

    def test_samples_differ(self):
        params = jnp.ones(5)
        opt = ivon(hess_init=1.0)
        state = opt.init(params)
        s1 = sample_ivon(state, params, jax.random.PRNGKey(0))
        s2 = sample_ivon(state, params, jax.random.PRNGKey(1))
        assert not jnp.allclose(s1, s2)


class TestIVONPosterior:
    def test_posterior_values(self):
        params = jnp.array([1.0, 2.0, 3.0])
        opt = ivon(hess_init=4.0, weight_decay=1.0)
        state = opt.init(params)
        m, v = get_posterior_ivon(state, params, weight_decay=1.0)
        # variance = 1 / (hess + wd) = 1 / (4 + 1) = 0.2
        assert jnp.allclose(m, params)
        assert jnp.allclose(v, jnp.full(3, 0.2))


class TestIVONImports:
    def test_public_symbols(self):
        assert hasattr(optax_bayes, "ivon")
        assert hasattr(optax_bayes, "IVONState")
        assert hasattr(optax_bayes, "sample_ivon")
        assert hasattr(optax_bayes, "get_posterior_ivon")
