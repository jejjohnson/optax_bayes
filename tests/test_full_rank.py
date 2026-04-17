"""Tests for full-rank BLR transform and public API."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest

import optax_bayes
from optax_bayes import (
    BLRFullRankState,
    blr_full_rank,
    blr_full_rank_for_loss,
    get_posterior_full_rank,
)


# ── State shape / structure ──────────────────────────────────────


class TestFullRankInit:
    def test_shapes(self):
        params = jnp.zeros(5)
        opt = blr_full_rank()
        state = opt.init(params)
        assert isinstance(state, BLRFullRankState)
        assert state.precision.shape == (5, 5)
        assert state.nat_mean.shape == (5,)
        assert state.count == 0

    def test_prior_precision_is_scaled_identity(self):
        params = jnp.zeros(3)
        opt = blr_full_rank(prior_precision=2.0)
        state = opt.init(params)
        assert jnp.allclose(state.precision, 2.0 * jnp.eye(3))

    def test_prior_mean(self):
        params = jnp.zeros(3)
        m0 = jnp.array([1.0, 2.0, 3.0])
        opt = blr_full_rank(prior_precision=1.0, prior_mean=m0)
        state = opt.init(params)
        # eta_0 = Lambda_0 @ m0 = I @ m0 = m0
        assert jnp.allclose(state.nat_mean, m0)


# ── Invalid estimator ────────────────────────────────────────────


class TestFullRankInvalidEstimator:
    def test_raises_on_unknown_string(self):
        with pytest.raises(ValueError, match="bad"):
            blr_full_rank(hessian_estimator="bad")

    def test_callable_accepted(self):
        # Should not raise
        opt = blr_full_rank(
            hessian_estimator=lambda mean, grads: -jnp.eye(mean.shape[0])
        )
        state = opt.init(jnp.zeros(3))
        assert state.precision.shape == (3, 3)


# ── Quadratic convergence ────────────────────────────────────────


class TestFullRankQuadratic:
    def test_exact_hessian_converges(self):
        """Full-rank BLR with exact Hessian on a quadratic.

        For L(theta) = 0.5 * ||theta - theta*||^2:
          H = -I (constant)
          Analytic posterior: m* = theta*/(1+s0), Sigma* = I/(1+s0)
        """
        d = 3
        theta_star = jnp.array([1.0, -2.0, 3.0])
        s0 = 1.0

        def loss_fn(theta):
            return 0.5 * jnp.sum((theta - theta_star) ** 2)

        # Exact Hessian: always -I for this quadratic
        opt = blr_full_rank_for_loss(
            learning_rate=0.1,
            prior_precision=s0,
            hessian_estimator=lambda m, g: -jnp.eye(d),
        )
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

        m, cov = get_posterior_full_rank(state)
        analytic_m = theta_star / (1 + s0)
        analytic_cov = jnp.eye(d) / (1 + s0)

        assert jnp.allclose(m, analytic_m, atol=0.1)
        assert jnp.allclose(cov, analytic_cov, atol=0.1)

    def test_ggn_loss_decreases(self):
        """GGN full-rank drives the loss down.

        The rank-1 outer product H = -g g^T couples dimensions,
        so we use a smaller learning rate to keep the update stable.
        """
        d = 3
        theta_star = jnp.ones(d) * 2.0

        def loss_fn(theta):
            return 0.5 * jnp.sum((theta - theta_star) ** 2)

        opt = blr_full_rank_for_loss(learning_rate=0.1, prior_precision=1e-2)
        theta = jnp.zeros(d)
        state = opt.init(theta)
        initial_loss = loss_fn(theta)

        @jax.jit
        def step(carry, _):
            theta, state = carry
            g = jax.grad(loss_fn)(theta)
            updates, state = opt.update(g, state)
            theta = optax.apply_updates(theta, updates)
            return (theta, state), None

        (theta, state), _ = jax.lax.scan(step, (theta, state), None, length=50)

        assert loss_fn(theta) < initial_loss * 0.1
        assert jnp.all(jnp.isfinite(theta))

    def test_identity_converges_to_fixed_point(self):
        """Identity Hessian with full-rank: same fixed-point as diagonal."""
        d = 3
        theta_star = jnp.array([1.0, -1.0, 2.0])
        s0 = 1.0

        def loss_fn(theta):
            return 0.5 * jnp.sum((theta - theta_star) ** 2)

        opt = blr_full_rank_for_loss(
            learning_rate=0.1,
            prior_precision=s0,
            hessian_estimator="identity",
        )
        theta = jnp.zeros(d)
        state = opt.init(theta)

        @jax.jit
        def step(carry, _):
            theta, state = carry
            g = jax.grad(loss_fn)(theta)
            updates, state = opt.update(g, state)
            theta = optax.apply_updates(theta, updates)
            return (theta, state), None

        (theta, state), _ = jax.lax.scan(step, (theta, state), None, length=100)

        m, _ = get_posterior_full_rank(state)
        analytic_m = theta_star / (1 + s0)
        assert jnp.allclose(m, analytic_m, atol=0.1)


# ── Loss wrapper sign flip ───────────────────────────────────────


class TestFullRankLossWrapper:
    def test_sign_flip(self):
        params = jnp.array([1.0, 2.0])
        grads_loss = jnp.array([0.5, -0.3])

        opt_loglik = blr_full_rank(learning_rate=0.1, prior_precision=1e-4)
        opt_loss = blr_full_rank_for_loss(learning_rate=0.1, prior_precision=1e-4)

        state_ll = opt_loglik.init(params)
        state_l = opt_loss.init(params)

        u_ll, _ = opt_loglik.update(-grads_loss, state_ll)
        u_l, _ = opt_loss.update(grads_loss, state_l)

        assert jnp.allclose(u_ll, u_l, atol=1e-7)


# ── Posterior extraction ─────────────────────────────────────────


class TestFullRankPosterior:
    def test_exact_values(self):
        precision = jnp.array([[2.0, 0.5], [0.5, 3.0]])
        nat_mean = jnp.array([1.0, 2.0])
        state = BLRFullRankState(
            precision=precision,
            nat_mean=nat_mean,
            count=jnp.array(5, jnp.int32),
        )
        m, cov = get_posterior_full_rank(state)

        expected_m = jnp.linalg.solve(precision, nat_mean)
        expected_cov = jnp.linalg.inv(precision)

        assert jnp.allclose(m, expected_m, atol=1e-5)
        assert jnp.allclose(cov, expected_cov, atol=1e-5)


# ── Root-level imports ───────────────────────────────────────────


class TestFullRankImports:
    def test_public_symbols(self):
        assert hasattr(optax_bayes, "BLRFullRankState")
        assert hasattr(optax_bayes, "blr_full_rank")
        assert hasattr(optax_bayes, "blr_full_rank_for_loss")
        assert hasattr(optax_bayes, "get_posterior_full_rank")
