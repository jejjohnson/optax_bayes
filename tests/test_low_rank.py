"""Tests for low-rank BLR transform and public API."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest

import optax_bayes
from optax_bayes import (
    BLRLowRankState,
    blr_low_rank,
    blr_low_rank_for_loss,
    get_posterior_low_rank,
)


# ── State shape / structure ──────────────────────────────────────


class TestLowRankInit:
    def test_shapes(self):
        params = jnp.zeros(10)
        opt = blr_low_rank(rank=3)
        state = opt.init(params)
        assert isinstance(state, BLRLowRankState)
        assert state.diag_precision.shape == (10,)
        assert state.low_rank_factor.shape == (10, 3)
        assert state.nat_mean.shape == (10,)
        assert state.count == 0

    def test_initial_precision_is_prior(self):
        params = jnp.zeros(5)
        opt = blr_low_rank(rank=2, prior_precision=0.5)
        state = opt.init(params)
        assert jnp.allclose(state.diag_precision, jnp.full(5, 0.5))
        # Initial U is zero (no low-rank component yet)
        assert jnp.allclose(state.low_rank_factor, jnp.zeros((5, 2)))


# ── Invalid estimator ────────────────────────────────────────────


class TestLowRankInvalidEstimator:
    def test_raises_on_unknown_string(self):
        with pytest.raises(ValueError, match="bad"):
            blr_low_rank(hessian_estimator="bad")


# ── Quadratic convergence ────────────────────────────────────────


class TestLowRankQuadratic:
    def test_ggn_loss_decreases(self):
        """Low-rank GGN drives the loss down."""
        d = 5
        theta_star = jnp.ones(d) * 2.0

        def loss_fn(theta):
            return 0.5 * jnp.sum((theta - theta_star) ** 2)

        opt = blr_low_rank_for_loss(learning_rate=0.1, rank=3, prior_precision=1e-4)
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

        (theta, state), _ = jax.lax.scan(step, (theta, state), None, length=100)

        assert loss_fn(theta) < initial_loss * 0.1
        assert jnp.all(jnp.isfinite(theta))

    def test_identity_converges_to_fixed_point(self):
        """Identity Hessian with low-rank: same as diagonal fixed-point."""
        d = 3
        theta_star = jnp.array([1.0, -1.0, 2.0])
        s0 = 1.0

        def loss_fn(theta):
            return 0.5 * jnp.sum((theta - theta_star) ** 2)

        opt = blr_low_rank_for_loss(
            learning_rate=0.1,
            rank=2,
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

        m, _ = get_posterior_low_rank(state)
        analytic_m = theta_star / (1 + s0)
        assert jnp.allclose(m, analytic_m, atol=0.1)


# ── Loss wrapper sign flip ───────────────────────────────────────


class TestLowRankLossWrapper:
    def test_sign_flip(self):
        params = jnp.array([1.0, 2.0, 3.0])
        grads_loss = jnp.array([0.5, -0.3, 0.1])

        opt_loglik = blr_low_rank(learning_rate=0.1, rank=2, prior_precision=1e-4)
        opt_loss = blr_low_rank_for_loss(
            learning_rate=0.1, rank=2, prior_precision=1e-4
        )

        state_ll = opt_loglik.init(params)
        state_l = opt_loss.init(params)

        u_ll, _ = opt_loglik.update(-grads_loss, state_ll)
        u_l, _ = opt_loss.update(grads_loss, state_l)

        assert jnp.allclose(u_ll, u_l, atol=1e-6)


# ── Posterior extraction ─────────────────────────────────────────


class TestLowRankPosterior:
    def test_matches_full_rank_inverse(self):
        """Low-rank posterior should match direct inverse of D + UU^T."""
        d_diag = jnp.array([1.0, 2.0, 3.0, 4.0])
        u = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
                [0.0, 0.0],
            ]
        )
        nat_mean = jnp.array([2.0, 4.0, 3.0, 1.0])

        state = BLRLowRankState(
            diag_precision=d_diag,
            low_rank_factor=u,
            nat_mean=nat_mean,
            count=jnp.array(10, jnp.int32),
        )

        m, cov = get_posterior_low_rank(state)

        # Reference: build full matrix and invert
        full_lambda = jnp.diag(d_diag) + u @ u.T
        expected_m = jnp.linalg.solve(full_lambda, nat_mean)
        expected_cov = jnp.linalg.inv(full_lambda)

        assert jnp.allclose(m, expected_m, atol=1e-5)
        assert jnp.allclose(cov, expected_cov, atol=1e-5)

    def test_zero_u_is_diagonal(self):
        """When U=0, low-rank posterior = diagonal posterior."""
        d_diag = jnp.array([2.0, 4.0])
        u = jnp.zeros((2, 1))
        nat_mean = jnp.array([6.0, -8.0])

        state = BLRLowRankState(
            diag_precision=d_diag,
            low_rank_factor=u,
            nat_mean=nat_mean,
            count=jnp.array(0, jnp.int32),
        )
        m, cov = get_posterior_low_rank(state)

        assert jnp.allclose(m, jnp.array([3.0, -2.0]))
        assert jnp.allclose(cov, jnp.diag(jnp.array([0.5, 0.25])), atol=1e-6)


# ── Root-level imports ───────────────────────────────────────────


class TestLowRankImports:
    def test_public_symbols(self):
        assert hasattr(optax_bayes, "BLRLowRankState")
        assert hasattr(optax_bayes, "blr_low_rank")
        assert hasattr(optax_bayes, "blr_low_rank_for_loss")
        assert hasattr(optax_bayes, "get_posterior_low_rank")
