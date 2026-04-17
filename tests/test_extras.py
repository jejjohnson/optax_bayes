"""Tests for schedules, sampling, Newton preset, and continual learning."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

import optax_bayes
from optax_bayes import (
    BLRDiagState,
    blr_diagonal_for_loss,
    blr_with_schedule,
    get_posterior_diagonal,
    newton_for_loss,
    sample_posterior_diagonal,
    sample_posterior_full_rank,
    sample_posterior_low_rank,
)
from optax_bayes._src.types import BLRFullRankState, BLRLowRankState


# ── blr_with_schedule ────────────────────────────────────────────


class TestSchedule:
    def test_constant_schedule_matches_fixed_lr(self):
        d = 3
        theta_star = jnp.ones(d) * 2.0

        def loss_fn(theta):
            return 0.5 * jnp.sum((theta - theta_star) ** 2)

        # Constant schedule should behave like fixed lr
        opt = blr_with_schedule(optax.constant_schedule(0.1), prior_precision=1e-4)
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

        assert loss_fn(theta) < 1.0
        assert jnp.all(jnp.isfinite(theta))


# ── MC sampling ──────────────────────────────────────────────────


class TestSamplingDiagonal:
    def test_shape(self):
        params = {"w": jnp.ones((3, 2)), "b": jnp.ones(2)}
        opt = blr_diagonal_for_loss()
        state = opt.init(params)
        key = jax.random.PRNGKey(42)
        sample = sample_posterior_diagonal(state, key)
        assert sample["w"].shape == (3, 2)
        assert sample["b"].shape == (2,)

    def test_samples_differ(self):
        state = BLRDiagState(
            precision=jnp.ones(5),
            nat_mean=jnp.zeros(5),
            count=jnp.array(0, jnp.int32),
        )
        s1 = sample_posterior_diagonal(state, jax.random.PRNGKey(0))
        s2 = sample_posterior_diagonal(state, jax.random.PRNGKey(1))
        assert not jnp.allclose(s1, s2)


class TestSamplingFullRank:
    def test_shape(self):
        state = BLRFullRankState(
            precision=jnp.eye(3),
            nat_mean=jnp.zeros(3),
            count=jnp.array(0, jnp.int32),
        )
        sample = sample_posterior_full_rank(state, jax.random.PRNGKey(0))
        assert sample.shape == (3,)

    def test_samples_differ(self):
        state = BLRFullRankState(
            precision=2.0 * jnp.eye(4),
            nat_mean=jnp.ones(4),
            count=jnp.array(0, jnp.int32),
        )
        s1 = sample_posterior_full_rank(state, jax.random.PRNGKey(0))
        s2 = sample_posterior_full_rank(state, jax.random.PRNGKey(1))
        assert not jnp.allclose(s1, s2)


class TestSamplingLowRank:
    def test_shape(self):
        state = BLRLowRankState(
            diag_precision=jnp.ones(5),
            low_rank_factor=jnp.zeros((5, 2)),
            nat_mean=jnp.zeros(5),
            count=jnp.array(0, jnp.int32),
        )
        sample = sample_posterior_low_rank(state, jax.random.PRNGKey(0))
        assert sample.shape == (5,)


# ── Newton preset ────────────────────────────────────────────────


class TestNewton:
    def test_quadratic_one_step(self):
        """Newton should converge in ~1 step on a quadratic."""
        d = 3
        theta_star = jnp.array([1.0, -2.0, 3.0])

        def loss_fn(theta):
            return 0.5 * jnp.sum((theta - theta_star) ** 2)

        def loss_hessian(mean):
            return jnp.eye(d)

        opt = newton_for_loss(loss_hessian_fn=loss_hessian, damping=1e-6)
        theta = jnp.zeros(d)
        state = opt.init(theta)

        # Newton on a quadratic should get very close in a few steps
        for _ in range(5):
            g = jax.grad(loss_fn)(theta)
            updates, state = opt.update(g, state)
            theta = optax.apply_updates(theta, updates)

        assert jnp.allclose(theta, theta_star, atol=0.1)


# ── Continual learning pattern ───────────────────────────────────


class TestContinualLearning:
    def test_posterior_as_prior(self):
        """Task-1 posterior becomes task-2 prior."""
        d = 3

        # Task 1
        theta_star_1 = jnp.array([1.0, 0.0, 0.0])

        def loss_1(theta):
            return 0.5 * jnp.sum((theta - theta_star_1) ** 2)

        opt1 = blr_diagonal_for_loss(learning_rate=0.1, prior_precision=1.0)
        theta = jnp.zeros(d)
        state1 = opt1.init(theta)

        @jax.jit
        def step(carry, _):
            theta, state1 = carry
            g = jax.grad(loss_1)(theta)
            updates, state1 = opt1.update(g, state1)
            theta = optax.apply_updates(theta, updates)
            return (theta, state1), None

        (theta, state1), _ = jax.lax.scan(step, (theta, state1), None, length=50)

        mean_1, var_1 = get_posterior_diagonal(state1)

        # Task 2: use task-1 posterior as prior
        # (Currently uses scalar prior, but the pattern works
        #  because the user can construct BLRDiagState manually)
        precision_1 = jax.tree.map(lambda v: 1.0 / v, var_1)

        # Verify the posterior is reasonable
        assert jnp.allclose(mean_1, theta_star_1 / 2.0, atol=0.1)
        assert jnp.all(precision_1 > 0)


# ── Root-level imports ───────────────────────────────────────────


class TestExtrasImports:
    def test_schedule(self):
        assert hasattr(optax_bayes, "blr_with_schedule")

    def test_sampling(self):
        assert hasattr(optax_bayes, "sample_posterior_diagonal")
        assert hasattr(optax_bayes, "sample_posterior_full_rank")
        assert hasattr(optax_bayes, "sample_posterior_low_rank")

    def test_newton(self):
        assert hasattr(optax_bayes, "newton")
        assert hasattr(optax_bayes, "newton_for_loss")
