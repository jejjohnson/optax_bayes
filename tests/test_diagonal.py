"""Tests for the diagonal BLR transform and public API (#32, #33, #34)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

import optax_bayes
from optax_bayes import (
    BLRDiagState,
    blr_diagonal,
    blr_diagonal_for_loss,
    get_posterior_diagonal,
)


# ── State shape / structure ──────────────────────────────────────


class TestInitState:
    def test_flat_params(self):
        params = jnp.zeros(5)
        opt = blr_diagonal()
        state = opt.init(params)
        assert isinstance(state, BLRDiagState)
        assert state.precision.shape == (5,)
        assert state.nat_mean.shape == (5,)
        assert state.count == 0

    def test_pytree_params(self):
        params = {"w": jnp.zeros((3, 4)), "b": jnp.zeros(4)}
        opt = blr_diagonal()
        state = opt.init(params)
        assert state.precision["w"].shape == (3, 4)
        assert state.precision["b"].shape == (4,)
        assert state.nat_mean["w"].shape == (3, 4)

    def test_prior_precision_broadcast(self):
        params = jnp.zeros(3)
        opt = blr_diagonal(prior_precision=2.0, prior_mean=1.0)
        state = opt.init(params)
        assert jnp.allclose(state.precision, jnp.full(3, 2.0))

    def test_initial_mean_is_params(self):
        # Standard optax drop-in semantics: the variational mean starts
        # at the params passed to init, not at the prior mean.
        params = jnp.array([1.0, -2.0, 0.5])
        opt = blr_diagonal(prior_precision=2.0, prior_mean=1.0)
        state = opt.init(params)
        # eta_0 = s_0 * params  =>  m_0 = eta_0 / s_0 = params
        assert jnp.allclose(state.nat_mean / state.precision, params)


# ── Invalid estimator ────────────────────────────────────────────


class TestInvalidEstimator:
    def test_raises_on_unknown(self):
        import pytest

        with pytest.raises(ValueError, match="bad"):
            blr_diagonal(hessian_estimator="bad")


# ── Quadratic objective convergence ──────────────────────────────


class TestQuadraticConvergence:
    """Train on f(theta) = 0.5 * ||theta - theta*||^2.

    With identity Hessian (H=0), the BLR fixed-point on this quadratic
    is m* = theta* / (1 + s0),  v* = 1 / s0.

    With GGN (H = -g^2), the approximation vanishes near the optimum
    so we test that the loss decreases and the mean moves toward theta*.
    """

    def test_diagonal_ggn_loss_decreases(self):
        """GGN BLR should drive the loss down on a quadratic."""
        d = 5
        theta_star = jnp.ones(d) * 3.0

        def loss_fn(theta):
            return 0.5 * jnp.sum((theta - theta_star) ** 2)

        opt = blr_diagonal_for_loss(learning_rate=0.1, prior_precision=1e-4)
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

        final_loss = loss_fn(theta)
        assert final_loss < initial_loss * 0.01
        assert jnp.all(jnp.isfinite(theta))

        # Mean should be in the right direction
        m, _v = get_posterior_diagonal(state)
        assert jnp.all(m > 1.0)  # toward theta_star=3

    def test_identity_estimator_converges(self):
        """Identity Hessian (H=0) converges to the BLR fixed-point.

        With H=0, precision stays at s0 forever, so:
          m* = theta* / (1 + s0),  v* = 1 / s0
        """
        d = 3
        theta_star = jnp.array([1.0, -1.0, 2.0])
        s0 = 1.0

        def loss_fn(theta):
            return 0.5 * jnp.sum((theta - theta_star) ** 2)

        opt = blr_diagonal_for_loss(
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

        m, v = get_posterior_diagonal(state)

        # Analytic fixed point for identity hessian
        analytic_m = theta_star / (1 + s0)  # [0.5, -0.5, 1.0]
        analytic_v = 1.0 / s0  # 1.0

        assert jnp.allclose(m, analytic_m, atol=0.1)
        assert jnp.allclose(v, jnp.full(d, analytic_v), atol=0.1)


# ── Loss wrapper sign flip ───────────────────────────────────────


class TestLossWrapper:
    def test_sign_flip(self):
        """blr_diagonal_for_loss should negate grads before forwarding."""
        params = jnp.array([1.0, 2.0])
        grads_loss = jnp.array([0.5, -0.3])

        opt_loglik = blr_diagonal(learning_rate=0.1, prior_precision=1e-4)
        opt_loss = blr_diagonal_for_loss(learning_rate=0.1, prior_precision=1e-4)

        state_loglik = opt_loglik.init(params)
        state_loss = opt_loss.init(params)

        # log-lik gets -grads_loss, loss wrapper gets grads_loss
        u_loglik, _ = opt_loglik.update(-grads_loss, state_loglik)
        u_loss, _ = opt_loss.update(grads_loss, state_loss)

        assert jnp.allclose(u_loglik, u_loss, atol=1e-7)


# ── Posterior extraction ─────────────────────────────────────────


class TestPosteriorExtraction:
    def test_exact_values(self):
        """get_posterior_diagonal should return eta/s and 1/s."""
        precision = jnp.array([2.0, 4.0, 0.5])
        nat_mean = jnp.array([6.0, -8.0, 1.5])
        state = BLRDiagState(
            precision=precision,
            nat_mean=nat_mean,
            count=jnp.array(10, jnp.int32),
        )

        m, v = get_posterior_diagonal(state)

        assert jnp.allclose(m, jnp.array([3.0, -2.0, 3.0]))
        assert jnp.allclose(v, jnp.array([0.5, 0.25, 2.0]))

    def test_pytree_state(self):
        state = BLRDiagState(
            precision={"w": jnp.array([1.0, 2.0]), "b": jnp.array([4.0])},
            nat_mean={"w": jnp.array([3.0, 6.0]), "b": jnp.array([8.0])},
            count=jnp.array(0, jnp.int32),
        )
        m, v = get_posterior_diagonal(state)
        assert jnp.allclose(m["w"], jnp.array([3.0, 3.0]))
        assert jnp.allclose(m["b"], jnp.array([2.0]))
        assert jnp.allclose(v["w"], jnp.array([1.0, 0.5]))
        assert jnp.allclose(v["b"], jnp.array([0.25]))


# ── Optax chain composition ─────────────────────────────────────


class TestOptaxComposition:
    def test_chain_with_clip(self):
        """BLR should compose with optax.chain."""
        opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            blr_diagonal_for_loss(learning_rate=0.01),
        )
        params = jnp.zeros(3)
        state = opt.init(params)
        grads = jnp.array([10.0, 10.0, 10.0])
        updates, state = opt.update(grads, state)
        # Should not crash; updates should be finite
        assert jnp.all(jnp.isfinite(updates))


# ── Root-level imports ───────────────────────────────────────────


class TestRootImports:
    def test_all_public_symbols(self):
        assert hasattr(optax_bayes, "BLRDiagState")
        assert hasattr(optax_bayes, "blr_diagonal")
        assert hasattr(optax_bayes, "blr_diagonal_for_loss")
        assert hasattr(optax_bayes, "get_posterior_diagonal")

    def test_version(self):
        assert optax_bayes.__version__ == "0.1.0"
