"""End-to-end posterior recovery on conjugate linear-Gaussian models.

For Bayesian linear regression with unit noise variance and prior
N(0, s0^{-1} I), the exact posterior is available in closed form:

    Lambda* = s0 I + X^T X
    m*      = Lambda*^{-1} X^T y

These tests train each BLR variant to convergence and check it against
the analytic answer, so the suite fails if any term in the update rule
(prior anchoring, the -H m correction, the gradient sign convention)
is wrong.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import optax_bayes


pytestmark = [pytest.mark.slow, pytest.mark.integration]

PRIOR_PRECISION = 1.0


@pytest.fixture(scope="module")
def linear_gaussian():
    """A well-conditioned Bayesian linear regression problem."""
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    n, d = 40, 4
    x = jax.random.normal(k1, (n, d)) / jnp.sqrt(n)
    w_true = jax.random.normal(k2, (d,))
    y = x @ w_true + 0.3 * jax.random.normal(k3, (n,))

    precision = PRIOR_PRECISION * jnp.eye(d) + x.T @ x
    mean = jnp.linalg.solve(precision, x.T @ y)
    return {"x": x, "y": y, "post_precision": precision, "post_mean": mean}


def _loss_grads(x, y, theta):
    """Gradient of the negative log-likelihood 0.5 * ||y - x theta||^2."""
    return x.T @ (x @ theta - y)


def _train(opt, d, grad_fn, num_steps):
    """Run a scan training loop from the prior mean (zeros)."""
    params = jnp.zeros(d)
    state = opt.init(params)

    def step(carry, _):
        params, state = carry
        updates, state = opt.update(grad_fn(params), state, params)
        params = optax.apply_updates(params, updates)
        return (params, state), None

    (params, state), _ = jax.lax.scan(step, (params, state), None, length=num_steps)
    return params, state


class TestFullRankExactRecovery:
    def test_loglik_convention_with_exact_hessian(self, linear_gaussian):
        x, y = linear_gaussian["x"], linear_gaussian["y"]
        d = x.shape[1]
        opt = optax_bayes.blr_full_rank(
            learning_rate=0.5,
            prior_precision=PRIOR_PRECISION,
            hessian_estimator=lambda mean, grads: -(x.T @ x),
            damping=0.0,
        )
        params, state = _train(opt, d, lambda t: -_loss_grads(x, y, t), 100)

        np.testing.assert_allclose(
            state.precision, linear_gaussian["post_precision"], rtol=1e-8
        )
        mean, cov = optax_bayes.get_posterior_full_rank(state)
        np.testing.assert_allclose(mean, linear_gaussian["post_mean"], rtol=1e-6)
        np.testing.assert_allclose(
            cov, jnp.linalg.inv(linear_gaussian["post_precision"]), rtol=1e-6
        )
        np.testing.assert_allclose(params, linear_gaussian["post_mean"], rtol=1e-6)

    def test_loss_convention_with_exact_hessian(self, linear_gaussian):
        x, y = linear_gaussian["x"], linear_gaussian["y"]
        d = x.shape[1]
        opt = optax_bayes.blr_full_rank_for_loss(
            learning_rate=0.5,
            prior_precision=PRIOR_PRECISION,
            hessian_estimator=lambda mean, grads: x.T @ x,  # loss Hessian
            damping=0.0,
        )
        _, state = _train(opt, d, lambda t: _loss_grads(x, y, t), 100)

        np.testing.assert_allclose(
            state.precision, linear_gaussian["post_precision"], rtol=1e-8
        )
        mean, _ = optax_bayes.get_posterior_full_rank(state)
        np.testing.assert_allclose(mean, linear_gaussian["post_mean"], rtol=1e-6)


class TestDiagonalMeanRecovery:
    """With the identity Hessian estimator, the precision stays at the
    prior and the converged mean solves the exact MAP equation — which
    for linear-Gaussian models is the exact posterior mean."""

    def test_identity_estimator_recovers_posterior_mean(self, linear_gaussian):
        x, y = linear_gaussian["x"], linear_gaussian["y"]
        d = x.shape[1]
        opt = optax_bayes.blr_diagonal_for_loss(
            learning_rate=0.2,
            prior_precision=PRIOR_PRECISION,
            hessian_estimator="identity",
            damping=1e-12,
        )
        _, state = _train(opt, d, lambda t: _loss_grads(x, y, t), 500)

        mean, variance = optax_bayes.get_posterior_diagonal(state)
        np.testing.assert_allclose(mean, linear_gaussian["post_mean"], rtol=1e-6)
        np.testing.assert_allclose(variance, 1.0 / PRIOR_PRECISION, rtol=1e-10)


class TestLowRankMeanRecovery:
    def test_identity_estimator_recovers_posterior_mean(self, linear_gaussian):
        x, y = linear_gaussian["x"], linear_gaussian["y"]
        d = x.shape[1]
        opt = optax_bayes.blr_low_rank_for_loss(
            learning_rate=0.2,
            rank=2,
            prior_precision=PRIOR_PRECISION,
            hessian_estimator="identity",
            damping=1e-12,
        )
        _, state = _train(opt, d, lambda t: _loss_grads(x, y, t), 500)

        mean, _ = optax_bayes.get_posterior_low_rank(state)
        np.testing.assert_allclose(mean, linear_gaussian["post_mean"], rtol=1e-6)


class TestNewtonRecovery:
    def test_quadratic_loss_converges_to_minimizer(self):
        a_mat = jnp.array([[3.0, 0.5], [0.5, 2.0]])
        target = jnp.array([1.0, -2.0])
        opt = optax_bayes.newton_for_loss(
            loss_hessian_fn=lambda mean: a_mat, damping=1e-10
        )
        params, _ = _train(opt, 2, lambda t: a_mat @ (t - target), num_steps=5)
        np.testing.assert_allclose(params, target, rtol=1e-6)


class TestIVONMapRecovery:
    def test_converges_to_weight_decay_map(self):
        target = jnp.array([2.0, -1.0, 0.5])
        weight_decay = 0.1
        opt = optax_bayes.ivon(learning_rate=0.05, weight_decay=weight_decay)
        params, _ = _train(opt, 3, lambda t: t - target, num_steps=3000)
        # argmin 0.5||t - a||^2 + wd/2 ||t||^2  =  a / (1 + wd)
        np.testing.assert_allclose(params, target / (1 + weight_decay), rtol=1e-4)


class TestSamplingMoments:
    """Monte Carlo moments of the posterior samplers must match the
    Gaussian they claim to sample from."""

    def test_full_rank_sample_mean_and_cov(self, linear_gaussian):
        precision = linear_gaussian["post_precision"]
        mean = linear_gaussian["post_mean"]
        state = optax_bayes.BLRFullRankState(
            precision=precision,
            nat_mean=precision @ mean,
            count=jnp.zeros([], jnp.int32),
        )
        keys = jax.random.split(jax.random.key(1), 8000)
        samples = jax.vmap(lambda k: optax_bayes.sample_posterior_full_rank(state, k))(
            keys
        )

        cov_true = jnp.linalg.inv(precision)
        np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.05)
        np.testing.assert_allclose(
            jnp.cov(samples.T), cov_true, atol=0.05 * float(jnp.max(cov_true))
        )

    def test_low_rank_sample_cov(self):
        d, r = 6, 2
        key = jax.random.key(2)
        diag = 2.0 * jnp.ones(d)
        u = jax.random.normal(key, (d, r))
        precision = jnp.diag(diag) + u @ u.T
        mean = jnp.linspace(-1.0, 1.0, d)
        state = optax_bayes.BLRLowRankState(
            diag_precision=diag,
            low_rank_factor=u,
            nat_mean=precision @ mean,
            count=jnp.zeros([], jnp.int32),
        )
        keys = jax.random.split(jax.random.key(3), 8000)
        samples = jax.vmap(lambda k: optax_bayes.sample_posterior_low_rank(state, k))(
            keys
        )

        cov_true = jnp.linalg.inv(precision)
        np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.05)
        np.testing.assert_allclose(
            jnp.cov(samples.T), cov_true, atol=0.05 * float(jnp.max(cov_true))
        )

    def test_diagonal_sample_variance(self):
        s = jnp.array([1.0, 4.0, 0.25])
        m = jnp.array([0.0, 1.0, -1.0])
        state = optax_bayes.BLRDiagState(
            precision=s,
            nat_mean=s * m,
            count=jnp.zeros([], jnp.int32),
        )
        keys = jax.random.split(jax.random.key(4), 8000)
        samples = jax.vmap(lambda k: optax_bayes.sample_posterior_diagonal(state, k))(
            keys
        )
        np.testing.assert_allclose(samples.mean(axis=0), m, atol=0.1)
        np.testing.assert_allclose(samples.var(axis=0), 1.0 / s, rtol=0.15)
