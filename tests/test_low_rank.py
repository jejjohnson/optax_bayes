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
        """Low-rank GGN drives the loss down.

        Same setting as the full-rank test: the rank-1 outer product
        H = -g g^T is unstable here under a near-flat prior, exactly as
        it is for ``blr_full_rank``.
        """
        d = 3
        theta_star = jnp.ones(d) * 2.0

        def loss_fn(theta):
            return 0.5 * jnp.sum((theta - theta_star) ** 2)

        opt = blr_low_rank_for_loss(learning_rate=0.1, rank=2, prior_precision=1e-2)
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


# ── SLANG update ─────────────────────────────────────────────────


def _exact_precision_problem(d=4):
    """A fixed concave quadratic log-likelihood with exact Hessian -A."""
    key = jax.random.key(1)
    x = jax.random.normal(key, (3 * d, d)) / jnp.sqrt(3 * d)
    a_mat = x.T @ x
    return a_mat, 1.0 * jnp.eye(d) + a_mat


class TestLowRankSLANG:
    def _fit(self, rank, n_steps=200):
        a_mat, post_precision = _exact_precision_problem()
        opt = blr_low_rank(
            learning_rate=0.3,
            rank=rank,
            prior_precision=1.0,
            hessian_estimator=lambda mean, grads: -a_mat,
            damping=0.0,
        )
        params = jnp.zeros(4)
        state = opt.init(params)
        for _ in range(n_steps):
            updates, state = opt.update(-a_mat @ params, state, params)
            params = params + updates
        precision = jnp.diag(state.diag_precision) + (
            state.low_rank_factor @ state.low_rank_factor.T
        )
        return precision, post_precision

    def test_full_rank_recovers_exact_precision(self):
        """With rank >= d nothing is truncated, so the update is exact."""
        precision, post_precision = self._fit(rank=4)
        assert jnp.allclose(precision, post_precision, atol=1e-5)

    def test_truncation_keeps_the_diagonal_exact(self):
        """Below full rank, the discarded diagonal is folded into D."""
        precision, post_precision = self._fit(rank=2)
        assert jnp.allclose(jnp.diag(precision), jnp.diag(post_precision), atol=1e-5)
        assert not jnp.allclose(precision, post_precision, atol=1e-3)

    @pytest.mark.parametrize(
        ("hessian_estimator", "prior_precision", "n_steps"),
        [
            # Exact Hessian, informative prior: contractive, so compare long.
            pytest.param(
                lambda mean, grads: jnp.array([[2.0, 0.5], [0.5, 1.0]]),
                1.0,
                30,
                id="exact",
            ),
            # The rank-1 GGN amplifies round-off ~8x per step on this toy
            # problem (in both transforms), so compare only a few steps.
            pytest.param(
                "ggn",
                1e-2,
                5,
                id="ggn",
                marks=pytest.mark.x64_only(
                    reason="the GGN dynamics amplify float32 round-off past 1e-6"
                ),
            ),
        ],
    )
    def test_rank_d_matches_full_rank(
        self, hessian_estimator, prior_precision, n_steps
    ):
        """Untruncated, the low-rank transform is the full-rank BLR."""
        theta_star = jnp.array([2.0, -1.0])
        kwargs = dict(
            learning_rate=0.1,
            prior_precision=prior_precision,
            hessian_estimator=hessian_estimator,
            damping=0.0,
        )
        full = optax_bayes.blr_full_rank_for_loss(**kwargs)
        low = blr_low_rank_for_loss(rank=2, **kwargs)
        theta_f = theta_l = jnp.zeros(2)
        state_f, state_l = full.init(theta_f), low.init(theta_l)
        for _ in range(n_steps):
            u_f, state_f = full.update(theta_f - theta_star, state_f)
            u_l, state_l = low.update(theta_l - theta_star, state_l)
            theta_f, theta_l = theta_f + u_f, theta_l + u_l
        assert jnp.allclose(theta_l, theta_f, rtol=1e-6, atol=1e-6)

    def test_ggn_never_eigendecomposes(self):
        """The built-in estimators stay O(d r^2): no dense eigh per step."""
        opt = blr_low_rank(rank=2)
        params = jnp.ones(6)
        state = opt.init(params)
        jaxpr = str(jax.make_jaxpr(opt.update)(params, state, params))
        assert "eigh" not in jaxpr

    def test_ggn_large_d_does_not_form_dense_matrices(self):
        """A (d, d) array would be 4e10 entries here; the step must not need one."""
        d = 200_000
        opt = blr_low_rank(rank=3)
        params = jnp.zeros(d)
        state = opt.init(params)
        out = jax.eval_shape(opt.update, params, state, params)
        largest = max(
            leaf.size for leaf in jax.tree.leaves(out) if hasattr(leaf, "size")
        )
        assert largest <= 3 * d
        jaxpr = jax.make_jaxpr(opt.update)(params, state, params)
        assert all(
            v.aval.size <= 4 * d
            for eqn in jaxpr.eqns
            for v in eqn.outvars
            if hasattr(v.aval, "size")
        )


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
