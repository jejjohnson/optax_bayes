"""Tests for Hessian estimators."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from optax_bayes._src.hessians import (
    ggn_diag,
    ggn_outer,
    identity_hessian,
    identity_hessian_full,
    psd_factor,
    resolve_hessian_estimator_full,
    resolve_hessian_factor_low_rank,
)


class TestGGNDiag:
    def test_values(self):
        g = jnp.array([2.0, -3.0, 0.0])
        h = ggn_diag(g)
        assert jnp.allclose(h, jnp.array([-4.0, -9.0, 0.0]))

    def test_always_non_positive(self):
        g = jnp.array([1.0, -2.0, 3.0, 0.0, -0.5])
        h = ggn_diag(g)
        assert jnp.all(h <= 0)

    def test_shape_preserved(self):
        g = jnp.ones((3, 4))
        h = ggn_diag(g)
        assert h.shape == (3, 4)

    def test_pytree(self):
        grads = {"w": jnp.array([1.0, 2.0]), "b": jnp.array([3.0])}
        h = jax.tree.map(ggn_diag, grads)
        assert jnp.allclose(h["w"], jnp.array([-1.0, -4.0]))
        assert jnp.allclose(h["b"], jnp.array([-9.0]))


class TestIdentityHessian:
    def test_returns_zeros(self):
        g = jnp.array([5.0, -3.0, 1.0])
        h = identity_hessian(g)
        assert jnp.allclose(h, jnp.zeros(3))

    def test_shape_preserved(self):
        g = jnp.ones((2, 5))
        h = identity_hessian(g)
        assert h.shape == (2, 5)
        assert jnp.all(h == 0)

    def test_pytree(self):
        grads = {"w": jnp.ones((3,)), "b": jnp.ones((1,))}
        h = jax.tree.map(identity_hessian, grads)
        assert jnp.allclose(h["w"], jnp.zeros(3))
        assert jnp.allclose(h["b"], jnp.zeros(1))


# ── Full-rank estimators ────────────────────────────────────────


class TestGGNOuter:
    def test_values(self):
        g = jnp.array([1.0, 2.0, 3.0])
        h = ggn_outer(g)
        expected = -jnp.outer(g, g)
        assert jnp.allclose(h, expected)

    def test_always_nsd(self):
        """Eigenvalues of -g g^T should be <= 0."""
        g = jnp.array([1.0, -2.0, 0.5])
        h = ggn_outer(g)
        eigvals = jnp.linalg.eigvalsh(h)
        assert jnp.all(eigvals <= 1e-6)

    def test_rank_one(self):
        g = jnp.array([1.0, 2.0, 3.0])
        h = ggn_outer(g)
        assert jnp.linalg.matrix_rank(h) == 1

    def test_shape(self):
        g = jnp.ones(5)
        h = ggn_outer(g)
        assert h.shape == (5, 5)


class TestIdentityHessianFull:
    def test_returns_zeros(self):
        g = jnp.array([1.0, 2.0, 3.0])
        h = identity_hessian_full(g)
        assert jnp.allclose(h, jnp.zeros((3, 3)))

    def test_shape(self):
        g = jnp.ones(7)
        h = identity_hessian_full(g)
        assert h.shape == (7, 7)


class TestResolveHessianEstimator:
    def test_ggn_string(self):
        fn = resolve_hessian_estimator_full("ggn")
        g = jnp.array([1.0, 2.0])
        m = jnp.zeros(2)
        h = fn(m, g)
        assert jnp.allclose(h, -jnp.outer(g, g))

    def test_identity_string(self):
        fn = resolve_hessian_estimator_full("identity")
        g = jnp.ones(3)
        m = jnp.zeros(3)
        h = fn(m, g)
        assert jnp.allclose(h, jnp.zeros((3, 3)))

    def test_callable(self):
        custom = lambda m, g: -2.0 * jnp.eye(m.shape[0])
        fn = resolve_hessian_estimator_full(custom)
        assert fn is custom

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="bad"):
            resolve_hessian_estimator_full("bad")


class TestResolveHessianFactorLowRank:
    def test_ggn_is_the_gradient(self):
        g = jnp.array([1.0, -2.0, 0.5])
        factor = resolve_hessian_factor_low_rank("ggn")(jnp.zeros(3), g)
        assert factor.shape == (3, 1)
        assert jnp.allclose(factor @ factor.T, jnp.outer(g, g))

    def test_identity_is_empty(self):
        factor = resolve_hessian_factor_low_rank("identity")(jnp.zeros(3), jnp.ones(3))
        assert factor.shape == (3, 0)

    def test_callable_factors_negative_hessian(self):
        neg_h = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        fn = resolve_hessian_factor_low_rank(lambda m, g: -neg_h)
        factor = fn(jnp.zeros(2), jnp.zeros(2))
        assert jnp.allclose(factor @ factor.T, neg_h, atol=1e-6)

    def test_psd_factor_drops_negative_curvature(self):
        factor = psd_factor(jnp.diag(jnp.array([3.0, -1.0])))
        assert jnp.allclose(factor @ factor.T, jnp.diag(jnp.array([3.0, 0.0])))

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="bad"):
            resolve_hessian_factor_low_rank("bad")

    def test_non_callable_raises(self):
        with pytest.raises(TypeError):
            resolve_hessian_factor_low_rank(3)
