"""Tests for Layer 0 primitives and conversions (#30)."""

from __future__ import annotations

import jax.numpy as jnp

from optax_bayes._src.conversions import (
    mean_to_natural_diag,
    natural_to_mean_diag,
)
from optax_bayes._src.primitives import blr_diag_update_step


# ── Conversion round-trips ──────────────────────────────────────


class TestConversions:
    def test_natural_to_mean_values(self):
        eta = jnp.array([2.0, 6.0])
        s = jnp.array([1.0, 3.0])
        m, v = natural_to_mean_diag(eta, s)
        assert jnp.allclose(m, jnp.array([2.0, 2.0]))
        assert jnp.allclose(v, jnp.array([1.0, 1.0 / 3.0]))

    def test_mean_to_natural_values(self):
        m = jnp.array([2.0, 2.0])
        v = jnp.array([1.0, 0.5])
        eta, s = mean_to_natural_diag(m, v)
        assert jnp.allclose(eta, jnp.array([2.0, 4.0]))
        assert jnp.allclose(s, jnp.array([1.0, 2.0]))

    def test_round_trip_natural_mean_natural(self):
        eta = jnp.array([3.0, -1.0, 0.5])
        s = jnp.array([2.0, 4.0, 1.0])
        m, v = natural_to_mean_diag(eta, s)
        eta2, s2 = mean_to_natural_diag(m, v)
        assert jnp.allclose(eta, eta2, atol=1e-6)
        assert jnp.allclose(s, s2, atol=1e-6)

    def test_round_trip_mean_natural_mean(self):
        m = jnp.array([1.0, -2.0, 0.0])
        v = jnp.array([0.5, 2.0, 1.0])
        eta, s = mean_to_natural_diag(m, v)
        m2, v2 = natural_to_mean_diag(eta, s)
        assert jnp.allclose(m, m2, atol=1e-6)
        assert jnp.allclose(v, v2, atol=1e-6)


# ── BLR diagonal update step ────────────────────────────────────


class TestBLRDiagUpdateStep:
    def test_exact_values(self):
        """Verify the update matches the equations term-for-term."""
        eta = jnp.array([1.0, 2.0])
        s = jnp.array([1.0, 1.0])
        g = jnp.array([0.5, -0.3])
        diag_h = jnp.array([-0.25, -0.09])  # h = -(g**2)
        m = eta / s  # [1.0, 2.0]
        eta0 = jnp.array([0.0, 0.0])
        s0 = jnp.array([0.01, 0.01])
        rho = 0.1

        eta_new, s_new, m_new = blr_diag_update_step(
            eta, s, g, diag_h, m, eta0, s0, rho
        )

        # Expected by hand:
        # s_next = 0.9 * [1, 1] + 0.1 * ([0.01, 0.01] - [-0.25, -0.09])
        #        = [0.9, 0.9] + 0.1 * [0.26, 0.10]
        #        = [0.926, 0.910]
        s_expected = (1 - rho) * s + rho * (s0 - diag_h)
        assert jnp.allclose(s_new, s_expected, atol=1e-6)

        # eta_next = 0.9*[1,2] + 0.1*([0,0]+[0.5,-0.3]-[-0.25,-0.09]*[1,2])
        # = [0.9, 1.8] + 0.1 * [0.75, -0.12] = [0.975, 1.788]
        eta_expected = (1 - rho) * eta + rho * (eta0 + g - diag_h * m)
        assert jnp.allclose(eta_new, eta_expected, atol=1e-6)

        # m_next = eta_next / s_next
        assert jnp.allclose(m_new, eta_expected / s_expected, atol=1e-6)

    def test_correction_term_matters(self):
        """The -h*m correction term must affect the result.

        If we zero out the correction (pretend h=0 in the eta update
        but keep it in the s update), the result should differ.
        """
        eta = jnp.array([1.0, 2.0])
        s = jnp.array([1.0, 1.0])
        g = jnp.array([0.5, 0.5])
        diag_h = jnp.array([-0.25, -0.25])
        m = eta / s
        eta0 = jnp.zeros(2)
        s0 = jnp.full(2, 0.01)
        rho = 0.1

        # With correction
        eta_correct, _, _ = blr_diag_update_step(eta, s, g, diag_h, m, eta0, s0, rho)

        # Without correction (h=0 for eta, but keep h for s would be
        # a different function — here we just check they differ)
        eta_wrong = (1 - rho) * eta + rho * (eta0 + g)

        assert not jnp.allclose(eta_correct, eta_wrong)

    def test_rho_zero_is_identity(self):
        """rho=0 should leave state unchanged."""
        eta = jnp.array([3.0, -1.0])
        s = jnp.array([2.0, 4.0])
        m = eta / s
        g = jnp.array([10.0, 10.0])
        diag_h = jnp.array([-100.0, -100.0])
        eta0 = jnp.zeros(2)
        s0 = jnp.ones(2)

        eta_new, s_new, m_new = blr_diag_update_step(
            eta, s, g, diag_h, m, eta0, s0, rho=0.0
        )
        assert jnp.allclose(eta_new, eta)
        assert jnp.allclose(s_new, s)
        assert jnp.allclose(m_new, m)
