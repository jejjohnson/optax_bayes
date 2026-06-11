"""The gaussx-backed surface is optional; the diagonal/IVON core is not.

Simulates a missing gaussx install by blocking the import via
``sys.modules`` and checks that (a) the slim core keeps working and
(b) the gaussx-backed entry points raise an informative ImportError.
"""

from __future__ import annotations

import sys

import jax.numpy as jnp
import optax
import pytest

import optax_bayes
from optax_bayes._src._optional import require_gaussx


@pytest.fixture
def no_gaussx(monkeypatch):
    """Make ``import gaussx`` raise ImportError for the test's duration."""
    monkeypatch.setitem(sys.modules, "gaussx", None)


class TestRequireGaussx:
    def test_returns_module_when_installed(self):
        mod = require_gaussx("test feature")
        assert mod.__name__ == "gaussx"

    def test_raises_with_install_hint(self, no_gaussx):
        with pytest.raises(ImportError, match=r"optax_bayes\[gaussx\]"):
            require_gaussx("test feature")

    def test_error_names_the_feature(self, no_gaussx):
        with pytest.raises(ImportError, match="blr_full_rank"):
            optax_bayes.blr_full_rank()


class TestGuardedEntryPoints:
    def test_blr_full_rank_factory_raises(self, no_gaussx):
        with pytest.raises(ImportError):
            optax_bayes.blr_full_rank()

    def test_blr_low_rank_factory_raises(self, no_gaussx):
        with pytest.raises(ImportError):
            optax_bayes.blr_low_rank()

    def test_full_rank_for_loss_raises(self, no_gaussx):
        with pytest.raises(ImportError):
            optax_bayes.blr_full_rank_for_loss()

    def test_posterior_extraction_raises(self, no_gaussx):
        state = optax_bayes.BLRFullRankState(
            precision=jnp.eye(2),
            nat_mean=jnp.zeros(2),
            count=jnp.zeros([], jnp.int32),
        )
        with pytest.raises(ImportError):
            optax_bayes.get_posterior_full_rank(state)

    def test_sampling_raises(self, no_gaussx):
        import jax

        state = optax_bayes.BLRFullRankState(
            precision=jnp.eye(2),
            nat_mean=jnp.zeros(2),
            count=jnp.zeros([], jnp.int32),
        )
        with pytest.raises(ImportError):
            optax_bayes.sample_posterior_full_rank(state, jax.random.key(0))


class TestSlimCoreWithoutGaussx:
    def test_diagonal_training_step_works(self, no_gaussx):
        params = jnp.array([1.0, -2.0, 3.0])
        opt = optax_bayes.blr_diagonal_for_loss(learning_rate=0.1)
        state = opt.init(params)
        updates, state = opt.update(params, state)
        params = optax.apply_updates(params, updates)
        mean, variance = optax_bayes.get_posterior_diagonal(state)
        assert params.shape == (3,)
        assert mean.shape == (3,)
        assert jnp.all(variance > 0)

    def test_ivon_works(self, no_gaussx):
        params = jnp.array([1.0, -2.0])
        opt = optax_bayes.ivon(learning_rate=0.1)
        state = opt.init(params)
        updates, state = opt.update(params, state, params)
        assert updates.shape == (2,)

    def test_diagonal_sampling_works(self, no_gaussx):
        import jax

        opt = optax_bayes.blr_diagonal(learning_rate=0.1)
        state = opt.init(jnp.ones(4))
        sample = optax_bayes.sample_posterior_diagonal(state, jax.random.key(0))
        assert sample.shape == (4,)
