"""Tests for the precision linear algebra behind the gaussx-backed transforms."""

from __future__ import annotations

import gaussx
import jax.numpy as jnp
import lineax as lx
import pytest

from optax_bayes import (
    blr_full_rank,
    blr_low_rank,
    get_posterior_full_rank,
    get_posterior_low_rank,
    newton,
)
from optax_bayes._src.linalg import low_rank_precision_operator, precision_operator


SOLVERS = [
    pytest.param(None, id="default"),
    pytest.param(lx.Cholesky(), id="lx.Cholesky"),
    pytest.param(gaussx.DenseSolver(), id="gaussx.DenseSolver"),
    pytest.param(gaussx.CGSolver(rtol=1e-10, atol=1e-10), id="gaussx.CGSolver"),
]


def _run(opt, params, grads, n_steps=3):
    state = opt.init(params)
    for _ in range(n_steps):
        updates, state = opt.update(grads, state, params)
        params = params + updates
    return params, state


class TestOperatorTags:
    def test_dense_precision_is_psd(self):
        op = precision_operator(2.0 * jnp.eye(3))
        assert lx.is_positive_semidefinite(op)

    def test_dense_precision_untagged_on_request(self):
        op = precision_operator(2.0 * jnp.eye(3), psd=False)
        assert not lx.is_positive_semidefinite(op)

    def test_low_rank_precision_is_psd(self):
        op = low_rank_precision_operator(jnp.ones(4), jnp.ones((4, 2)))
        assert lx.is_positive_semidefinite(op)


class TestSolverArgument:
    """Every documented solver kind runs and agrees with the default."""

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_full_rank(self, solver):
        params, grads = jnp.ones(4), jnp.arange(4.0)
        ref, ref_state = _run(blr_full_rank(), params, grads)
        out, state = _run(blr_full_rank(solver=solver), params, grads)
        assert jnp.allclose(out, ref, atol=1e-8)
        mean, cov = get_posterior_full_rank(state, solver=solver)
        ref_mean, ref_cov = get_posterior_full_rank(ref_state)
        assert jnp.allclose(mean, ref_mean, atol=1e-8)
        assert jnp.allclose(cov, ref_cov, rtol=1e-6)

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_low_rank(self, solver):
        params, grads = jnp.ones(4), jnp.arange(4.0)
        ref, ref_state = _run(blr_low_rank(rank=2), params, grads)
        out, state = _run(blr_low_rank(rank=2, solver=solver), params, grads)
        assert jnp.allclose(out, ref, atol=1e-8)
        mean, cov = get_posterior_low_rank(state, solver=solver)
        ref_mean, ref_cov = get_posterior_low_rank(ref_state)
        assert jnp.allclose(mean, ref_mean, atol=1e-8)
        assert jnp.allclose(cov, ref_cov, rtol=1e-6)


class TestNewtonIndefinite:
    def test_saddle_step_is_finite(self):
        """Newton keeps a general solve, so an indefinite Hessian still works."""
        hess = jnp.diag(jnp.array([-2.0, 1.0]))
        opt = newton(hessian_fn=lambda mean, grads: hess)
        params = jnp.array([1.0, 1.0])
        state = opt.init(params)
        grads = hess @ params
        updates, _ = opt.update(grads, state, params)
        assert jnp.all(jnp.isfinite(updates))
        # One Newton step on a quadratic lands on its stationary point.
        assert jnp.allclose(params + updates, jnp.zeros(2), atol=1e-4)
