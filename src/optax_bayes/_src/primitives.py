"""Diagonal BLR update equations — pure JAX functions.

Implements the Bayesian Learning Rule (Khan & Rue, 2023) specialised to
diagonal Gaussians.  All functions operate on leaf arrays, not pytrees.
"""

from __future__ import annotations

from jaxtyping import Array, Float


def blr_diag_update_step(
    eta: Float[Array, ...],
    s: Float[Array, ...],
    g: Float[Array, ...],
    diag_h: Float[Array, ...],
    m: Float[Array, ...],
    eta0: Float[Array, ...],
    s0: Float[Array, ...],
    rho: float,
) -> tuple[Float[Array, ...], Float[Array, ...], Float[Array, ...]]:
    """One diagonal BLR update step on leaf arrays.

    Implements the natural-parameter update:

    .. code-block:: text

        s_next   = (1 - rho) * s   + rho * (s0   - diag_h)
        eta_next = (1 - rho) * eta + rho * (eta0 + g - diag_h * m)
        m_next   = eta_next / s_next

    The ``-diag_h * m`` correction is the Bonnet-Price identity term;
    omitting it yields incorrect fixed points.

    Args:
        eta: Current natural mean, eta = s * m.
        s: Current precision (inverse variance).
        g: Log-likelihood gradient, g = nabla_theta ell(theta).
        diag_h: Diagonal Hessian estimate (elementwise <= 0 for concave ell).
        m: Current mean, m = eta / s.
        eta0: Prior natural mean, eta0 = s0 * m0.
        s0: Prior precision.
        rho: Learning rate in (0, 1].

    Returns:
        Tuple ``(eta_next, s_next, m_next)``.
    """
    s_next = (1 - rho) * s + rho * (s0 - diag_h)
    eta_next = (1 - rho) * eta + rho * (eta0 + g - diag_h * m)
    m_next = eta_next / s_next
    return eta_next, s_next, m_next
