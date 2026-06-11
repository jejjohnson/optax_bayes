"""Optional-dependency handling.

``gaussx`` backs the full-rank and low-rank transforms (structured
operators, Woodbury-dispatched solves).  The diagonal and IVON surface
must stay importable and usable without it, so the gaussx-backed modules
import it lazily through :func:`require_gaussx`.
"""

from __future__ import annotations

from types import ModuleType


def require_gaussx(feature: str) -> ModuleType:
    """Import and return ``gaussx``, or raise an informative error.

    Args:
        feature: Name of the public API requiring gaussx, used in the
            error message (e.g. ``"blr_full_rank"``).

    Returns:
        The imported ``gaussx`` module.

    Raises:
        ImportError: If gaussx is not installed.
    """
    try:
        import gaussx
    except ImportError as err:
        raise ImportError(
            f"{feature} requires the optional dependency 'gaussx'. "
            "Install it with:\n"
            '    pip install "optax_bayes[gaussx]"'
        ) from err
    return gaussx
