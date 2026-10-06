import os

import jax
import pytest


# The suite runs with x64 on by default. OPTAX_BAYES_TEST_X64=0 runs it in
# JAX's default configuration (float32, no float64) instead: the no-x64 CI lane.
X64 = os.environ.get("OPTAX_BAYES_TEST_X64", "1") != "0"
jax.config.update("jax_enable_x64", X64)


def pytest_collection_modifyitems(config, items):
    """Skip ``x64_only`` tests when the lane runs without x64.

    Every ``x64_only`` marker must say why the test needs float64, as
    ``@pytest.mark.x64_only(reason="...")``; one without a reason is an error
    in both lanes, so it cannot slip in through the x64 lane.
    """
    for item in items:
        marker = item.get_closest_marker("x64_only")
        if marker is None:
            continue
        reason = marker.kwargs.get("reason")
        if not reason:
            raise pytest.UsageError(
                f'{item.nodeid}: x64_only needs a reason, as x64_only(reason="...")'
            )
        if not X64:
            item.add_marker(pytest.mark.skip(reason=f"x64_only: {reason}"))
