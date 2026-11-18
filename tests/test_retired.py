"""optax_bayes is retired: importing it warns and names argminax.blr."""

import importlib

import pytest

import optax_bayes


def test_import_warns_that_the_package_moved_to_argminax_blr():
    with pytest.warns(DeprecationWarning, match=r"argminax\.blr") as record:
        importlib.reload(optax_bayes)
    assert "optax_bayes is retired" in str(record[0].message)


def test_the_transforms_still_work_after_the_warning():
    optimiser = optax_bayes.blr_diagonal_for_loss(0.1)
    assert callable(optimiser.update)
