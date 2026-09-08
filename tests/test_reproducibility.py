"""Seeded NumPy draws must match after set_seed(42) is applied twice."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from ml.src.utils import set_seed  # noqa: E402


def test_set_seed_numpy_reproducible() -> None:
    set_seed(42)
    first = np.random.rand(16)
    set_seed(42)
    second = np.random.rand(16)
    assert np.array_equal(first, second)
