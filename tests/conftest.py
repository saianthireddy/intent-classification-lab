"""Shared fixtures. Nothing here touches the network."""
from __future__ import annotations

import pytest

from intent_lab.data import build_dataset


@pytest.fixture(scope="session")
def dataset():
    return build_dataset()


@pytest.fixture(scope="session")
def small_dataset():
    """Fewer renderings per phrase — same structure, faster to train in tests."""
    return build_dataset(variants_per_phrase=2)
