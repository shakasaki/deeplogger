"""Shared pytest configuration and fixtures for DeepLogger tests."""

import matplotlib
import pytest

# Use non-interactive backend so tests don't pop up windows
matplotlib.use("Agg")


@pytest.fixture
def sample_identity_image():
    """A simple 20x20 identity matrix for filter testing."""
    import numpy as np
    return np.eye(20, 20)
