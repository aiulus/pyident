"""Tests for :mod:`pyident.experiments.sim_combreg`."""

from __future__ import annotations

import numpy as np

from ..experiments.sim_regcomb import parse_grid


def test_parse_grid_range_does_not_overshoot_stop() -> None:
    values = parse_grid("0.1:0.2:1.0")

    # The final value should not exceed the requested stop value.
    assert np.all(values <= 1.0 + 1e-12)

    # Ensure we still get the expected regularly spaced values.
    expected = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9])
    assert np.allclose(values, expected)