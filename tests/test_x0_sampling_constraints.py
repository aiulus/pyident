import numpy as np
import pytest

from pyident.experiments.sim_unctrb_x0_boxplot import sample_masked_sphere


def test_sample_masked_sphere_nonzero_with_support(rng):
    x0 = sample_masked_sphere(
        n=8,
        rng=rng,
        p_keep=1.0,
        renorm=True,
        min_norm=1e-6,
        min_support=1,
        max_attempts=4,
    )
    assert np.linalg.norm(x0) > 0.0


def test_sample_masked_sphere_raises_when_impossible(rng):
    with pytest.raises(RuntimeError):
        sample_masked_sphere(
            n=5,
            rng=rng,
            p_keep=0.0,
            renorm=True,
            min_norm=1e-6,
            min_support=1,
            max_attempts=4,
        )
