import numpy as np
import pytest

from pyident.experiments.unctrb_utils import (
    condition_number,
    generate_pe_input,
    min_T_for_block_pe,
    pe_order_from_nmax,
)


def test_pe_order_from_nmax():
    assert pe_order_from_nmax(10) == 21


def test_min_T_for_block_pe():
    assert min_T_for_block_pe(5, 2) == 14


def test_generate_pe_input_too_short_raises(rng):
    pe_order = 5
    m = 2
    min_T = min_T_for_block_pe(pe_order, m)
    with pytest.raises(ValueError):
        generate_pe_input(
            T=min_T - 1,
            m=m,
            dt=1.0,
            dwell=1,
            rng=rng,
            pe_order=pe_order,
            family="prbs",
            pe_method="block",
        )


def test_condition_number_rank_deficient():
    Z = np.zeros((3, 3))
    assert np.isinf(condition_number(Z))
