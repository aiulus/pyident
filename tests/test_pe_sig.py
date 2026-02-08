import numpy as np

from pyident.pe_sig import PRBSSpec, MultisineSpec, generate_pe_signal
from pyident.signals import estimate_pe_order


def test_pe_sig_prbs_block_pe(rng):
    T = 256
    m = 2
    dt = 0.05
    pe_order = 4

    bundle = generate_pe_signal(
        family="prbs",
        T=T,
        m=m,
        dt=dt,
        pe_order=pe_order,
        rng=rng,
        prbs=PRBSSpec(register=7, clock=dt, taps=[7, 1]),
        ensure_pe=True,
        pe_method="block",
        max_tries=16,
    )

    assert bundle.u.shape == (T, m)
    assert bundle.pe_order_est is not None
    assert bundle.pe_order_est >= pe_order
    assert estimate_pe_order(bundle.u, s_max=pe_order) >= pe_order


def test_pe_sig_multisine_block_pe(rng):
    T = 256
    m = 2
    dt = 0.05
    pe_order = 4

    bundle = generate_pe_signal(
        family="multisine",
        T=T,
        m=m,
        dt=dt,
        pe_order=pe_order,
        rng=rng,
        multisine=MultisineSpec(k_lines=6),
        ensure_pe=True,
        pe_method="block",
        max_tries=32,
    )

    assert bundle.u.shape == (T, m)
    assert bundle.pe_order_est is not None
    assert bundle.pe_order_est >= pe_order
    assert estimate_pe_order(bundle.u, s_max=pe_order) >= pe_order
