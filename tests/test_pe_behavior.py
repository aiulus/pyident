import numpy as np
from statistics import median
from ..signals import prbs, multisine, restrict_pointwise, estimate_pe_order

def test_prbs_pe_increases_with_period():
    rng = np.random.default_rng(0)
    T, m = 256, 2
    periods = [7, 15, 31, 63]
    outs = []
    for p in periods:
        pe = []
        for s in range(10):
            u = prbs(T, m, np.random.default_rng(s), period=p)
            pe.append(estimate_pe_order(u, s_max=T // 2, tol=1e-8))
        outs.append(median(pe))
    assert all(outs[i] <= outs[i+1] for i in range(len(outs)-1))

def test_multisine_pe_increases_with_k_lines_and_restriction_caps_it():
    rng = np.random.default_rng(1)
    T, m = 256, 3
    ks = [2, 4, 8]
    meds = []
    for k in ks:
        vals = []
        for s in range(8):
            u = multisine(T, m, np.random.default_rng(s), k_lines=k)
            vals.append(estimate_pe_order(u, s_max=T//2, tol=1e-8))
        meds.append(median(vals))
    assert all(meds[i] <= meds[i+1] for i in range(len(meds)-1))

    # Restrict to q < m dims
    q = 2
    W = np.eye(m)[:, :q]
    u = multisine(T, m, rng, k_lines=8)
    ur = restrict_pointwise(u, W)
    pe_full = estimate_pe_order(u, s_max=T//2, tol=1e-8)
    pe_rest = estimate_pe_order(ur, s_max=T//2, tol=1e-8)
    assert pe_rest <= pe_full and pe_rest <= q
