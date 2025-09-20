import numpy as np
import pytest
from statistics import median
from .helpers import run_many, med, assert_monotone_nondec
from ..config import ExpConfig

@pytest.mark.parametrize("mvals", [(1,2,3,4)])
def test_underactuation_trend_Krank_and_PBH(mvals, seeds, sopts):
    # Expect K_rank↑ and delta_pbh↑ with more actuation (median over seeds)
    meds_rank, meds_pbh = [], []
    for m in mvals:
        cfg = ExpConfig(n=4, m=m, T=60, dt=0.05, ensemble="ginibre", signal="prbs", sigPE=12)
        outs = run_many(cfg, seeds, sopts, algs=("dmdc",), use_jax=False)
        meds_rank.append(med(outs, "K_rank"))
        meds_pbh.append(med(outs, "delta_pbh"))
    assert_monotone_nondec(meds_rank, slack=0.0)
    assert_monotone_nondec(meds_pbh, slack=1e-10)

@pytest.mark.parametrize("densities", [(0.2,0.4,0.6,0.8)])
def test_sparsity_trend(densities, seeds, sopts):
    # More dense (higher p_density) should improve K_rank and PBH margin
    meds_rank, meds_pbh = [], []
    for p in densities:
        cfg = ExpConfig(n=6, m=2, T=60, dt=0.05, ensemble="sparse", signal="prbs",
                        sigPE=12, p_density=p, sparse_which="both")
        outs = run_many(cfg, seeds, sopts, algs=("dmdc",), use_jax=False)
        meds_rank.append(med(outs, "K_rank"))
        meds_pbh.append(med(outs, "delta_pbh"))
    assert_monotone_nondec(meds_rank, slack=0.0)
    assert_monotone_nondec(meds_pbh, slack=1e-10)

def test_lower_pe_order_hurts_identifiability(seeds, sopts):
    # Reduce effective PE by using few PRBS transitions (small sigPE) -> worse PBH median
    cfg_hi = ExpConfig(n=6, m=2, T=120, dt=0.05, ensemble="ginibre", signal="prbs", sigPE=31)
    cfg_lo = ExpConfig(n=6, m=2, T=120, dt=0.05, ensemble="ginibre", signal="prbs", sigPE=4)
    hi = run_many(cfg_hi, seeds, sopts, algs=("dmdc",), use_jax=False)
    lo = run_many(cfg_lo, seeds, sopts, algs=("dmdc",), use_jax=False)
    assert med(hi, "delta_pbh") >= med(lo, "delta_pbh") - 1e-10
