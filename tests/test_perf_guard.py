import os, time
import numpy as np
import pytest
from ..config import ExpConfig, SolverOpts
from ..run_single import run_single

@pytest.mark.slow
def test_many_runs_under_budget():
    # Skip on known slow CI environments
    if os.getenv("CI", "") or os.getenv("GITHUB_ACTIONS", ""):
        pytest.skip("Skip perf guard on CI")
    cfg = ExpConfig(n=4, m=2, T=50, dt=0.05, ensemble="ginibre", signal="prbs", sigPE=12)
    sopts = SolverOpts()
    seeds = list(range(8))
    t0 = time.perf_counter()
    outs = [run_single(cfg, seed=s, sopts=sopts, algs=("dmdc",), use_jax=False) for s in seeds]
    dt = time.perf_counter() - t0
    assert dt < 10.0  # generous budget for local laptop
    assert all("K_rank" in o for o in outs)
