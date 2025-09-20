import numpy as np
from pyident.config import ExpConfig, SolverOpts
from pyident.run_single import run_single

def test_k_mode_pointwise_caps_rank():
    n, m, q = 6, 4, 2
    cfg = ExpConfig(n=n, m=m, T=60, dt=0.05, ensemble="ginibre", signal="prbs",
                    sigPE=12, U_restr=np.eye(m)[:, :q])
    out = run_single(cfg, seed=0, sopts=SolverOpts(), algs=("dmdc",), use_jax=False)
    assert out["K_mode"] == "pointwise"
    assert out["K_rank"] <= q
    # estimator projected errors exist
    assert "dmdc" in out["estimators"]

def test_k_mode_moment_pe_caps_rank():
    n, m, r = 6, 3, 3
    cfg = ExpConfig(n=n, m=m, T=60, dt=0.05, ensemble="ginibre", signal="prbs",
                    sigPE=12, PE_r=r)
    out = run_single(cfg, seed=0, sopts=SolverOpts(), algs=("dmdc",), use_jax=False)
    assert out["K_mode"] == "moment-pe"
    assert out["K_rank"] <= r
    assert "dmdc" in out["estimators"]
