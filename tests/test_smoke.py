import numpy as np
from ..config import ExpConfig, SolverOpts
from ..run_single import run_single

def test_run_single_smoke():
    cfg = ExpConfig(n=4, m=2, T=60, dt=0.05, ensemble="ginibre", signal="prbs", sigPE=12)
    out = run_single(cfg, seed=0, sopts=SolverOpts(), algs=("dmdc",), use_jax=False)
    assert out["n"] == 4 and out["m"] == 2
    assert isinstance(out["K_rank"], int) and 0 <= out["K_rank"] <= out["n"]
    assert out["gram_mode"] in {"CT","DT-infinite","DT-finite","none"}
    assert "delta_pbh" in out and np.isfinite(out["delta_pbh"])
    assert "estimators" in out and "dmdc" in out["estimators"]

def test_numpy_vs_jax_parity_if_available():
    try:
        import jax  # noqa
    except Exception:
        return  # skip if no jax
    cfg = ExpConfig(n=4, m=2, T=60, dt=0.05, ensemble="ginibre", signal="prbs", sigPE=12)
    out_np = run_single(cfg, seed=0, sopts=SolverOpts(), algs=("dmdc",), use_jax=False)
    out_jx = run_single(cfg, seed=0, sopts=SolverOpts(), algs=("dmdc",), use_jax=True)
    # Same qualitative outputs
    assert out_np["K_rank"] == out_jx["K_rank"]
    # PBH and gram_min need not be identical, but should be close
    assert abs(out_np["delta_pbh"] - out_jx["delta_pbh"]) <= 1e-8
