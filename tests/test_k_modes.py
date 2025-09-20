import numpy as np
from ..config import ExpConfig, SolverOpts
from ..run_single import run_single

def test_k_mode_pointwise_monotone_in_q():
    n, m = 6, 4
    cfg_un = ExpConfig(n=n, m=m, T=60, dt=0.05, ensemble="ginibre", signal="prbs", sigPE=12)
    out_un = run_single(cfg_un, seed=0, sopts=SolverOpts(), algs=("dmdc",), use_jax=False)

    q1, q2 = 1, 3
    U1 = np.eye(m)[:, :q1]
    U2 = np.eye(m)[:, :q2]
    cfg_q1 = ExpConfig(n=n, m=m, T=60, dt=0.05, ensemble="ginibre", signal="prbs", sigPE=12, U_restr=U1)
    cfg_q2 = ExpConfig(n=n, m=m, T=60, dt=0.05, ensemble="ginibre", signal="prbs", sigPE=12, U_restr=U2)

    out_q1 = run_single(cfg_q1, seed=0, sopts=SolverOpts(), algs=("dmdc",), use_jax=False)
    out_q2 = run_single(cfg_q2, seed=0, sopts=SolverOpts(), algs=("dmdc",), use_jax=False)

    assert out_q1["K_mode"] == "pointwise" and out_q2["K_mode"] == "pointwise"
    assert out_q1["K_rank"] <= out_q2["K_rank"] <= out_un["K_rank"] <= n

def test_k_mode_moment_pe_monotone_in_r():
    n, m = 6, 3
    cfg_un = ExpConfig(n=n, m=m, T=60, dt=0.05, ensemble="ginibre", signal="prbs", sigPE=12)
    out_un = run_single(cfg_un, seed=0, sopts=SolverOpts(), algs=("dmdc",), use_jax=False)

    r1, r2 = 1, 3
    cfg_r1 = ExpConfig(n=n, m=m, T=60, dt=0.05, ensemble="ginibre", signal="prbs", sigPE=12, PE_r=r1)
    cfg_r2 = ExpConfig(n=n, m=m, T=60, dt=0.05, ensemble="ginibre", signal="prbs", sigPE=12, PE_r=r2)

    out_r1 = run_single(cfg_r1, seed=0, sopts=SolverOpts(), algs=("dmdc",), use_jax=False)
    out_r2 = run_single(cfg_r2, seed=0, sopts=SolverOpts(), algs=("dmdc",), use_jax=False)

    assert out_r1["K_mode"] == "moment-pe" and out_r2["K_mode"] == "moment-pe"
    assert out_r1["K_rank"] <= out_r2["K_rank"] <= out_un["K_rank"] <= n
