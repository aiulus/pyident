# pyident/cli.py
from __future__ import annotations
import argparse, json
from .config import ExpConfig, SolverOpts
from .run_single import run_single

def parse_args():
    p = argparse.ArgumentParser(description="pyident experiments")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--T", type=int, default=400)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--ensemble", type=str, default="ginibre", choices=["ginibre","sparse","stable","binary"])
    p.add_argument("--p_density", type=float, default=0.8)
    p.add_argument("--x0_mode", type=str, default="gaussian", choices=["gaussian","rademacher","ones","zero"])
    p.add_argument("--signal", type=str, default="prbs", choices=["prbs","multisine"])
    p.add_argument("--PE_order", type=int, default=12)
    p.add_argument("--pointwise_W_dim", type=int, default=None)
    p.add_argument("--moment_pe_r", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--estimators", type=str, default="dmdc,moesp")
    return p.parse_args()

def main():
    a = parse_args()
    cfg = ExpConfig(
        n=a.n, m=a.m, horizon=a.T, dt=a.dt,
        ensemble=a.ensemble, sparsity_p=a.p_density, x0_mode=a.x0_mode,
        signal=a.signal, PE_order=a.PE_order,
        pointwise_W_dim=a.pointwise_W_dim,
        moment_pe_r=a.moment_pe_r,
        estimators=tuple(a.estimators.split(","))
    )
    sopts = SolverOpts()
    out = run_single(cfg, seed=a.seed, sopts=sopts, estimators=cfg.estimators)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
