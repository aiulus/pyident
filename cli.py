# pyident/cli.py
from __future__ import annotations
import argparse
import json
import numpy as np

from .config import ExpConfig, SolverOpts
from .run_single import run_single


def parse_args():
    p = argparse.ArgumentParser(description="pyident experiments")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--T", type=int, default=400)
    p.add_argument("--dt", type=float, default=0.05)

    p.add_argument("--ensemble", type=str, default="ginibre",
                   choices=["ginibre", "sparse", "stable", "binary"])
    p.add_argument("--p_density", type=float, default=0.8,
                   help="Nonzero fraction for A (and B if sparse_which='both' and p_density_B unset).")
    p.add_argument("--sparse_which", type=str, default="both",
                   choices=["A", "B", "both"])
    p.add_argument("--p_density_B", type=float, default=None,
                   help="Optional B density (if sparse_which includes B).")

    p.add_argument("--x0_mode", type=str, default="gaussian",
                   choices=["gaussian", "rademacher", "ones", "zero"])

    p.add_argument("--signal", type=str, default="prbs",
                   choices=["prbs", "multisine"])
    p.add_argument("--pe_order_target", type=int, default=12,
                   help="Targeted richness for signal design.")

    # Pointwise admissibility: restrict to span(W) with q directions
    p.add_argument("--U_restr_dim", type=int, default=None,
                   help="If set, build W = I_m[:, :q] and project inputs onto span(W).")

    # Nonlocal admissibility: moment-PE order r for analysis (not for generation)
    p.add_argument("--PE_r", type=int, default=None,
                   help="If set, analyze identifiability with moment-PE order r.")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--estimators", type=str, default="dmdc,moesp",
                   help="Comma-separated list, e.g. 'dmdc,moesp'.")

    return p.parse_args()


def main():
    a = parse_args()

    # Build U_restr if requested (canonical subspace spanned by first q basis vectors)
    U_restr = None
    if a.U_restr_dim is not None:
        q = int(a.U_restr_dim)
        if q < 1 or q > a.m:
            raise ValueError(f"--U_restr_dim must be in [1, m]={a.m}, got {q}.")
        U_restr = np.eye(a.m)[:, :q]

    cfg = ExpConfig(
        n=a.n,
        m=a.m,
        T=a.T,
        dt=a.dt,
        ensemble=a.ensemble,
        p_density=a.p_density,
        sparse_which=a.sparse_which,
        p_density_A=None,              # use p_density for A unless changed later
        p_density_B=a.p_density_B,     # optional separate B density
        x0_mode=a.x0_mode,
        signal=a.signal,
        pe_order_target=a.pe_order_target,
        U_restr=U_restr,
        PE_r=a.PE_r,
        estimators=tuple(s.strip() for s in a.estimators.split(",") if s.strip()),
        light=True,
    )
    sopts = SolverOpts()

    out = run_single(cfg, seed=a.seed, sopts=sopts, estimators=cfg.estimators)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
