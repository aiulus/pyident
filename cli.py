from __future__ import annotations
import argparse
import json
import os
import numpy as np

from .config import ExpConfig, SolverOpts
from .run_single import run_single
from  .experiments.sparsity import sweep_sparsity
from .experiments.underactuation import sweep_underactuation 


# ------------------------
# small parsing utilities
# ------------------------
def _parse_int_list(s: str) -> list[int]:
    """
    Accepts:
      - comma list: "1,2,3"
      - range: "0:10" (0..9), or "0:10:2" (step=2)
    """
    s = s.strip()
    if ":" in s:
        parts = [int(x) for x in s.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = 1
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError(f"Bad range '{s}'. Use 'start:stop[:step]'.")
        return list(range(start, stop, step))
    return [int(x) for x in s.split(",") if x.strip()]

def _parse_float_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _add_common_single_args(p: argparse.ArgumentParser) -> None:
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
    p.add_argument("--sigPE", type=int, default=12,
                   help="Targeted richness for signal design.")

    # Pointwise admissibility: restrict to span(W) with q directions
    p.add_argument("--U_restr_dim", type=int, default=None,
                   help="If set, build W = I_m[:, :q] and project inputs onto span(W).")

    # Nonlocal admissibility: moment-PE order r for analysis (not for generation)
    p.add_argument("--PE_r", type=int, default=None,
                   help="If set, analyze identifiability with moment-PE order r.")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--algs", type=str, default="dmdc,moesp",
                   help="Comma-separated list, e.g. 'dmdc,moesp,dmdc_tls,dmdc_iv'.")


def parse_args():
    p = argparse.ArgumentParser(description="pyident experiments")
    sub = p.add_subparsers(dest="cmd")

    # -------- single --------
    ps = sub.add_parser("single", help="Run a single experiment (default).")
    _add_common_single_args(ps)

    # -------- sweep-underactuation --------
    pu = sub.add_parser("sweep-underactuation", help="Sweep over m (underactuation study).")
    pu.add_argument("--n", type=int, default=10)
    pu.add_argument("--m-values", type=str, default="1,2,3,4,5,6,8,10",
                    help="List or range, e.g. '1,2,3' or '1:11'.")
    pu.add_argument("--T", type=int, default=400)
    pu.add_argument("--dt", type=float, default=0.05)
    pu.add_argument("--ensemble", type=str, default="ginibre",
                    choices=["ginibre", "sparse", "stable", "binary"])
    pu.add_argument("--signal", type=str, default="prbs", choices=["prbs", "multisine"])
    pu.add_argument("--sigPE", type=int, default=12)
    pu.add_argument("--seeds", type=str, default="0:50", help="e.g. '0:50' or '0,1,2'")
    pu.add_argument("--algs", type=str, default="dmdc,moesp,dmdc_tls,dmdc_iv")
    pu.add_argument("--out-csv", type=str, default="results_underactuation.csv")

    # Optional admissibility for the whole sweep
    pu.add_argument("--U_restr_dim", type=int, default=None)
    pu.add_argument("--PE_r", type=int, default=None)

    # -------- sweep-sparsity --------
    psr = sub.add_parser("sweep-sparsity", help="Sweep over sparsity levels.")
    psr.add_argument("--n", type=int, default=10)
    psr.add_argument("--m", type=int, default=3)
    psr.add_argument("--p-values", type=str, default="1.0,0.9,0.8,0.6,0.4,0.2",
                     help="Comma list of densities (0..1).")
    psr.add_argument("--T", type=int, default=400)
    psr.add_argument("--dt", type=float, default=0.05)
    psr.add_argument("--sparse-which", type=str, default="both", choices=["A", "B", "both"])
    psr.add_argument("--signal", type=str, default="prbs", choices=["prbs", "multisine"])
    psr.add_argument("--sigPE", type=int, default=12)
    psr.add_argument("--seeds", type=str, default="0:50", help="e.g. '0:50' or '0,1,2'")
    psr.add_argument("--algs", type=str, default="dmdc,moesp,dmdc_tls,dmdc_iv")
    psr.add_argument("--out-csv", type=str, default="results_sparsity.csv")

    # If no subcommand, fall back to single
    p.set_defaults(cmd="single")
    return p.parse_args()


def _build_U_restr(m: int, q: int | None) -> np.ndarray | None:
    if q is None:
        return None
    if q < 1 or q > m:
        raise ValueError(f"--U_restr_dim must be in [1, m]={m}, got {q}.")
    return np.eye(m)[:, :q]


def main():
    a = parse_args()

    if a.cmd == "single":
        # Build U_restr if requested (canonical subspace spanned by first q basis vectors)
        U_restr = _build_U_restr(a.m, a.U_restr_dim)

        cfg = ExpConfig(
            n=a.n,
            m=a.m,
            T=a.T,
            dt=a.dt,
            ensemble=a.ensemble,
            p_density=a.p_density,
            sparse_which=a.sparse_which,
            p_density_A=None,
            p_density_B=a.p_density_B,
            x0_mode=a.x0_mode,
            signal=a.signal,
            sigPE=a.sigPE,
            U_restr=U_restr,
            PE_r=a.PE_r,
            algs=tuple(s.strip() for s in a.algs.split(",") if s.strip()),
            light=True,
        )
        sopts = SolverOpts()
        out = run_single(cfg, seed=a.seed, sopts=sopts, algs=cfg.algs)
        print(json.dumps(out, indent=2))
        return

    if a.cmd == "sweep-underactuation":
        m_values = _parse_int_list(a.m_values)
        seeds = _parse_int_list(a.seeds)
        algs = tuple(s.strip() for s in a.algs.split(",") if s.strip())

        # Ensure output dir exists
        os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)

        sweep_underactuation(
            n=a.n,
            m_values=m_values,
            T=a.T,
            dt=a.dt,
            ensemble=a.ensemble,
            signal=a.signal,
            sigPE=a.sigPE,
            seeds=seeds,
            algs=algs,
            out_csv=a.out_csv,
        )
        print(f"Wrote {a.out_csv}")
        return

    if a.cmd == "sweep-sparsity":
        p_values = _parse_float_list(a.p_values)
        seeds = _parse_int_list(a.seeds)
        algs = tuple(s.strip() for s in a.algs.split(",") if s.strip())

        os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)

        sweep_sparsity(
            n=a.n,
            m=a.m,
            p_values=p_values,
            T=a.T,
            dt=a.dt,
            sparse_which=a.sparse_which,
            signal=a.signal,
            sigPE=a.sigPE,
            seeds=seeds,
            algs=algs,
            out_csv=a.out_csv,
        )
        print(f"Wrote {a.out_csv}")
        return


if __name__ == "__main__":
    main()
