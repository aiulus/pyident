#!/usr/bin/env python3
"""
Unidentifiability vs. input dimension m (continuous-time, sparse ensemble).

(1) For each m in {1,2,...,10}, generate N=200 random (A,B) with the
    **sparse_continuous** ensemble at fixed density p=0.8 and state dim n=5.
    Sample x0 ~ Unif(S^{n-1}).
(2) Compute binary criteria with tolerance `--tol`:
      - PBH margin: d_pbh = pbh_margin_structured(A,B,x0) → unident_pbh = (d_pbh <= tol)
      - Left-eigen: mu_min = min_i μ_i(A,[x0 B])          → unident_mu  = (mu_min <= tol)
      - Union: unident_any = unident_pbh OR unident_mu
(3) Plot % unidentifiable pairs (y-axis) vs input dimension m (x-axis),
    with the union as the main line and the two individual criteria as dotted lines.
"""
from __future__ import annotations
import argparse, math, sys, pathlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from ..metrics import(
    pbh_margin_structured,
    unified_generator,
    left_eigvec_overlap,
    cont2discrete_zoh
)
from ..estimators import(
    dmdc_pinv, 
    moesp_fit,
    sindy_fit,
    node_fit
)
from ..ensembles import draw_with_ctrb_rank, sparse_continuous  
from ..metrics import pair_distance

def unit(v):
    n = np.linalg.norm(v)
    return v / (n if n>0 else 1.0)

def try_sparse_continuous(n, m, p, rng):
    return sparse_continuous(n=n, m=m, which="both", rng=rng, p_density=p)

def run(n=5, m_min=1, m_max=10, trials_per_m=200, density=0.8, tol=1e-12,
        seed=123, outdir="unident_vs_udim"):
    rng = np.random.default_rng(seed)
    out = pathlib.Path(outdir); (out/"plots").mkdir(parents=True, exist_ok=True)

    rows = []
    for m in range(m_min, m_max+1):
        n_any = n_pbh = n_mu = 0
        for _ in range(trials_per_m):
            A, B = try_sparse_continuous(n, m, density, rng)
            x0 = unit(rng.standard_normal(n))

            # Metrics
            d_pbh = float(pbh_margin_structured(A, B, x0))
            Xaug  = np.concatenate([x0.reshape(-1,1), B], axis=1)
            mu    = left_eigvec_overlap(A, Xaug)
            mu_min = float(np.min(mu)) if mu.size else 0.0

            # Binary flags
            f_pbh = (d_pbh <= tol)
            f_mu  = (mu_min <= tol)
            f_any = bool(f_pbh or f_mu)

            n_pbh += int(f_pbh); n_mu += int(f_mu); n_any += int(f_any)

        rows.append(dict(
            m=m,
            n=n,
            density=density,
            N=trials_per_m,
            unident_pbh=n_pbh,
            unident_mu=n_mu,
            unident_any=n_any,
            pct_pbh=100.0*n_pbh/trials_per_m,
            pct_mu=100.0*n_mu/trials_per_m,
            pct_any=100.0*n_any/trials_per_m,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(out/"summary.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(df["m"], df["pct_any"], marker="o", label="Zero PBH OR Zero μ_min")
    ax.plot(df["m"], df["pct_pbh"], marker=".", linestyle="--", label="Zero PBH only")
    ax.plot(df["m"], df["pct_mu"],  marker=".", linestyle="--", label="Zero μ_min only")
    ax.set_xlabel("Input dimension m")
    ax.set_ylabel("% unidentifiable pairs")
    ax.set_title(f"Unidentifiable fraction vs. input dimension (n={n}, density={density}, N={trials_per_m}/m, tol={tol:g})")
    ax.set_xlim(m_min-0.5, m_max+0.5); ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout()
    fig.savefig(out/"plots"/"unident_vs_udim.png", dpi=150)
    plt.close(fig)

    return out, df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--m-min", type=int, default=1)
    ap.add_argument("--m-max", type=int, default=10)
    ap.add_argument("--trials-per-m", type=int, default=200)
    ap.add_argument("--density", type=float, default=0.8)
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--outdir", type=str, default="out_undera")
    args = ap.parse_args()
    out, df = run(n=args.n, m_min=args.m_min, m_max=args.m_max, trials_per_m=args.trials_per_m,
                  density=args.density, tol=args.tol, seed=args.seed, outdir=args.outdir)
    print("Saved summary to:", out/"summary.csv")
    print("Plot:", out/"plots"/"unident_vs_udim.png")

if __name__ == "__main__":
    main()
