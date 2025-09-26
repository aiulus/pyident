#!/usr/bin/env python3
"""
Unidentifiability heatmap vs. (state dim n, input dim m) for sparse CT ensembles.

Spec:
  (1) For each (n, m) on a grid, generate N random (A,B) pairs from the
      **sparse_continuous** ensemble at fixed density p=0.8; sample x0 ~ Unif(S^{n-1}).
  (2) Compute binary criteria with tolerance `tol=1e-12` (configurable):
        - PBH structured margin: d_pbh = pbh_margin_structured(A,B,x0) → unident_pbh = (d_pbh <= tol)
        - Left-eigen overlap:  μ_min = min_i μ_i(A,[x0 B])              → unident_mu  = (μ_min <= tol)
        - Union: unident_any = unident_pbh OR unident_mu
  (3) Plot a heatmap with x-axis = n (size(A,1)) in {1..20}, y-axis = m (size(B,2)) in {1..40},
      values = percentage of unidentifiable pairs.

Outputs:
  - summary_long.csv   (tall table; one row per (n,m))
  - summary_wide_*.csv (matrices of percentages per criterion)
  - plots/heatmap_*.png (union + per-criterion heatmaps)
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

def run(n_min=1, n_max=20, m_min=1, m_max=40, trials=200, density=0.8, tol=1e-12,
        seed=123, outdir="/mnt/data/unident_heatmap_nm_run"):
    rng = np.random.default_rng(seed)
    out = pathlib.Path(outdir); (out/"plots").mkdir(parents=True, exist_ok=True)

    n_vals = list(range(n_min, n_max+1))
    m_vals = list(range(m_min, m_max+1))

    # Matrices of percentages (rows=m, cols=n)
    P_any = np.zeros((len(m_vals), len(n_vals)))
    P_pbh = np.zeros_like(P_any)
    P_mu  = np.zeros_like(P_any)

    rows = []
    for j, m in enumerate(m_vals):
        for i, n in enumerate(n_vals):
            n_any = n_pbh = n_mu = 0
            for _ in range(trials):
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
            pct_pbh = 100.0 * n_pbh / trials
            pct_mu  = 100.0 * n_mu  / trials
            pct_any = 100.0 * n_any / trials
            P_pbh[j, i] = pct_pbh; P_mu[j, i] = pct_mu; P_any[j, i] = pct_any
            rows.append(dict(n=n, m=m, density=density, N=trials,
                             pct_pbh=pct_pbh, pct_mu=pct_mu, pct_any=pct_any))

    # Save long summary
    df = pd.DataFrame(rows)
    df.to_csv(out/"summary_long.csv", index=False)

    # Save wide CSVs
    def to_wide(M, name):
        dfw = pd.DataFrame(M, index=m_vals, columns=n_vals)
        dfw.index.name = "m"; dfw.columns.name = "n"
        dfw.to_csv(out/f"summary_wide_{name}.csv")
    to_wide(P_any, "any")
    to_wide(P_pbh, "pbh")
    to_wide(P_mu,  "mu")

    # Heatmap helper
    def plot_heat(M, title, fname):
        plt.figure(figsize=(8.8, 6.0))
        # extent so that ticks align with integer n,m
        extent=[n_min-0.5, n_max+0.5, m_min-0.5, m_max+0.5]
        im = plt.imshow(M, origin="lower", aspect="auto", extent=extent, cmap="viridis")
        cbar = plt.colorbar(im); cbar.set_label("% unidentifiable")
        plt.xlabel("State dimension n (size(A,1))")
        plt.ylabel("Input dimension m (size(B,2))")
        plt.title(title)
        # show integer ticks every few steps to keep things readable
        plt.xticks(range(n_min, n_max+1, max(1,(n_max-n_min)//10 or 1)))
        plt.yticks(range(m_min, m_max+1, max(2,(m_max-m_min)//10 or 1)))
        plt.tight_layout()
        plt.savefig(out/"plots"/fname, dpi=150); plt.close()

    plot_heat(P_any, f"Unidentifiable fraction (PBH==0 OR μ_min==0)\n density={density}, N={trials}/cell, tol={tol:g}", "heatmap_any.png")
    plot_heat(P_pbh, f"Unidentifiable fraction by PBH==0\n density={density}, N={trials}/cell, tol={tol:g}", "heatmap_pbh.png")
    plot_heat(P_mu,  f"Unidentifiable fraction by μ_min==0\n density={density}, N={trials}/cell, tol={tol:g}", "heatmap_mu.png")

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-min", type=int, default=1)
    ap.add_argument("--n-max", type=int, default=20)
    ap.add_argument("--m-min", type=int, default=1)
    ap.add_argument("--m-max", type=int, default=40)
    ap.add_argument("--trials", type=int, default=200, help="Samples per (n,m) cell")
    ap.add_argument("--density", type=float, default=0.8)
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--outdir", type=str, default="out_undera_heatmap")
    args = ap.parse_args()
    out = run(n_min=args.n_min, n_max=args.n_max, m_min=args.m_min, m_max=args.m_max,
              trials=args.trials, density=args.density, tol=args.tol, seed=args.seed,
              outdir=args.outdir)
    print("Saved CSVs to:", out)
    print("Plots under:", out/"plots")

if __name__ == "__main__":
    main()
