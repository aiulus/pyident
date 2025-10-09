#!/usr/bin/env python3
"""
Unidentifiability heatmap vs. (state dim n, density p) with single input channel (m=1).

Spec:
  (1) For each (n, p) on a grid, generate N random (A,B) pairs from the
      **sparse_continuous** ensemble at density p; sample x0 ~ Unif(S^{n-1}).
  (2) Decide unidentifiability by metric + threshold (default tol=1e-12):
        - PBH structured margin: d_pbh = pbh_margin_structured(A,B,x0) → unident_pbh = (d_pbh <= tol)
        - Left-eigen overlap:  μ_min = min_i μ_i(A,[x0 B])              → unident_mu  = (μ_min <= tol)
        - Union (primary output): unident_any = unident_pbh OR unident_mu
  (3) Output a heatmap whose (row=p, col=n) value is the % of unidentifiable pairs for the union rule.
      Also writes CSV summaries (long + a wide matrix for the union).
Grid:
  - x-axis (n): 1..50 (step 1)
  - y-axis (density p): 0.05..1.0 (step 0.05)  → 20 density levels
  - m is fixed to 1 (single input channel)
  - N per cell: 200 (configurable)
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
    n = np.linalg.norm(v);  return v / (n if n>0 else 1.0)

def try_sparse_continuous(n, m, p, rng):
    return sparse_continuous(n=n, m=m, which="both", rng=rng, p_density=p)


def run(n_min=1, n_max=50, p_start=0.05, p_end=1.0, p_step=0.05,
        trials=200, tol=1e-12, seed=123, outdir="/mnt/data/unident_heatmap_n_by_density_m1"):
    rng = np.random.default_rng(seed)
    out = pathlib.Path(outdir); (out/"plots").mkdir(parents=True, exist_ok=True)

    n_vals = list(range(n_min, n_max+1))
    # build densities 0.05..1.00
    P = []
    p = p_start
    while p <= p_end + 1e-12:
        P.append(round(p, 2))
        p += p_step
    # ensure exactly expected count
    P = P[:int(round((p_end-p_start)/p_step))+1]

    m = 1  # single input channel

    # Matrix of percentages (rows=density index, cols=n)
    P_any = np.zeros((len(P), len(n_vals)))

    rows = []
    for i, p in enumerate(P):
        for j, n in enumerate(n_vals):
            n_any = 0
            for _ in range(trials):
                A, B = try_sparse_continuous(n, m, p, rng)
                x0 = unit(rng.standard_normal(n))
                # Metrics
                d_pbh = float(pbh_margin_structured(A, B, x0))
                Xaug  = np.concatenate([x0.reshape(-1,1), B], axis=1)
                mu    = left_eigvec_overlap(A, Xaug)
                mu_min = float(np.min(mu)) if mu.size else 0.0
                # Binary (union)
                f_any = (d_pbh <= tol) or (mu_min <= tol)
                n_any += int(f_any)
            pct_any = 100.0 * n_any / trials
            P_any[i, j] = pct_any
            rows.append(dict(n=n, density=p, m=m, N=trials, pct_any=pct_any))

    # Save long summary and wide matrix
    df = pd.DataFrame(rows)
    df.to_csv(out/"summary_long.csv", index=False)
    dfw = pd.DataFrame(P_any, index=P, columns=n_vals)
    dfw.index.name = "density"; dfw.columns.name = "n"
    dfw.to_csv(out/"summary_wide_any.csv")

    # Heatmap (density x n)
    plt.figure(figsize=(10.0, 6.2))
    extent = (n_min-0.5, n_max+0.5, p_start-0.05, p_end+0.05)
    im = plt.imshow(P_any, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    cbar = plt.colorbar(im); cbar.set_label("% unidentifiable (PBH==0 OR μ_min==0)")
    plt.xlabel("State dimension n (size(A,1))")
    plt.ylabel("Density p")
    plt.title(f"Unidentifiable fraction (union criterion), m=1\nN={trials}/cell, tol={tol:g}")
    # readable ticks
    plt.xticks(range(n_min, n_max+1, max(2,(n_max-n_min)//10 or 1)))
    p_ticks = [round(x,2) for x in np.linspace(p_start, p_end, num=int((p_end-p_start)/p_step)+1)]
    plt.yticks(p_ticks)
    plt.tight_layout()
    plt.savefig(out/"plots"/"heatmap_any_densityxN_m1.png", dpi=150)
    plt.close()

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-min", type=int, default=1)
    ap.add_argument("--n-max", type=int, default=50)
    ap.add_argument("--p-start", type=float, default=0.05)
    ap.add_argument("--p-end", type=float, default=1.0)
    ap.add_argument("--p-step", type=float, default=0.05)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--outdir", type=str, default="out_n_vs_sparse")
    args = ap.parse_args()
    out = run(n_min=args.n_min, n_max=args.n_max, p_start=args.p_start, p_end=args.p_end, p_step=args.p_step,
              trials=args.trials, tol=args.tol, seed=args.seed, outdir=args.outdir)
    print("Saved CSVs to:", out)
    print("Plots under:", out/"plots")

if __name__ == "__main__":
    main()
    
