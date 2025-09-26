#!/usr/bin/env python3
"""
Unidentifiability vs. input dimension m *and* density p (sparse CT ensemble).

(1) For each density p in {0.1, 0.2, ..., 1.0} and each m in {1,...,10},
    generate N=200 random (A,B) with the **sparse_continuous** ensemble (n fixed, default n=5).
    Sample x0 ~ Unif(S^{n-1}).
(2) Binary criteria with tolerance `--tol`:
      - PBH margin: d_pbh = pbh_margin_structured(A,B,x0) → unident_pbh = (d_pbh <= tol)
      - Left-eigen: mu_min = min_i μ_i(A,[x0 B])          → unident_mu  = (mu_min <= tol)
      - Union: unident_any = unident_pbh OR unident_mu
(3) Outputs:
      - Line plot: % unidentifiable vs m, with one curve per density p (union + per-criterion optional)
      - Heatmaps over (p,m) for union / PBH / μ_min
      - CSV summaries (long + wide)
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

def run(n=5, m_min=1, m_max=10, trials_per_cell=200,
        p_min=0.1, p_max=1.0, p_step=0.1,
        tol=1e-12, seed=123, outdir="/mnt/data/unident_vs_udim_by_density"):
    rng = np.random.default_rng(seed)
    out = pathlib.Path(outdir); (out/"plots").mkdir(parents=True, exist_ok=True)

    m_vals = list(range(m_min, m_max+1))
    # build density grid with step 0.1 inclusive of p_max
    P = []
    p = p_min
    while p <= p_max + 1e-12:
        P.append(round(p, 2))
        p += p_step

    # store percentages (rows = densities, cols = m)
    P_any = np.zeros((len(P), len(m_vals)))
    P_pbh = np.zeros_like(P_any)
    P_mu  = np.zeros_like(P_any)

    long_rows = []
    for i, p in enumerate(P):
        for j, m in enumerate(m_vals):
            n_any = n_pbh = n_mu = 0
            for _ in range(trials_per_cell):
                A, B = try_sparse_continuous(n, m, p, rng)
                x0 = unit(rng.standard_normal(n))

                d_pbh = float(pbh_margin_structured(A, B, x0))
                Xaug  = np.concatenate([x0.reshape(-1,1), B], axis=1)
                mu    = left_eigvec_overlap(A, Xaug)
                mu_min = float(np.min(mu)) if mu.size else 0.0

                f_pbh = (d_pbh <= tol)
                f_mu  = (mu_min <= tol)
                f_any = bool(f_pbh or f_mu)

                n_pbh += int(f_pbh); n_mu += int(f_mu); n_any += int(f_any)

            pct_pbh = 100.0 * n_pbh / trials_per_cell
            pct_mu  = 100.0 * n_mu  / trials_per_cell
            pct_any = 100.0 * n_any / trials_per_cell
            P_pbh[i, j] = pct_pbh; P_mu[i, j] = pct_mu; P_any[i, j] = pct_any

            long_rows.append(dict(n=n, m=m, density=p, N=trials_per_cell,
                                  pct_pbh=pct_pbh, pct_mu=pct_mu, pct_any=pct_any))

    # Save long and wide CSVs
    df_long = pd.DataFrame(long_rows)
    df_long.to_csv(out/"summary_long.csv", index=False)

    def save_wide(M, name):
        dfw = pd.DataFrame(M, index=P, columns=m_vals)
        dfw.index.name = "density"; dfw.columns.name = "m"
        dfw.to_csv(out/f"summary_wide_{name}.csv")

    save_wide(P_any, "any")
    save_wide(P_pbh, "pbh")
    save_wide(P_mu,  "mu")

    # Line plot: % unidentifiable (union) vs m, one curve per density
    plt.figure(figsize=(8.0, 5.0))
    for i, p in enumerate(P):
        plt.plot(m_vals, P_any[i, :], marker="o", label=f"p={p:.1f}")
    plt.xlabel("Input dimension m")
    plt.ylabel("% unidentifiable (PBH==0)")
    plt.title(f"Unidentifiable fraction vs m at varying density p (n={n}, N={trials_per_cell}/cell, tol={tol:g})")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out/"plots"/"lines_any_by_density.png", dpi=150)
    plt.close()

    # Heatmaps (density x m)
    def heatmap(M, title, fname):
        plt.figure(figsize=(8.8, 6.0))
        extent=(m_min-0.5, m_max+0.5, p_min-0.05, p_max+0.05)
        im = plt.imshow(M, origin="lower", aspect="auto", extent=extent, cmap="viridis")
        cbar = plt.colorbar(im); cbar.set_label("% unidentifiable")
        plt.xlabel("Input dimension m")
        plt.ylabel("Density p")
        plt.title(title)
        # ticks at integers for m, and 0.1 steps for p
        plt.xticks(range(m_min, m_max+1, max(1,(m_max-m_min)//10 or 1)))
        p_ticks = [round(x,1) for x in np.linspace(p_min, p_max, num=int((p_max-p_min)/p_step)+1)]
        plt.yticks(p_ticks)
        plt.tight_layout()
        plt.savefig(out/"plots"/fname, dpi=150)
        plt.close()

    heatmap(P_any, f"Unidentifiable fraction (PBH==0 OR μ_min==0)\n n={n}, N={trials_per_cell}/cell, tol={tol:g}", "heatmap_any_densityxpng")
    heatmap(P_pbh, f"Unidentifiable fraction by PBH==0\n n={n}, N={trials_per_cell}/cell, tol={tol:g}", "heatmap_pbh_densityxpng")
    heatmap(P_mu,  f"Unidentifiable fraction by μ_min==0\n n={n}, N={trials_per_cell}/cell, tol={tol:g}", "heatmap_mu_densityxpng")

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--m-min", type=int, default=1)
    ap.add_argument("--m-max", type=int, default=10)
    ap.add_argument("--trials-per-cell", type=int, default=200)
    ap.add_argument("--p-min", type=float, default=0.1)
    ap.add_argument("--p-max", type=float, default=1.0)
    ap.add_argument("--p-step", type=float, default=0.1)
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--outdir", type=str, default="out_undera_sweep")
    args = ap.parse_args()
    out = run(n=args.n, m_min=args.m_min, m_max=args.m_max, trials_per_cell=args.trials_per_cell,
              p_min=args.p_min, p_max=args.p_max, p_step=args.p_step,
              tol=args.tol, seed=args.seed, outdir=args.outdir)
    print("Saved CSVs to:", out)
    print("Plots under:", out/"plots")

if __name__ == "__main__":
    main()
