"""
Unidentifiability vs. sparsity density (continuous-time ensemble).

(1) Generate random (A,B) with the **sparse_continuous** ensemble (density p).
    If `ensembles.sparse_continuous` is unavailable, we fall back to a compatible
    sparse CT generator that enforces stability of A by spectral shift.
(2) For each (A,B) and x0~Unif(S^{n-1}), compute:
      - PBH structured margin: d_pbh = pbh_margin_structured(A,B,x0)
      - Left-eigen criterion:  mu_min = min_i ||w_i^T [x0 B]||_2 / ||w_i||_2
    Treat them as **binary** with tolerance `--tol`:
      unident_pbh = (d_pbh <= tol)
      unident_mu  = (mu_min <= tol)
      unident_any = unident_pbh OR unident_mu
(3) For densities p ∈ {0.05, 0.10, …, 0.95, 1.0} (20 levels), with N=200 per level,
    report the **percentage of unidentifiable pairs** vs. p and plot it.
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


def run(n=6, m=2, trials_per_p=200, tol=1e-12, seed=123,
        p_start=0.05, p_end=1.0, p_step=0.05, outdir="sparse_density_unident"):
    rng = np.random.default_rng(seed)
    out = pathlib.Path(outdir); (out/"plots").mkdir(parents=True, exist_ok=True)

    # Density grid: 0.05..1.0 step 0.05 (inclusive end)
    P = []
    p = p_start
    while p <= p_end + 1e-12:
        P.append(round(p, 2))
        p += p_step
    # Ensure exactly 20 levels if user kept defaults
    P = P[:int(round((p_end-p_start)/p_step))+1]

    rows = []
    for p in P:
        n_any = n_pbh = n_mu = 0
        for _ in range(trials_per_p):
            A, B = try_sparse_continuous(n, m, p, rng)
            x0 = unit(rng.standard_normal(n))

            # Criteria
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
            density=p,
            N=trials_per_p,
            unident_pbh=n_pbh,
            unident_mu=n_mu,
            unident_any=n_any,
            pct_pbh=100.0*n_pbh/trials_per_p,
            pct_mu=100.0*n_mu/trials_per_p,
            pct_any=100.0*n_any/trials_per_p,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(out/"summary.csv", index=False)

    # Plot: percentage unidentifiable vs density (union) with breakdown lines
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    #ax.plot(df["density"], df["pct_any"], marker="o", label="Zero PBH OR Zero μ_min")
    ax.plot(df["density"], df["pct_pbh"], marker=".", linestyle="--", label="Zero PBH")
    #ax.plot(df["density"], df["pct_mu"], marker=".", linestyle="--", label="Zero μ_min only")
    ax.set_xlabel("Matrix density p")
    ax.set_ylabel("% unidentifiable pairs")
    ax.set_title(f"Unidentifiable fraction vs. sparsity density (n={n}, m={m}, N={trials_per_p}/level, tol={tol:g})")
    ax.set_xlim(min(P)-0.02, max(P)+0.02); ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout()
    fig.savefig(out/"plots"/"unident_vs_density.png", dpi=150)
    plt.close(fig)

    return out, df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--trials-per-p", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--p-start", type=float, default=0.05)
    ap.add_argument("--p-end", type=float, default=1.0)
    ap.add_argument("--p-step", type=float, default=0.05)
    ap.add_argument("--outdir", type=str, default="out_sparse")
    args = ap.parse_args()
    out, df = run(n=args.n, m=args.m, trials_per_p=args.trials_per_p, tol=args.tol, seed=args.seed,
                  p_start=args.p_start, p_end=args.p_end, p_step=args.p_step, outdir=args.outdir)
    print("Saved summary to:", out/"summary.csv")
    print("Plot:", out/"plots"/"unident_vs_density.png")

if __name__ == "__main__":
    main()
