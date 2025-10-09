"""
sim1_krylov_stress.py
---------------------
Stress-test the reliability of the correlation between the Krylov metric and REE
under different structural regimes, reusing the core sim1_extended pipeline.

Batches:
  (A) n fixed at 20, deficiency d = 0..15, m = 5
  (B) d fixed at 1, n = 5..30, m = 5
  (C) d fixed at 2, n = 5..30, m = 5

For each configuration:
  - draw (A,B) with rank(Ctrb) = n - d via draw_with_ctrb_rank
  - ZOH -> (Ad,Bd)
  - x0 ~ Unif(S^{n-1}), PRBS input, simulate DT
  - estimator(s): by default MOESP (order fixed to n), but any subset allowed
  - metric: x = 1 / sigma_min(K(A,B,x0)) with K = unified_generator(..., "unrestricted")
  - target: y = REE(\hat A_d, \hat B_d ; A_d, B_d) using pair_distance
  - summary: Spearman rho(x,y), p, Fisher-z 95% CI, n_eff

Outputs:
  out_krylov_stress/
    summary_batchA.csv, summary_batchB_d1.csv, summary_batchC_d2.csv
    plot_batchA_deficiency.png
    plot_batchB_n_d1.png
    plot_batchC_n_d2.png
    (optional) scatters/...
"""

from __future__ import annotations
import argparse, math, sys, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# --- module path setup (robust to different runners) ---
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
sys.path.append("/mnt/data")  # uploaded modules fallback

try:
    from ..metrics import (
        unified_generator, cont2discrete_zoh, pair_distance,
        pbh_margin_structured, left_eigvec_overlap,  # not used, but kept for parity
    )
    from ..estimators import (dmdc_pinv, moesp_fit, sindy_fit, node_fit)
    from ..ensembles import draw_with_ctrb_rank
except Exception:
    from metrics import (
        unified_generator, cont2discrete_zoh, pair_distance,
        pbh_margin_structured, left_eigvec_overlap,
    )
    from estimators import (dmdc_pinv, moesp_fit, sindy_fit, node_fit)
    from ensembles import draw_with_ctrb_rank

EPS = 1e-12

# ------------------------------
# Core helpers (unchanged logic)
# ------------------------------
def sample_unit_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(n)
    nrm = np.linalg.norm(v)
    if nrm == 0.0:
        return sample_unit_sphere(n, rng)
    return v / nrm

def prbs(T: int, m: int, rng: np.random.Generator, dwell: int = 1) -> np.ndarray:
    steps = math.ceil(T / dwell)
    seq = rng.choice([-1.0, 1.0], size=(steps, m))
    u = np.repeat(seq, repeats=dwell, axis=0)[:T, :]
    return u

def simulate_dt(Ad: np.ndarray, Bd: np.ndarray, u: np.ndarray, x0: np.ndarray) -> np.ndarray:
    n, T = Ad.shape[0], u.shape[0]
    X = np.empty((n, T + 1), dtype=float)
    X[:, 0] = x0
    for k in range(T):
        X[:, k + 1] = Ad @ X[:, k] + Bd @ u[k, :]
    return X

def relative_error_fro(Ahat: np.ndarray, Bhat: np.ndarray, Atrue: np.ndarray, Btrue: np.ndarray) -> float:
    return float(pair_distance(Ahat, Bhat, Atrue, Btrue))

def make_estimator(name: str):
    lname = name.lower()
    if lname == "dmdc":
        return lambda X0, X1, U, n=None, dt=None: dmdc_pinv(X0, X1, U)
    if lname == "moesp":
        return lambda X0, X1, U, n=None, dt=None: moesp_fit(X0, X1, U, n=n)
    if lname == "sindy":
        return lambda X0, X1, U, n=None, dt=None: sindy_fit(X0, X1, U, dt)
    if lname == "node":
        return lambda X0, X1, U, n=None, dt=None: node_fit(X0, X1, U, dt)
    raise ValueError(f"Unknown estimator '{name}'. Choose from dmdc, moesp, sindy, node.")

def draw_system(n: int, m: int, deficiency: int, rng: np.random.Generator,
                ensemble: str, embed_random_basis: bool = True):
    r = max(0, n - int(deficiency))
    base_map = {
        "ginibre": ("ginibre", "ginibre"),
        "stable":  ("stable",  "stable"),
        "sparse":  ("sparse",  "sparse"),
        "binary":  ("binary",  "binary"),
    }
    if ensemble not in base_map:
        raise ValueError(f"Unknown ensemble '{ensemble}'. Choose from {list(base_map)}.")
    base_c, base_u = base_map[ensemble]
    A, B, meta = draw_with_ctrb_rank(
        n=n, m=m, r=r, rng=rng,
        base_c=base_c, base_u=base_u, embed_random_basis=embed_random_basis
    )
    return A, B, meta

# ------------------------------
# Krylov metric (unrestricted)
# ------------------------------
def krylov_smin(A: np.ndarray, B: np.ndarray, x0: np.ndarray) -> float:
    K = unified_generator(A, B, x0, mode="unrestricted")
    if K.size == 0:
        return 0.0
    s = np.linalg.svd(K, compute_uv=False)
    return float(s.min()) if s.size else 0.0

# ------------------------------
# Spearman with Fisher-z CI
# ------------------------------
def spearman_with_ci(x: np.ndarray, y: np.ndarray):
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, (np.nan, np.nan), n
    r, p = spearmanr(x[mask], y[mask])
    # Fisher z CI for correlation: z = atanh(r), se = 1/sqrt(n-3)
    if not np.isfinite(r) or abs(r) >= 1:
        return float(r), float(p), (np.nan, np.nan), n
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(max(n - 3, 1))
    z_lo, z_hi = z - 1.96 * se, z + 1.96 * se
    r_lo, r_hi = np.tanh([z_lo, z_hi])
    return float(r), float(p), (float(r_lo), float(r_hi)), n

# ------------------------------
# Single configuration run
# ------------------------------
def run_config(*, n: int, m: int, d: int, T: int, dt: float, trials: int,
               noise_std: float, seed: int, ensemble: str, estimators: list[str],
               save_scatter: bool, scatter_dir: pathlib.Path, tag: str):
    rng = np.random.default_rng(seed)
    A, B, meta = draw_system(n=n, m=m, deficiency=d, rng=rng, ensemble=ensemble)
    Ad, Bd = cont2discrete_zoh(A, B, dt)

    # Krylov inverse vector and REEs per estimator
    xvals = []  # 1 / sigma_min(K)
    errs_by_est = {est: [] for est in estimators}

    est_funcs = {}
    for name in estimators:
        try:
            est_funcs[name] = make_estimator(name)
        except Exception as e:
            print(f"[warn] Skipping estimator '{name}': {e}")

    for t in range(trials):
        x0 = sample_unit_sphere(n, rng)
        u = prbs(T, m, rng, dwell=1)
        X = simulate_dt(Ad, Bd, u, x0)
        if noise_std > 0:
            X = X + noise_std * rng.standard_normal(X.shape)
        X0, X1, U = X[:, :-1], X[:, 1:], u.T

        smin = krylov_smin(A, B, x0)
        xvals.append(1.0 / max(smin, EPS))

        for name, f in est_funcs.items():
            try:
                Ahat, Bhat = f(X0, X1, U, n=n, dt=dt)
                errs_by_est[name].append(relative_error_fro(Ahat, Bhat, Ad, Bd))
            except Exception as e:
                print(f"[warn] Estimator '{name}' failed on trial {t}: {e}")
                errs_by_est[name].append(np.nan)

    xvals = np.asarray(xvals, float)
    # Summaries
    rows = []
    for name in estimators:
        y = np.asarray(errs_by_est[name], float)
        rho, p, (lo, hi), n_eff = spearman_with_ci(xvals, y)
        rows.append(dict(n=n, m=m, deficiency=d, estimator=name,
                         rho=rho, p=p, rho_lo=lo, rho_hi=hi, n_eff=n_eff,
                         T=T, dt=dt, trials=trials, ensemble=ensemble, seed=seed))
        if save_scatter:
            scatter_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(5.2, 4.0))
            mask = np.isfinite(xvals) & np.isfinite(y)
            ax.scatter(xvals[mask], y[mask], s=18,
                       label=rf"Spearman $\rho$={rho:.3f}, p={p:.1e}, n={n_eff}")
            ax.set_xlabel("1 / σ_min(K)")
            ax.set_ylabel(f"REE ({name})")
            ax.set_title(f"Krylov vs. REE — n={n}, m={m}, d={d}")
            ax.legend(frameon=True, loc="best")
            fig.savefig(scatter_dir / f"krylov_vs_ree_{tag}_n{n}_m{m}_d{d}_{name}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    return pd.DataFrame(rows)

# ------------------------------
# Sweep runners
# ------------------------------
def sweep_deficiency(*, n_fixed: int, m_fixed: int, d_min: int, d_max: int, **kw):
    """Sweep d; ensure 'tag' is passed only once to run_config."""
    dfs = []
    batch_tag = kw.pop("tag", "batchA")  # remove 'tag' from kw to avoid duplication
    for d in range(d_min, d_max + 1):
        tag = f"{batch_tag}"
        df = run_config(n=n_fixed, m=m_fixed, d=d, tag=tag, **kw)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def sweep_n_at_d(*, d_fixed: int, n_min: int, n_max: int, m_fixed: int, **kw):
    """Sweep n at fixed d; ensure 'tag' is passed only once to run_config."""
    dfs = []
    batch_tag = kw.pop("tag", "batchB")  # remove 'tag' from kw to avoid duplication
    for n in range(n_min, n_max + 1):
        tag = f"{batch_tag}"
        df = run_config(n=n, m=m_fixed, d=d_fixed, tag=tag, **kw)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# ------------------------------
# Summary plots
# ------------------------------
def plot_summary_def(summary: pd.DataFrame, out_png: pathlib.Path):
    # One line per estimator: rho vs deficiency (with CI band)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for est, grp in summary.groupby("estimator"):
        grp = grp.sort_values("deficiency")
        ax.plot(grp["deficiency"], grp["rho"], marker="o", linewidth=1.5, label=est)
        # CI band
        if np.isfinite(grp["rho_lo"]).all() and np.isfinite(grp["rho_hi"]).all():
            ax.fill_between(grp["deficiency"], grp["rho_lo"], grp["rho_hi"], alpha=0.2)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Deficiency d (rank(Ctrb)=n−d)")
    ax.set_ylabel("Spearman ρ( 1/σ_min(K), REE )")
    ax.set_title("Krylov–REE correlation vs. deficiency (n fixed)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_summary_n(summary: pd.DataFrame, out_png: pathlib.Path, d_fixed: int):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for est, grp in summary.groupby("estimator"):
        grp = grp.sort_values("n")
        ax.plot(grp["n"], grp["rho"], marker="o", linewidth=1.5, label=est)
        if np.isfinite(grp["rho_lo"]).all() and np.isfinite(grp["rho_hi"]).all():
            ax.fill_between(grp["n"], grp["rho_lo"], grp["rho_hi"], alpha=0.2)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("State dimension n")
    ax.set_ylabel("Spearman ρ( 1/σ_min(K), REE )")
    ax.set_title(f"Krylov–REE correlation vs. n (deficiency d={d_fixed})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ------------------------------
# CLI
# ------------------------------
def parse_estimators(s: str) -> list[str]:
    return [t.strip().lower() for t in s.split(",") if t.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="out_krylov_stress")
    ap.add_argument("--ensemble", type=str, default="ginibre",
                    choices=["ginibre", "stable", "sparse", "binary"])
    ap.add_argument("--estimators", type=str, default="moesp",
                    help="Comma-separated subset of {dmdc,moesp,sindy,node}; defaults to moesp.")
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--save-scatter", action="store_true",
                    help="Also save per-configuration scatter plots.")
    # What to run
    ap.add_argument("--run", type=str, default="all",
                    choices=["batchA", "batchB_d1", "batchC_d2", "all"],
                    help="Which batch to run.")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    scatter_dir = outdir / "scatters"

    est_list = parse_estimators(args.estimators)

    summaries = {}

    # (A) n=20 fixed, d=0..15, m=5
    if args.run in ("batchA", "all"):
        dfA = sweep_deficiency(
            n_fixed=20, m_fixed=5, d_min=0, d_max=15,
            T=args.T, dt=args.dt, trials=args.trials,
            noise_std=args.noise_std, seed=args.seed, ensemble=args.ensemble,
            estimators=est_list, save_scatter=args.save_scatter,
            scatter_dir=scatter_dir, tag="batchA"
        )
        dfA.to_csv(outdir / "summary_batchA.csv", index=False)
        plot_summary_def(dfA, outdir / "plot_batchA_deficiency.png")
        summaries["A"] = dfA

    # (B) d=1 fixed, n=5..30, m=5
    if args.run in ("batchB_d1", "all"):
        dfB = sweep_n_at_d(
            d_fixed=1, n_min=5, n_max=30, m_fixed=5,
            T=args.T, dt=args.dt, trials=args.trials,
            noise_std=args.noise_std, seed=args.seed, ensemble=args.ensemble,
            estimators=est_list, save_scatter=args.save_scatter,
            scatter_dir=scatter_dir, tag="batchB_d1"
        )
        dfB.to_csv(outdir / "summary_batchB_d1.csv", index=False)
        plot_summary_n(dfB, outdir / "plot_batchB_n_d1.png", d_fixed=1)
        summaries["B_d1"] = dfB

    # (C) d=2 fixed, n=5..30, m=5
    if args.run in ("batchC_d2", "all"):
        dfC = sweep_n_at_d(
            d_fixed=2, n_min=5, n_max=30, m_fixed=5,
            T=args.T, dt=args.dt, trials=args.trials,
            noise_std=args.noise_std, seed=args.seed, ensemble=args.ensemble,
            estimators=est_list, save_scatter=args.save_scatter,
            scatter_dir=scatter_dir, tag="batchC_d2"
        )
        dfC.to_csv(outdir / "summary_batchC_d2.csv", index=False)
        plot_summary_n(dfC, outdir / "plot_batchC_n_d2.png", d_fixed=2)
        summaries["B_d2"] = dfC

    print("Saved to:", outdir)
    for k, df in summaries.items():
        print(f"  {k}: {len(df)} rows, estimators={sorted(df['estimator'].unique())}")

if __name__ == "__main__":
    main()
