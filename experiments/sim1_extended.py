"""
Identifiability criteria vs. estimation error (DD-ID setting, with inputs).
Extended version with selectable ensembles and estimators (CLI).

Core logic preserved:
  - Fix (A,B) with controllability rank n - deficiency via draw_with_ctrb_rank.
  - For each trial: x0 ~ Unif(S^{n-1}), PRBS input, simulate DT system, estimate (A,B).
  - Compute criteria: PBH-structured margin, σ_min(K) on unified generator, left-eig overlap.
  - X-axis transforms: 1/pbh, 1/σ_min(K), 1/mu_min.  Y: relative Frobenius error vs (Ad,Bd).
  - Save trial-wise CSV, Spearman correlations, scatter plots.

New CLI:
  --ensemble {ginibre,stable,sparse,binary}  (default: ginibre)
  --deficiency D  (default: 1 → rank n-1)
  --estimators dmdc,moesp[,sindy,node]  (default: dmdc,moesp)
"""

from __future__ import annotations
import argparse, math, sys, os, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# --- module path setup (robust to different runners) ---
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
sys.path.append("/mnt/data")  # uploaded modules fallback

# Try relative imports first (package mode), then absolute (script mode).
try:
    from ..metrics import (
        pbh_margin_structured,
        unified_generator,
        left_eigvec_overlap,
        cont2discrete_zoh,
        pair_distance,
    )
    from ..estimators import (
        dmdc_pinv,
        moesp_fit,
        sindy_fit,
        node_fit,
    )
    from ..ensembles import draw_with_ctrb_rank
except Exception:
    from metrics import (
        pbh_margin_structured,
        unified_generator,
        left_eigvec_overlap,
        cont2discrete_zoh,
        pair_distance,
    )
    from estimators import (
        dmdc_pinv,
        moesp_fit,
        sindy_fit,
        node_fit,
    )
    from ensembles import draw_with_ctrb_rank

# ------------------------------
# Helpers
# ------------------------------
EPS = 1e-12

def sample_unit_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(n)
    nrm = np.linalg.norm(v)
    if nrm == 0.0:
        return sample_unit_sphere(n, rng)
    return v / nrm

def prbs(T: int, m: int, rng: np.random.Generator, dwell: int = 1) -> np.ndarray:
    """±1 PRBS with optional dwell; shape (T, m)."""
    steps = math.ceil(T / dwell)
    seq = rng.choice([-1.0, 1.0], size=(steps, m))
    u = np.repeat(seq, repeats=dwell, axis=0)[:T, :]
    return u

def simulate_dt(Ad: np.ndarray, Bd: np.ndarray, u: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """
    Simulate x_{k1} = Ad x_k  Bd u_k.
    Returns X with shape (n, T1).
    """
    n = Ad.shape[0]
    T = u.shape[0]
    X = np.empty((n, T + 1), dtype=float)
    X[:, 0] = x0
    for k in range(T):
        X[:, k + 1] = Ad @ X[:, k] + Bd @ u[k, :]
    return X

def relative_error_fro(Ahat: np.ndarray, Bhat: np.ndarray, Atrue: np.ndarray, Btrue: np.ndarray) -> float:
    # Uses your shared metric from metrics.py
    return float(pair_distance(Ahat, Bhat, Atrue, Btrue))

# ------------------------------
# Ensembles & Estimators registry
# ------------------------------
def draw_system(n: int, m: int, deficiency: int, rng: np.random.Generator,
                ensemble: str, embed_random_basis: bool = True):
    """
    Uses draw_with_ctrb_rank but switches base_c/base_u to realize different ensembles.
    Keeps signature uniform and the core logic intact.
    """
    r = max(0, n - int(deficiency))  # controllability rank
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

def make_estimator(name: str):
    """
    Return a callable (X0, X1, U, n, dt) -> (Ahat, Bhat).
    Wrap signatures so the main loop remains clean.
    """
    lname = name.lower()
    if lname == "dmdc":
        def _f(X0, X1, U, n=None, dt=None):
            return dmdc_pinv(X0, X1, U)
        return _f
    if lname == "moesp":
        def _f(X0, X1, U, n=None, dt=None):
            # full-state wrapper: pass known n
            return moesp_fit(X0, X1, U, n=n)
        return _f
    if lname == "sindy":
        def _f(X0, X1, U, n=None, dt=None):
            if dt is None:
                raise ValueError("sindy requires dt")
            return sindy_fit(X0, X1, U, dt)
        return _f
    if lname == "node":
        def _f(X0, X1, U, n=None, dt=None):
            if dt is None:
                raise ValueError("node requires dt")
            return node_fit(X0, X1, U, dt)
        return _f
    raise ValueError(f"Unknown estimator '{name}'. Choose from dmdc, moesp, sindy, node.")

# ------------------------------
# Criteria
# ------------------------------
def compute_core_metrics(A: np.ndarray, B: np.ndarray, x0: np.ndarray) -> dict:
    """
    Three manuscript-linked criteria:
      - PBH structured margin proxy: pbh_struct
      - Krylov sigma_min on unified generator K (unrestricted)
      - Left-eigen overlap mu_min for Xaug=[x0 B]
    """
    pbh_struct = float(pbh_margin_structured(A, B, x0))
    K = unified_generator(A, B, x0, mode="unrestricted")
    svals = np.linalg.svd(K, compute_uv=False)
    krylov_smin = float(svals.min()) if svals.size else 0.0
    Xaug = np.concatenate([x0.reshape(-1, 1), B], axis=1)
    mu = left_eigvec_overlap(A, Xaug)
    mu_min = float(np.min(mu)) if np.size(mu) else 0.0
    return dict(pbh_struct=pbh_struct, krylov_smin=krylov_smin, mu_min=mu_min)

def add_transforms(df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """
    X1 = 1 / pbh_struct
    X2 = 1 / krylov_smin  (NOTE: your original comment said 'no transform' for K, but you also used 1/· elsewhere)
    X3 = 1 / mu_min
    """
    df = df.copy()
    df["x_inv_pbh"] = 1.0 / np.maximum(df["pbh_struct"].to_numpy(), eps)
    df["x_inv_krylov_smin"] = 1.0 / np.maximum(df["krylov_smin"].to_numpy(), eps)
    df["x_inv_mu"] = 1.0 / np.maximum(df["mu_min"].to_numpy(), eps)
    return df

# ------------------------------
# Trials
# ------------------------------
def run_trials(*, n=6, m=2, T=100, dt=0.01, trials=200, noise_std=0.0, seed=123,
               ensemble="ginibre", deficiency=1, estimators=("dmdc","moesp")):
    rng = np.random.default_rng(seed)

    # Draw CT (A,B) at requested deficiency and ensemble; convert to DT
    A, B, meta = draw_system(n=n, m=m, deficiency=deficiency, rng=rng, ensemble=ensemble)
    Ad, Bd = cont2discrete_zoh(A, B, dt)

    # Wrap estimators
    est_funcs = {}
    for name in estimators:
        try:
            est_funcs[name] = make_estimator(name)
        except Exception as e:
            print(f"[warn] Skipping estimator '{name}': {e}")

    if not est_funcs:
        raise RuntimeError("No valid estimators selected.")

    rows = []
    for t in range(trials):
        x0 = sample_unit_sphere(n, rng)
        u = prbs(T, m, rng, dwell=1)            # (T, m)
        X = simulate_dt(Ad, Bd, u, x0)          # (n, T1)
        if noise_std > 0:
            X = X + noise_std * rng.standard_normal(X.shape)

        X0, X1, U = X[:, :-1], X[:, 1:], u.T    # shapes: (n,T), (n,T), (m,T)

        # Criteria (computed on CT A,B per your manuscript linkage)
        crit = compute_core_metrics(A, B, x0)

        # Estimation & errors
        errs = {}
        for name, f in est_funcs.items():
            try:
                Ahat, Bhat = f(X0, X1, U, n=n, dt=dt)
                errs[f"err_{name}"] = relative_error_fro(Ahat, Bhat, Ad, Bd)
            except Exception as e:
                print(f"[warn] Estimator '{name}' failed on trial {t}: {e}")
                errs[f"err_{name}"] = np.nan

        rows.append(dict(trial=t, **crit, **errs))

    df = pd.DataFrame(rows)
    meta_out = {"A": A, "B": B, "Ad": Ad, "Bd": Bd, "meta": meta,
                "ensemble": ensemble, "deficiency": deficiency, "estimators": list(est_funcs.keys())}
    return df, meta_out

# ------------------------------
# Reporting
# ------------------------------
def spearman_table(df: pd.DataFrame, estimator_cols: list[str]) -> pd.DataFrame:
    xcols = [
        ("x_inv_pbh", "1 / PBH metric"),
        ("x_inv_krylov_smin", "1 / σ_min(K)"),
        ("x_inv_mu", "1 / mu_min"),
    ]
    records = []
    for x, _ in xcols:
        rec = {"metric": x}
        for y in estimator_cols:
            # drop NaNs for correlation
            mask = np.isfinite(df[x].to_numpy()) & np.isfinite(df[y].to_numpy())
            if np.sum(mask) < 3:
                rho, p = np.nan, np.nan
            else:
                rho, p = spearmanr(df.loc[mask, x], df.loc[mask, y])
            rec[f"{y}_rho"] = str(float(rho)) if isinstance(rho, (int, float)) and np.isfinite(float(rho)) else "nan"
            rec[f"{y}_p"]   = str(float(p))   if isinstance(p, (int, float)) and np.isfinite(float(p))   else "nan"
        records.append(rec)
    return pd.DataFrame(records)

def scatter_plots(df: pd.DataFrame, ykey: str, outdir: pathlib.Path, tag: str):
    pairs = [
        ("x_inv_pbh", "1 / PBH (structured)"),
        ("x_inv_krylov_smin", "1 / σ_min(K_n)"),
        ("x_inv_mu", "1 / mu_min"),
    ]
    for x, xlabel in pairs:
        #fig, ax = plt.subplots(figsize=(5.2, 4.0))
        #ax.scatter(df[x].to_numpy(), df[ykey].to_numpy(), s=18)
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        xvals = df[x].to_numpy()
        yvals = df[ykey].to_numpy()
        mask = np.isfinite(xvals) & np.isfinite(yvals)
        n_eff = int(mask.sum())
        if n_eff >= 3:
            rho, p = spearmanr(xvals[mask], yvals[mask])
        else:
            rho, p = np.nan, np.nan
        # format safe strings for legend
        rho_s = "nan" if not np.isfinite(np.asarray(rho)) else f"{float(rho):.3f}"
        p_s   = "nan" if not np.isfinite(np.asarray(p)) else f"{float(p):.1e}"
        # plot full sample and attach correlation legend
        ax.scatter(xvals, yvals, s=18, label=rf"Spearman $\rho$={rho_s}, p={p_s}, n={n_eff}")
         
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"REE ({ykey.replace('err_', '')})")
        ax.set_title("Estimation error vs. identifiability criterion")
        ax.legend(frameon=True, loc="best")
        fig.savefig(outdir / f"{x}_vs_{ykey}_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _compute_zoom_limits(xvals: np.ndarray, yvals: np.ndarray, q: float = 0.9):
    """
    Lower-left zoom box: [min(x), q-quantile(x)] × [min(y), q-quantile(y)].
    With q=0.9 on both axes, the rectangle contains at least 80% of points (2q−1 bound).
    Returns (xlo, xhi, ylo, yhi) with a small padding.
    """
    mask = np.isfinite(xvals) & np.isfinite(yvals)
    if mask.sum() < 3:
        return None
    x = xvals[mask]; y = yvals[mask]
    xlo = float(np.nanmin(x));  ylo = float(np.nanmin(y))
    xhi = float(np.nanquantile(x, q));  yhi = float(np.nanquantile(y, q))
    # Avoid degeneracy and add gentle padding
    if not np.isfinite(xhi) or xhi <= xlo: xhi = xlo + 1e-12
    if not np.isfinite(yhi) or yhi <= ylo: yhi = ylo + 1e-12
    xpad = 0.02 * (xhi - xlo + 1e-12)
    ypad = 0.02 * (yhi - ylo + 1e-12)
    return (xlo - xpad, xhi + xpad, ylo - ypad, yhi + ypad)

def scatter_plots_zoom(df: pd.DataFrame, ykey: str, outdir: pathlib.Path, tag: str, q_zoom: float = 0.9):
    """
    Produce a second set of scatter plots with both axes truncated to a lower-left
    rectangle that covers ≈80% of points by default (q_zoom=0.9 per-axis).
    Legends still report correlations computed on the full sample (unchanged logic).
    """
    pairs = [
        ("x_inv_pbh", "1 / PBH (structured)"),
        ("x_inv_krylov_smin", "1 / σ_min(K_n)"),
        ("x_inv_mu", "1 / mu_min"),
    ]
    for x, xlabel in pairs:
        xvals = df[x].to_numpy()
        yvals = df[ykey].to_numpy()
        # Reuse full-sample correlation (logic unchanged)
        mask = np.isfinite(xvals) & np.isfinite(yvals)
        n_eff = int(mask.sum())
        if n_eff >= 3:
            rho, p = spearmanr(xvals[mask], yvals[mask])
        else:
            rho, p = np.nan, np.nan
        rho_s = "nan" if not np.isfinite(rho) else f"{rho:.3f}"
        p_s   = "nan" if not np.isfinite(p)   else f"{p:.1e}"

        # Compute zoom limits
        lims = _compute_zoom_limits(xvals, yvals, q=q_zoom)
        if lims is None:
            continue
        xlo, xhi, ylo, yhi = lims

        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        ax.scatter(xvals, yvals, s=18, label=rf"Spearman $\rho$={rho_s}, p={p_s}, n={n_eff}")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"REE ({ykey.replace('err_', '')})")
        ax.set_title(f"Estimation error vs. identifiability criterion (zoom {int(100*q_zoom)}th)")
        ax.legend(frameon=True, loc="best")
        fig.savefig(outdir / f"{x}_vs_{ykey}_{tag}_zoom.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
# ------------------------------
# CLI
# ------------------------------
def parse_estimators(s: str) -> list[str]:
    return [t.strip().lower() for t in s.split(",") if t.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=31415)
    ap.add_argument("--outdir", type=str, default="ident_vs_error_out_ext")
    ap.add_argument("--tag", type=str, default="demo")

    # New toggles
    ap.add_argument("--ensemble", type=str, default="ginibre",
                    choices=["ginibre", "stable", "sparse", "binary"],
                    help="Base ensemble for draw_with_ctrb_rank (affects base_c/base_u).")
    ap.add_argument("--deficiency", type=int, default=1,
                    help="d such that rank(Ctrb)=n-d. Default 1 (barely uncontrollable).")
    ap.add_argument("--estimators", type=str, default="dmdc,moesp",
                    help="Comma-separated list from {dmdc,moesp,sindy,node}.")
    ap.add_argument("--zoom", action="store_true",
                    help="Also save zoomed scatter plots capturing ≈80% of points in the lower-left (per-axis q≈0.9).")
    ap.add_argument("--zoom-q", type=float, default=0.9,
                    help="Per-axis quantile for zoom box (default 0.9 → ≥80% joint coverage).")

    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    plotdir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)

    est_list = parse_estimators(args.estimators)

    df, sysinfo = run_trials(
        n=args.n, m=args.m, T=args.T, dt=args.dt, trials=args.trials,
        noise_std=args.noise_std, seed=args.seed,
        ensemble=args.ensemble, deficiency=args.deficiency,
        estimators=est_list
    )
    df = add_transforms(df)

    # Save trial-wise results
    tag = f"{args.tag}_ens-{args.ensemble}_def-{args.deficiency}_ests-{'-'.join(sysinfo['estimators'])}"
    df.to_csv(outdir / f"results_{tag}.csv", index=False)

    # Spearman table over chosen estimators
    ycols = [c for c in df.columns if c.startswith("err_")]
    stab = spearman_table(df, ycols)
    stab.to_csv(outdir / f"spearman_{tag}.csv", index=False)

    # Plots for each estimator
    for y in ycols:
        scatter_plots(df, y, plotdir, tag)
        if args.zoom:
            scatter_plots_zoom(df, y, plotdir, tag, q_zoom=args.zoom_q)

    # Minimal metadata
    meta_path = outdir / f"system_{tag}.npz"
    np.savez(meta_path, A=sysinfo["A"], B=sysinfo["B"], Ad=sysinfo["Ad"], Bd=sysinfo["Bd"])

    print("Saved:")
    print("  ", outdir / f"results_{tag}.csv")
    print("  ", outdir / f"spearman_{tag}.csv")
    print("  ", meta_path)
    print("  plots ->", plotdir)

if __name__ == "__main__":
    main()
