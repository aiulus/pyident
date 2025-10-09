# pyident/experiments/sim1_regimes.py
from __future__ import annotations
import argparse, math, sys, pathlib, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Module path setup
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
sys.path.append("/mnt/data")

try:
    from ..metrics import unified_generator, cont2discrete_zoh, pair_distance
    from ..estimators import dmdc_pinv, moesp_fit
    from ..ensembles import draw_with_ctrb_rank
except Exception:
    from metrics import unified_generator, cont2discrete_zoh, pair_distance
    from estimators import dmdc_pinv, moesp_fit
    from ensembles import draw_with_ctrb_rank

EPS = 1e-12

# ---------------- Core helpers ----------------
def sample_unit_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(n); nrm = np.linalg.norm(v)
    return v / (nrm if nrm > 0 else 1.0)

def prbs(T: int, m: int, rng: np.random.Generator, dwell: int = 1) -> np.ndarray:
    steps = math.ceil(T / dwell)
    seq = rng.choice([-1.0, 1.0], size=(steps, m))
    return np.repeat(seq, repeats=dwell, axis=0)[:T, :]

def simulate_dt(Ad: np.ndarray, Bd: np.ndarray, u: np.ndarray, x0: np.ndarray) -> np.ndarray:
    n, T = Ad.shape[0], u.shape[0]
    X = np.empty((n, T + 1), dtype=float); X[:, 0] = x0
    for k in range(T):
        X[:, k + 1] = Ad @ X[:, k] + Bd @ u[k, :]
    return X

def krylov_smin(A: np.ndarray, B: np.ndarray, x0: np.ndarray) -> float:
    K = unified_generator(A, B, x0, mode="unrestricted")
    if K.size == 0: return 0.0
    s = np.linalg.svd(K, compute_uv=False)
    return float(s.min()) if s.size else 0.0

def relative_error_fro(Ahat: np.ndarray, Bhat: np.ndarray, Atrue: np.ndarray, Btrue: np.ndarray) -> float:
    return float(pair_distance(Ahat, Bhat, Atrue, Btrue))

def mat_sparsity(M: np.ndarray, tol: float = 0.0) -> float:
    """Fraction of entries with |M_ij| <= tol."""
    return float(np.mean(np.abs(M) <= tol))

# ------------- Draw with constraints -------------
def draw_with_properties(n: int, m: int, d: int, ensemble: str,
                         target_sparsity: float | None,
                         sparsity_tol: float,
                         max_tries: int,
                         seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Draw (A,B) with controllability rank n-d via draw_with_ctrb_rank.
    If ensemble=='sparse' and target_sparsity is set, enforce achieved sparsity
    of both A and B within ±sparsity_tol via simple acceptance-rejection.
    """
    rng = np.random.default_rng(seed)
    r = max(0, n - int(d))
    tries, accepted = 0, False
    last_A = last_B = None
    meta: dict = {}

    while tries < max_tries and not accepted:
        tries += 1
        A, B, _meta = draw_with_ctrb_rank(
            n=n, m=m, r=r, rng=rng,
            ensemble_type=("sparse" if ensemble == "sparse" else ensemble),
            base_u=("sparse" if ensemble == "sparse" else ensemble),
            embed_random_basis=(ensemble != "sparse")
        )
        last_A, last_B = A, B
        if ensemble != "sparse" or target_sparsity is None:
            accepted, meta = True, _meta
        else:
            sA, sB = mat_sparsity(A), mat_sparsity(B)
            if (abs(sA - target_sparsity) <= sparsity_tol) and (abs(sB - target_sparsity) <= sparsity_tol):
                accepted = True
                meta = dict(_meta)
                meta.update(dict(achieved_sparsity_A=sA, achieved_sparsity_B=sB))

    if not accepted:
        # Fall back to last draw; still record achieved sparsity/tries
        A, B = last_A, last_B
        sA, sB = mat_sparsity(A), mat_sparsity(B)
        meta.update(dict(achieved_sparsity_A=sA, achieved_sparsity_B=sB, note="max_tries_exceeded"))

    meta["tries"] = tries
    meta["target_sparsity"] = (None if target_sparsity is None else float(target_sparsity))
    return A, B, meta

# ------------- One regime (many trials) -------------
def run_regime(n: int, m: int, d: int, *,
               ensemble: str,
               target_sparsity: float | None,
               sparsity_tol: float,
               max_draw_tries: int,
               T: int, dt: float, trials: int,
               base_seed: int,
               with_ree: bool,
               estimator: str = "moesp") -> dict:
    """
    Fix (n,m,d[,sparsity]); draw (A,B); ZOH; then over trials:
      - x0 ~ Unif(S^{n-1}), compute inv_smin = 1/sigma_min(K)
      - optionally compute REE via chosen estimator (fast: moesp or dmdc)
    Returns robust summary (medians/quantiles) and bookkeeping.
    """
    # Deterministic per-regime seed
    tag = f"n={n}|m={m}|d={d}|ens={ensemble}|s={target_sparsity}"
    h = int(hashlib.sha1(tag.encode()).hexdigest(), 16) & 0x7FFFFFFF
    seed = (base_seed ^ h) % (2**31 - 1)

    A, B, meta = draw_with_properties(
        n=n, m=m, d=d, ensemble=ensemble,
        target_sparsity=target_sparsity, sparsity_tol=sparsity_tol,
        max_tries=max_draw_tries, seed=seed
    )
    Ad, Bd = cont2discrete_zoh(A, B, dt)

    rng = np.random.default_rng(seed + 1337)
    inv_smin_vals = []
    ree_vals = []

    if with_ree:
        e = estimator.lower()
        if e == "moesp":
            est_fun = lambda X0, X1, U: moesp_fit(X0, X1, U, n=n)
        elif e == "dmdc":
            est_fun = lambda X0, X1, U: dmdc_pinv(X0, X1, U)
        else:
            raise ValueError("with_ree supports 'moesp' or 'dmdc' only.")

    for _ in range(trials):
        x0 = sample_unit_sphere(n, rng)
        smin = krylov_smin(A, B, x0)
        inv_smin_vals.append(1.0 / max(smin, EPS))

        if with_ree:
            u = prbs(T, m, rng, dwell=1)
            X = simulate_dt(Ad, Bd, u, x0)
            X0, X1, U = X[:, :-1], X[:, 1:], u.T
            Ahat, Bhat = est_fun(X0, X1, U)
            ree_vals.append(relative_error_fro(Ahat, Bhat, Ad, Bd))

    inv_smin = np.asarray(inv_smin_vals, float)
    def q(v, p): return float(np.quantile(v, p))
    summary = dict(
        n=int(n), m=int(m), deficiency=int(d), underact=float(n)/float(m),
        ensemble=ensemble,
        target_sparsity=(None if target_sparsity is None else float(target_sparsity)),
        achieved_sparsity_A=float(meta.get("achieved_sparsity_A", np.nan)),
        achieved_sparsity_B=float(meta.get("achieved_sparsity_B", np.nan)),
        tries=int(meta.get("tries", 0)),
        note=meta.get("note", ""),
        trials=int(trials), T=int(T), dt=float(dt),
        inv_smin_median=float(np.median(inv_smin)),
        inv_smin_p10=q(inv_smin, 0.10),
        inv_smin_p90=q(inv_smin, 0.90),
    )
    if with_ree:
        ree = np.asarray(ree_vals, float)
        summary.update(
            ree_median=float(np.nanmedian(ree)),
            ree_p10=float(np.nanquantile(ree, 0.10)),
            ree_p90=float(np.nanquantile(ree, 0.90)),
        )
    return summary

# ---------------- Utilities (parsing & plotting) ----------------
def parse_range(s: str) -> list[int]:
    """Parse 'a:b[:step]' or csv 'a,b,c' or single 'k' (ints)."""
    if "," in s: return [int(t) for t in s.split(",") if t.strip()]
    if ":" in s:
        parts = [int(t) for t in s.split(":")]
        a, b = parts[0], parts[1]; step = parts[2] if len(parts) >= 3 else 1
        return list(range(a, b + 1, step))
    return [int(s)]

def parse_float_range(s: str) -> list[float]:
    """Parse float range 'a:b[:step]' or csv 'a,b,c' or single 'k'."""
    if "," in s: return [float(t) for t in s.split(",") if t.strip()]
    if ":" in s:
        parts = [float(t) for t in s.split(":")]
        a, b = parts[0], parts[1]; step = parts[2] if len(parts) >= 3 else 0.1
        cnt = int(round((b - a) / step)) + 1
        return [a + i*step for i in range(cnt)]
    return [float(s)]

def plot_heat(Z: np.ndarray, xvals, yvals, xlabel: str, ylabel: str, title: str, out_png: pathlib.Path, cbar_label: str):
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    im = ax.imshow(Z, origin="lower", aspect="auto")
    ax.set_xticks(range(len(xvals))); ax.set_xticklabels([f"{v:g}" for v in xvals], rotation=45, ha="right")
    ax.set_yticks(range(len(yvals))); ax.set_yticklabels([f"{v:g}" for v in yvals])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax); cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------------- Grids (three couples) ----------------
def grid_sparsity_vs_n(*, s_grid: list[float], n_grid: list[int], m_fixed: int, d_fixed: int,
                       **kw_common) -> pd.DataFrame:
    rows = []
    for s in s_grid:
        for n in n_grid:
            row = run_regime(n=n, m=m_fixed, d=d_fixed, target_sparsity=s, **kw_common)
            row["axis_x"] = s; row["axis_y"] = n; row["grid"] = "s_vs_n"
            rows.append(row)
    return pd.DataFrame(rows)

def grid_sparsity_vs_d(*, s_grid: list[float], d_grid: list[int], n_fixed: int, m_fixed: int,
                       **kw_common) -> pd.DataFrame:
    rows = []
    for s in s_grid:
        for d in d_grid:
            row = run_regime(n=n_fixed, m=m_fixed, d=d, target_sparsity=s, **kw_common)
            row["axis_x"] = s; row["axis_y"] = d; row["grid"] = "s_vs_d"
            rows.append(row)
    return pd.DataFrame(rows)

def grid_n_vs_d(*, n_grid: list[int], d_grid: list[int], m_fixed: int, target_sparsity_fixed: float | None,
                **kw_common) -> pd.DataFrame:
    rows = []
    for n in n_grid:
        for d in d_grid:
            row = run_regime(n=n, m=m_fixed, d=d, target_sparsity=target_sparsity_fixed, **kw_common)
            row["axis_x"] = n; row["axis_y"] = d; row["grid"] = "n_vs_d"
            rows.append(row)
    return pd.DataFrame(rows)

def to_heat(df: pd.DataFrame, xvals, yvals, value_col: str) -> np.ndarray:
    """Build Z[y,x] array for heatmap."""
    Z = np.full((len(yvals), len(xvals)), np.nan, float)
    pos = {(xvals[i], yvals[j]): (j, i) for i in range(len(xvals)) for j in range(len(yvals))}
    for _, r in df.iterrows():
        key = (r["axis_x"], r["axis_y"])
        if key in pos:
            j, i = pos[key]
            Z[j, i] = float(r[value_col]) if value_col in r and pd.notna(r[value_col]) else np.nan
    return Z

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Regime heatmaps for Krylov and REE.")
    ap.add_argument("--outdir", type=str, default="out_regimes")
    ap.add_argument("--ensemble", type=str, default="ginibre",
                    choices=["ginibre", "stable", "sparse", "binary"])
    ap.add_argument("--s-grid", type=str, default="0.0:0.9:0.1",
                    help="Sparsity sweep when ensemble=='sparse' (fraction zeros).")
    ap.add_argument("--n-grid", type=str, default="5:30:1")
    ap.add_argument("--d-grid", type=str, default="0:15:1")
    ap.add_argument("--n-fixed", type=int, default=20)
    ap.add_argument("--m-fixed", type=int, default=5)
    ap.add_argument("--d-fixed", type=int, default=1)
    ap.add_argument("--s-fixed", type=float, default=None,
                    help="Fixed sparsity for n_vs_d grid (if ensemble=='sparse').")
    ap.add_argument("--sparsity-tol", type=float, default=0.05)
    ap.add_argument("--max-draw-tries", type=int, default=2000)

    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20250101)
    ap.add_argument("--with-ree", action="store_true")
    ap.add_argument("--estimator", type=str, default="moesp", choices=["moesp", "dmdc"])

    args = ap.parse_args()
    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    s_grid = parse_float_range(args.s_grid)
    n_grid = parse_range(args.n_grid)
    d_grid = parse_range(args.d_grid)

    common = dict(
        ensemble=args.ensemble,
        sparsity_tol=args.sparsity_tol,
        max_draw_tries=args.max_draw_tries,
        T=args.T, dt=args.dt, trials=args.trials,
        base_seed=args.seed, with_ree=args.with_ree,
        estimator=args.estimator
    )

    dfs: list[pd.DataFrame] = []

    # Figure couple 1: (sparsity, n) @ fixed d, m
    if args.ensemble == "sparse":
        df_s_n = grid_sparsity_vs_n(
            s_grid=s_grid, n_grid=n_grid,
            m_fixed=args.m_fixed, d_fixed=args.d_fixed,
            **{**common}
        )
        df_s_n.to_csv(outdir / "grid_s_vs_n.csv", index=False)
        dfs.append(df_s_n)

        # Heatmaps
        Z_k = to_heat(df_s_n, xvals=s_grid, yvals=n_grid, value_col="inv_smin_median")
        plot_heat(Z_k, s_grid, n_grid,
                  xlabel="sparsity (fraction zeros)", ylabel="state dimension n",
                  title=f"Krylov badness median (1/σ_min) — d={args.d_fixed}, m={args.m_fixed}, ens={args.ensemble}",
                  out_png=outdir / "heat_s_vs_n_krylov.png",
                  cbar_label="median(1/σ_min(K))")

        if args.with_ree:
            Z_r = to_heat(df_s_n, xvals=s_grid, yvals=n_grid, value_col="ree_median")
            plot_heat(Z_r, s_grid, n_grid,
                      xlabel="sparsity (fraction zeros)", ylabel="state dimension n",
                      title=f"REE median — d={args.d_fixed}, m={args.m_fixed}, ens={args.ensemble}",
                      out_png=outdir / "heat_s_vs_n_ree.png",
                      cbar_label="median(REE)")
    else:
        print("[note] ensemble is not 'sparse'; skipping sparsity×n heatmaps.")

    # Figure couple 2: (sparsity, deficiency) @ fixed n, m
    if args.ensemble == "sparse":
        df_s_d = grid_sparsity_vs_d(
            s_grid=s_grid, d_grid=d_grid,
            n_fixed=args.n_fixed, m_fixed=args.m_fixed,
            **{**common}
        )
        df_s_d.to_csv(outdir / "grid_s_vs_d.csv", index=False)
        dfs.append(df_s_d)

        Z_k = to_heat(df_s_d, xvals=s_grid, yvals=d_grid, value_col="inv_smin_median")
        plot_heat(Z_k, s_grid, d_grid,
                  xlabel="sparsity (fraction zeros)", ylabel="rank deficiency d",
                  title=f"Krylov badness median (1/σ_min) — n={args.n_fixed}, m={args.m_fixed}, ens={args.ensemble}",
                  out_png=outdir / "heat_s_vs_d_krylov.png",
                  cbar_label="median(1/σ_min(K))")

        if args.with_ree:
            Z_r = to_heat(df_s_d, xvals=s_grid, yvals=d_grid, value_col="ree_median")
            plot_heat(Z_r, s_grid, d_grid,
                      xlabel="sparsity (fraction zeros)", ylabel="rank deficiency d",
                      title=f"REE median — n={args.n_fixed}, m={args.m_fixed}, ens={args.ensemble}",
                      out_png=outdir / "heat_s_vs_d_ree.png",
                      cbar_label="median(REE)")
    else:
        print("[note] ensemble is not 'sparse'; skipping sparsity×deficiency heatmaps.")

    # Figure couple 3: (n, deficiency) @ fixed m, (optionally fixed sparsity)
    df_n_d = grid_n_vs_d(
        n_grid=n_grid, d_grid=d_grid,
        m_fixed=args.m_fixed,
        target_sparsity_fixed=(args.s_fixed if args.ensemble == "sparse" else None),
        **{**common}
    )
    df_n_d.to_csv(outdir / "grid_n_vs_d.csv", index=False)
    dfs.append(df_n_d)

    Z_k = to_heat(df_n_d, xvals=n_grid, yvals=d_grid, value_col="inv_smin_median")
    plot_heat(Z_k, n_grid, d_grid,
              xlabel="state dimension n", ylabel="rank deficiency d",
              title=f"Krylov badness median (1/σ_min) — m={args.m_fixed}, ens={args.ensemble}, s_fixed={args.s_fixed}",
              out_png=outdir / "heat_n_vs_d_krylov.png",
              cbar_label="median(1/σ_min(K))")

    if args.with_ree:
        Z_r = to_heat(df_n_d, xvals=n_grid, yvals=d_grid, value_col="ree_median")
        plot_heat(Z_r, n_grid, d_grid,
                  xlabel="state dimension n", ylabel="rank deficiency d",
                  title=f"REE median — m={args.m_fixed}, ens={args.ensemble}, s_fixed={args.s_fixed}",
                  out_png=outdir / "heat_n_vs_d_ree.png",
                  cbar_label="median(REE)")

    # Master CSV
    master = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if not master.empty:
        master.to_csv(outdir / "regimes_master.csv", index=False)
        print("Saved outputs to:", outdir)
        for f in sorted(outdir.iterdir()):
            if f.suffix in {".csv", ".png"}:
                print("  ", f)
    else:
        print("No grids were produced (check ensemble/sparsity settings).")

if __name__ == "__main__":
    main()
