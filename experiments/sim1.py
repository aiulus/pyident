"""
Identifiability criteria vs. estimation error (DD-ID setting, with inputs).

What it does (faithful to the manuscript + your request):
  1) Fix (A,B) with controllability rank n-1 (barely uncontrollable) using `ensembles.draw_with_ctrb_rank`.
  2) For each trial: draw x0 ~ Unif(S^{n-1}), simulate DT system under PRBS inputs,
     estimate (A,B) using DMDc (pinv) and MOESP (full-state wrapper).
  3) Compute identifiability/robustness criteria:
        (a) structured PBH margin proxy: delta_pbh = pbh_margin_structured(A,B,x0)
        (b) Krylov generator statistics (we use sigma_min(K_n))
        (c) left-eigenvector criterion: mu_min = min_i ||w_i^T [x0 B]|| / ||w_i||
  4) Transform metrics per your request and plot on the x-axis:
        X1 = 1 / delta_pbh,   X2 = sigma_min(K_n),   X3 = 1 / mu_min
     Y = REE(Ahat) against the *discrete-time* A_d (Frobenius relative error).
  5) Compute Spearman rho + p-values and save CSVs + scatter plots.

Outputs:
  out_dir / (CSV: trial-wise results, CSV: spearman, PNGs: scatter plots)
"""
from __future__ import annotations
import argparse, math, sys, os, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, theilslopes

# Local modules expected alongside this script or in sys.path
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
sys.path.append("/mnt/data")  # fallback for the uploaded modules

from ..metrics import(
    pbh_margin_structured,
    unified_generator,
    left_eigvec_overlap,
    cont2discrete_zoh,
    visible_subspace,
    projected_errors,
    krylov_smin_norm
)
from ..estimators import(
    dmdc_pinv, 
    moesp_fit,
    sindy_fit,
    node_fit,
    project_identifiable
)
from ..ensembles import draw_with_ctrb_rank  
from ..metrics import pair_distance


# --- Orientation helpers: turn any metric into "badness" (higher = worse)
METRICS = {
    # larger is better → convert to badness with −log
    "pbh":    dict(col="pbh_struct",   orient="good", label="PBH badness (−log margin)"),
    "krylov": dict(col="krylov_smin",  orient="good", label="Krylov badness (−log σ_min K)"),
    "mu":     dict(col="mu_min",       orient="good", label="Left-eig overlap badness (−log μ_min)"),
    "gram":   dict(col="gram_lam_min", orient="good", label="Gramian badness (−log λ_min W_T)"),
    "ZV_s":   dict(col="ZV_smin",      orient="good", label="Regressor-on-V badness (−log σ_min Z_V)"),
    "Z_s":    dict(col="Z_smin",       orient="good", label="Regressor badness (−log σ_min Z)"),

    # larger is worse → keep as badness with +log
    "ZV_k":   dict(col="ZV_kappa",     orient="bad",  label="Regressor-on-V badness (log κ Z_V)"),
    "Z_k":    dict(col="Z_kappa",      orient="bad",  label="Regressor badness (log κ Z)"),
}

# === NEW: binning + stratified sampling helpers =====================

def _make_quantile_edges(vals: np.ndarray, n_bins: int) -> np.ndarray:
    """Unique quantile edges (0..1)."""
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(vals, q))
    if edges.size < 3:  # degenerate
        lo, hi = float(np.min(vals)), float(np.max(vals))
        edges = np.array([lo, (lo+hi)/2, hi], float)
    return edges

def _find_bin(edges: np.ndarray, v: float) -> int:
    """Return bin index in [0, len(edges)-2]. Values at the right edge go to the last bin."""
    j = int(np.searchsorted(edges, v, side="right") - 1)
    return max(0, min(j, edges.size - 2))

def _metric_badness_scalar(Ad: np.ndarray, Bd: np.ndarray, x0: np.ndarray, key: str, eps=1e-12) -> float:
    """Same semantics as metric_badness() but without needing a DataFrame."""
    if key == "pbh":
        from ..metrics import pbh_margin_structured
        m = float(pbh_margin_structured(Ad, Bd, x0))
        return float(-np.log(max(m, eps)))
    elif key == "krylov":
        from ..metrics import unified_generator
        K = unified_generator(Ad, Bd, x0, mode="unrestricted")
        smin = float(np.min(np.linalg.svd(K, compute_uv=False))) if K.size else 0.0
        return float(-np.log(max(smin, eps)))
    elif key == "mu":
        from ..metrics import left_eigvec_overlap
        Xaug = np.concatenate([x0.reshape(-1,1), Bd], axis=1)
        mu = left_eigvec_overlap(Ad, Xaug)
        m = float(np.min(mu)) if mu.size else 0.0
        return float(-np.log(max(m, eps)))
    else:
        raise ValueError(f"Unsupported key for stratification: {key}")

def _pilot_badness_pool(Ad, Bd, key, rng, n=4000, n_keep=4000):
    vals = []
    xs = []
    for _ in range(n):
        x0 = sample_unit_sphere(Ad.shape[0], rng)
        xs.append(x0)
        vals.append(_metric_badness_scalar(Ad, Bd, x0, key))
    vals = np.asarray(vals, float)
    xs = np.asarray(xs, float)
    # Keep all (or top-k if you wish to bias toward tails)
    return xs, vals



plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 140,
})


def metric_badness(df: pd.DataFrame, key: str, eps: float = 1e-12):
    spec = METRICS[key]
    v = df[spec["col"]].to_numpy()
    if spec["orient"] == "good":
        v = np.maximum(v, eps)
        x = -np.log(v)   # higher = worse
    else:
        v = np.maximum(v, 1.0)
        x = np.log(v)    # higher = worse
    return x, spec["label"]


def dt_augmented_gramian(Ad: np.ndarray, Bd: np.ndarray, x0: np.ndarray, T: int) -> np.ndarray:
    """
    Finite-horizon discrete-time Gramian for C=[x0 B]:
      W_T = sum_{k=0}^{T-1} A^k C C^T (A^T)^k
    """
    C = np.concatenate([x0.reshape(-1,1), Bd], axis=1)
    n = Ad.shape[0]
    W = np.zeros((n, n), dtype=float)
    Ak = np.eye(n)
    for _ in range(T):
        W += Ak @ (C @ C.T) @ Ak.T
        Ak = Ad @ Ak
    return W

def compute_core_metrics(A: np.ndarray, B: np.ndarray, x0: np.ndarray) -> dict:
    """
    Metrics on the CT pair (A,B) for the simple correlation plots:
      - PBH structured margin proxy: pbh_struct
      - Krylov σ_min on normalized K_n built from [x0 B]
      - Left-eigenvector overlap μ_min for Xaug=[x0 B]
    """
    # PBH margin (structured) on CT pair
    pbh_struct = float(pbh_margin_structured(A, B, x0))

    # Short, normalized Krylov K_n on (A,B,x0)
    # (assumes metrics.krylov_smin_norm exists; otherwise paste the fallback you prefer)
    krylov_smin = float(krylov_smin_norm(A, B, x0))

    # Left-eig overlap on CT pair
    Xaug = np.concatenate([x0.reshape(-1, 1), B], axis=1)
    mu = left_eigvec_overlap(A, Xaug)
    mu_min = float(np.min(mu)) if mu.size else 0.0

    return dict(pbh_struct=pbh_struct, krylov_smin=krylov_smin, mu_min=mu_min)


def sample_unit_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(n)
    nrm = np.linalg.norm(v)
    if nrm == 0.0:
        return sample_unit_sphere(n, rng)
    return v / nrm


def prbs(T: int, m: int, rng: np.random.Generator, dwell: int = 1) -> np.ndarray:
    """±1 PRBS with optional dwell; shape (T,m)."""
    steps = math.ceil(T / dwell)
    seq = rng.choice([-1.0, 1.0], size=(steps, m))
    u = np.repeat(seq, repeats=dwell, axis=0)[:T, :]
    return u


def simulate_dt(Ad: np.ndarray, Bd: np.ndarray, u: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """
    Simulate x_{k+1} = Ad x_k + Bd u_k.
    Returns X with shape (n, T+1).
    """
    n = Ad.shape[0]
    T = u.shape[0]
    X = np.empty((n, T + 1), dtype=float)
    X[:, 0] = x0
    for k in range(T):
        X[:, k + 1] = Ad @ X[:, k] + Bd @ u[k, :]
    return X

def relative_error_fro(Ahat: np.ndarray, Bhat: np.ndarray, Atrue: np.ndarray, Btrue:  np.ndarray) -> float:
    return pair_distance(Ahat, Bhat, Atrue, Btrue)



def run_trials(n=6, m=2, T=200, dt=0.05, trials=64, noise_std=0.0, seed=123):
    rng = np.random.default_rng(seed)

    # Keep the barely uncontrollable case as in your small script
    A, B, meta = draw_with_ctrb_rank(
        n, m, r=n - 1, rng=rng,
        ensemble_type="ginibre", base_u="ginibre", embed_random_basis=True
    )
    Ad, Bd = cont2discrete_zoh(A, B, dt)

    rows = []
    for t in range(trials):
        x0 = sample_unit_sphere(n, rng)
        u  = prbs(T, m, rng, dwell=1)
        X  = simulate_dt(Ad, Bd, u, x0)
        if noise_std > 0:
            X = X + noise_std * rng.standard_normal(X.shape)

        Xtrain, Xp, Utrain = X[:, :-1], X[:, 1:], u.T

        # Estimators (DT)
        A_dmdc, B_dmdc = dmdc_pinv(Xtrain, Xp, Utrain)
        A_moesp, B_moesp = moesp_fit(Xtrain, Xp, Utrain, n=n)

        # Simple targets: pair REE vs (Ad,Bd)
        ree_dmdc  = relative_error_fro(A_dmdc, B_dmdc, Ad, Bd)
        ree_moesp = relative_error_fro(A_moesp, B_moesp, Ad, Bd)

        # Metrics on (A,B,x0)
        crit = compute_core_metrics(A, B, x0)

        rows.append(dict(trial=t, **crit, err_dmdc=ree_dmdc, err_moesp=ree_moesp))

    df = pd.DataFrame(rows)
    return df, {"A": A, "B": B, "Ad": Ad, "Bd": Bd, "meta": meta}


# === fixed-(A,B) stratified trials =============================

def run_trials_stratified(
    n=6, m=2, T=200, dt=0.05, seed=123, *,
    estimator="dmdc", noise_std=0.0,
    stratify_by="pbh", n_bins=12, per_bin=12,
) -> pd.DataFrame:
    """
    Fix (A,B), draw x0's but accept/reject to hit ~per_bin samples per quantile bin
    of the chosen 'stratify_by' badness metric ('pbh'|'krylov'|'mu').
    Keeps all original metrics + error targets.
    """
    rng = np.random.default_rng(seed)

    # (A,B) full-ctrb for baseline comparability
    from ..ensembles import draw_with_ctrb_rank
    A, B, _ = draw_with_ctrb_rank(n, m, r=n, rng=rng, ensemble_type="ginibre", base_u="ginibre", embed_random_basis=True)
    Ad, Bd = cont2discrete_zoh(A, B, dt)

    # Persistently exciting inputs (same U for all trials)
    u = prbs(T, m, rng, dwell=1)
    Utrain = u.T

    # Pilot pool to build bin edges
    xs, vals = _pilot_badness_pool(Ad, Bd, stratify_by, rng, n=max(2000, 6*n_bins*per_bin))
    edges = _make_quantile_edges(vals, n_bins)
    counts = np.zeros(edges.size - 1, dtype=int)
    target_total = per_bin * (edges.size - 1)

    rows = []
    tries = 0
    while counts.sum() < target_total and tries < 200000:
        tries += 1
        x0 = sample_unit_sphere(n, rng)
        b = _metric_badness_scalar(Ad, Bd, x0, stratify_by)
        j = _find_bin(edges, b)
        if counts[j] >= per_bin:
            continue  # bin already full

        # Simulate + noise
        X  = simulate_dt(Ad, Bd, u, x0)
        if noise_std > 0:
            X = X + noise_std * rng.standard_normal(X.shape)

        X0, X1 = X[:, :-1], X[:, 1:]
        if estimator == "dmdc":
            Ahat, Bhat = dmdc_pinv(X0, X1, Utrain)
        else:   # moesp still available if desired
            Ahat, Bhat = moesp_fit(X0, X1, Utrain, n=n)

        # errors + metrics (DT, Z/ZV conditioning tracked here)
        crit = compute_core_metrics(Ad, Bd, x0)
        err_pair = relative_error_fro(Ahat, Bhat, Ad, Bd)
        P, _ = visible_subspace(Ad, Bd, x0)
        dA_V, dB_V = projected_errors(Ahat, Bhat, Ad, Bd, P)
        err_V = 0.5 * (dA_V + dB_V)
        Z = np.hstack([Ahat, Bhat]); Zt = np.hstack([Ad, Bd])  # for completeness if needed
        Th_true = np.hstack([Ad, Bd]); Th_hat = np.hstack([Ahat, Bhat])
        Th_id_true = project_identifiable(Th_true, np.vstack([X0, Utrain]))
        Th_id_hat  = project_identifiable(Th_hat, np.vstack([X0, Utrain]))
        err_ident = float(np.linalg.norm(Th_id_hat - Th_id_true, "fro"))
        err_unident = float(np.linalg.norm(Th_hat - Th_id_hat, "fro"))

        rows.append(dict(
            trial=len(rows), bin_idx=int(j), bin_key=stratify_by, badness=b,
            **crit,
            err_pair_dmdc=err_pair, err_V_dmdc=err_V,
            err_ident_dmdc=err_ident, err_unident_dmdc=err_unident,
            noise_std=float(noise_std), cohort="stratified", r=n  # full-ctrb baseline
        ))
        counts[j] += 1

    df = pd.DataFrame(rows)
    # for plotting labels
    mids = 0.5*(edges[:-1] + edges[1:])
    df["bin_mid"] = df["bin_idx"].map({i: mids[i] for i in range(mids.size)})
    return df

# === NEW: explicit A/B/C controls ===================================

def _pick_x0_extreme(Ad, Bd, rng, key, select="max", n_draw=4000):
    """
    Draw many x0 and pick the one that maximizes/minimizes *goodness* metric,
    then convert to badness orientation consistently.
    key in {'pbh','krylov','mu'}.
    """
    best_x0, best_val = None, None
    for _ in range(n_draw):
        x0 = sample_unit_sphere(Ad.shape[0], rng)
        # 'goodness' versions
        if key == "pbh":
            v = float(pbh_margin_structured(Ad, Bd, x0))
        elif key == "krylov":
            K = unified_generator(Ad, Bd, x0, mode="unrestricted")
            smin = float(np.min(np.linalg.svd(K, compute_uv=False))) if K.size else 0.0
            v = smin
        else:  # 'mu'
            Xaug = np.concatenate([x0.reshape(-1,1), Bd], axis=1)
            mu = left_eigvec_overlap(Ad, Xaug)
            v = float(np.min(mu)) if mu.size else 0.0
        if best_val is None:
            best_x0, best_val = x0, v
        else:
            if (select == "max" and v > best_val) or (select == "min" and v < best_val):
                best_x0, best_val = x0, v
    return best_x0

def run_controls(
    n=6, m=2, T=200, dt=0.05, seed=123, *,
    noise_std=0.0, per_cohort=64
) -> pd.DataFrame:
    """
    Three explicit cohorts:
      A) 'healthy PBH' (full-ctrb pair; pick x0 with large PBH margin)
      B) 'kill a visible direction' (full-ctrb pair; pick x0 with tiny μ_min)
      C) 'rank-deficient controllability' (r=n-1; random x0)
    """
    rng = np.random.default_rng(seed)

    # Full-ctrb pair for A/B
    A_f, B_f, _ = draw_with_ctrb_rank(n, m, r=n, rng=rng, ensemble_type="ginibre", base_u="ginibre", embed_random_basis=True)
    Ad_f, Bd_f = cont2discrete_zoh(A_f, B_f, dt)
    # Deficient pair for C
    A_d, B_d, _ = draw_with_ctrb_rank(n, m, r=max(0, n-1), rng=rng, ensemble_type="ginibre", base_u="ginibre", embed_random_basis=True)
    Ad_d, Bd_d = cont2discrete_zoh(A_d, B_d, dt)

    u = prbs(T, m, rng, dwell=1); Utrain = u.T

    cohorts = []
    # A: healthy PBH
    xA = _pick_x0_extreme(Ad_f, Bd_f, rng, key="pbh", select="max")
    # B: kill visible direction (μ_min small)
    xB = _pick_x0_extreme(Ad_f, Bd_f, rng, key="mu", select="min")

    for label, (Ad, Bd, x0_ref, r_rank) in [
        ("A_healthy_PBH", (Ad_f, Bd_f, xA, n)),
        ("B_kill_visible", (Ad_f, Bd_f, xB, n)),
        ("C_def_ctrb",     (Ad_d, Bd_d, None, max(0, n-1))),
    ]:
        for t in range(per_cohort):
            x0 = sample_unit_sphere(n, rng) if x0_ref is None else x0_ref
            X = simulate_dt(Ad, Bd, u, x0)
            if noise_std > 0:
                X = X + noise_std * rng.standard_normal(X.shape)
            X0, X1 = X[:, :-1], X[:, 1:]
            Ahat, Bhat = dmdc_pinv(X0, X1, Utrain)

            crit = compute_core_metrics(Ad, Bd, x0)
            err_pair = relative_error_fro(Ahat, Bhat, Ad, Bd)
            P, _ = visible_subspace(Ad, Bd, x0)
            dA_V, dB_V = projected_errors(Ahat, Bhat, Ad, Bd, P)
            err_V = 0.5 * (dA_V + dB_V)
            Th_true = np.hstack([Ad, Bd]); Th_hat = np.hstack([Ahat, Bhat])
            Th_id_true = project_identifiable(Th_true, np.vstack([X0, Utrain]))
            Th_id_hat  = project_identifiable(Th_hat, np.vstack([X0, Utrain]))
            err_ident = float(np.linalg.norm(Th_id_hat - Th_id_true, "fro"))
            err_unident = float(np.linalg.norm(Th_hat - Th_id_hat, "fro"))

            cohorts.append(dict(
                trial=len(cohorts), cohort=label, r=r_rank,
                **crit,
                err_pair_dmdc=err_pair, err_V_dmdc=err_V,
                err_ident_dmdc=err_ident, err_unident_dmdc=err_unident,
                noise_std=float(noise_std)
            ))
    return pd.DataFrame(cohorts)


# === NEW: mechanism cohorts (PBH vs excitation) =====================

def add_mechanism_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each row into one of 4 quadrants using upper/lower quartiles of:
     - PBH badness (−log margin)  -> large = structurally weak
     - Z_V conditioning (log κ(Z_V)) or (−log σ_min Z_V) -> large = poor excitation/geometry on V
    """
    out = df.copy()
    # build badness axes from df columns you already compute
    pbh_bad = -np.log(np.maximum(out["pbh_struct"].to_numpy(), 1e-18))
    zv_bad  = np.log(np.maximum(out["ZV_kappa"].to_numpy(), 1.0))  # alternative: -log(ZV_smin)
    q_p = np.quantile(pbh_bad, [0.25, 0.75])
    q_z = np.quantile(zv_bad,  [0.25, 0.75])

    labels = []
    for pb, zb in zip(pbh_bad, zv_bad):
        if pb >= q_p[1] and zb < q_z[0]:
            labels.append("weak_PBH_only")
        elif pb < q_p[0] and zb >= q_z[1]:
            labels.append("poor_ZV_only")
        elif pb >= q_p[1] and zb >= q_z[1]:
            labels.append("both_weak")
        else:
            labels.append("neither")
    out["mech_cohort"] = labels
    return out


# === NEW: bin/ribbon & cohort box/violin summaries ==================

def plot_bin_boxes(df: pd.DataFrame, ykey: str, outdir: pathlib.Path, tag: str):
    dd = df[df["cohort"]=="stratified"].copy()
    if dd.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    # order by bin mid
    mids = np.sort(dd["bin_mid"].unique())
    data = [dd.loc[dd["bin_mid"]==m, ykey].to_numpy() for m in mids]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(np.arange(1, len(mids)+1), [f"{m:.2g}" for m in mids], rotation=0)
    ax.set_xlabel(f"{dd['bin_key'].iloc[0]} badness bin (mid)")
    ax.set_ylabel(ykey)
    ax.set_title(f"Stratified equal-count bins — {ykey}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(outdir / f"bins_{dd['bin_key'].iloc[0]}_{ykey}_{tag}.png", dpi=150)
    plt.close(fig)

def plot_cohort_violins(df: pd.DataFrame, ykey: str, outdir: pathlib.Path, tag: str):
    dd = df.copy()
    groups = ["A_healthy_PBH", "B_kill_visible", "C_def_ctrb"]
    dd = dd[dd["cohort"].isin(groups)]
    if dd.empty:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    data = [dd.loc[dd["cohort"]==g, ykey].to_numpy() for g in groups]
    parts = ax.violinplot(data, positions=np.arange(1, len(groups)+1), showmeans=True, showextrema=False, widths=0.9)
    ax.set_xticks(np.arange(1, len(groups)+1), groups, rotation=0)
    ax.set_ylabel(ykey)
    ax.set_title(f"Controls (A/B/C) — {ykey}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(outdir / f"controls_viol_{ykey}_{tag}.png", dpi=150)
    plt.close(fig)



def add_transforms(df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    df = df.copy()
    df["x_inv_pbh"] = 1.0 / np.maximum(df["pbh_struct"].to_numpy(), eps)
    df["x_krylov"]  = df["krylov_smin"].to_numpy()                 # raw σ_min(K_n)
    df["x_inv_mu"]  = 1.0 / np.maximum(df["mu_min"].to_numpy(), eps)
    return df


def spearman_table(df: pd.DataFrame, ycols=("err_dmdc", "err_moesp")) -> pd.DataFrame:
    xcols = [("x_inv_pbh", "1 / PBH metric"),
             ("x_krylov",  "σ_min(K_n)"),
             ("x_inv_mu",  "1 / mu_min")]
    records = []
    for x, _ in xcols:
        rec = {"metric": x}
        for y in ycols:
            rho, p = spearmanr(df[x], df[y])
            rec[f"{y}_rho"] = str(rho)
            rec[f"{y}_p"] = str(p)
        records.append(rec)
    return pd.DataFrame(records)



def _boot_ci_stat(x, y, stat_fn, B=1000, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    n = len(x); idx = np.arange(n)
    boots = []
    for _ in range(B):
        ii = rng.choice(idx, size=n, replace=True)
        boots.append(stat_fn(x[ii], y[ii]))
    boots = np.array(boots)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return lo, hi

def rankcorr_badness_table(
    df: pd.DataFrame,
    ycols=("err_pair_dmdc", "err_V_dmdc", "err_ident_dmdc"),
    keys=("pbh","krylov","mu","gram","ZV_s","ZV_k","Z_s","Z_k"),
) -> pd.DataFrame:
    recs = []
    for k in keys:
        x, lab = metric_badness(df, k)
        for y in ycols:
            yv = df[y].to_numpy()
            r_s, p_s = spearmanr(x, yv)
            r_k, p_k = kendalltau(x, yv, variant="b")
            lo, hi = _boot_ci_stat(x, yv, lambda a,b: spearmanr(a,b)[0], B=800)
            recs.append(dict(
                metric=k, label=lab, target=y,
                spearman=f"{r_s:.3f}", sp_p=f"{p_s:.3g}", sp_ci=f"[{lo:.3f}, {hi:.3f}]",
                kendall=f"{r_k:.3f}",  kd_p=f"{p_k:.3g}",
            ))
    return pd.DataFrame(recs)


def spearman_kendall_table(df: pd.DataFrame,
                           ycols=("err_pair_dmdc","err_V_dmdc","err_ident_dmdc"),
                           xcols=(
                               ("pbh_log","PBH log"),
                               ("krylov_smin_log","log σ_min(K_n)"),
                               ("mu_log","log μ_min"),
                               ("gram_lam_min_log","log λ_min(W_T)"),
                               ("ZV_smin_log","log σ_min(Z_V)"),
                               ("ZV_kappa_log","log κ(Z_V)"),
                           )) -> pd.DataFrame:
    recs = []
    for x, xname in xcols:
        xv = df[x].to_numpy()
        for y in ycols:
            yv = df[y].to_numpy()
            r_s, p_s = spearmanr(xv, yv)
            r_k, p_k = kendalltau(xv, yv, variant="b")
            # bootstrap CIs for Spearman
            def _s(xx, yy): return spearmanr(xx, yy)[0]
            lo, hi = _boot_ci_stat(xv, yv, _s, B=800)
            recs.append(dict(
                metric=xname, target=y,
                spearman=f"{r_s:.3f}", sp_p=f"{p_s:.3g}", sp_ci=f"[{lo:.3f}, {hi:.3f}]",
                kendall=f"{r_k:.3f}", kd_p=f"{p_k:.3g}",
            ))
    return pd.DataFrame(recs)


def _add_binned_medians(ax, x, y, bins=20):
    q = np.linspace(0, 1, bins+1)
    edges = np.quantile(x, q)
    mids, meds = [], []
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        sel = (x >= lo) & (x <= hi)
        if sel.sum() >= 3:
            mids.append(np.median(x[sel]))
            meds.append(np.median(y[sel]))
    if mids:
        ax.plot(mids, meds, lw=2, alpha=0.9)

def _binned_summary(x, y, bins=16, qs=(0.10, 0.50, 0.90)):
    """
    Equal-count (quantile) bins in x, return bin centers + requested y-quantiles.
    """
    x = np.asarray(x); y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    q = np.linspace(0, 1, bins + 1)
    edges = np.quantile(x, q)
    # collapse duplicates if x is concentrated
    edges = np.unique(edges)
    if edges.size < 3:  # too few distinct bins
        return np.array([]), [np.array([]) for _ in qs]
    mids = []; quants = [[] for _ in qs]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x <= hi)
        if sel.sum() >= 4:
            mids.append(np.median(x[sel]))
            vals = np.quantile(y[sel], qs)
            for j, v in enumerate(vals):
                quants[j].append(v)
    mids = np.array(mids)
    quants = [np.array(col) for col in quants]
    return mids, quants

def scatter_plots_badness(
    df: pd.DataFrame, ykey: str, outdir: pathlib.Path, tag: str,
    keys=("pbh","krylov","mu","gram","ZV_s","ZV_k","Z_s","Z_k"),
):
    for k in keys:
        x, xlabel = metric_badness(df, k)
        y = df[ykey].to_numpy()

        # Winsorize a hair to avoid one spike flattening the axis
        y_clean = y.copy()
        lo, hi = np.quantile(y_clean, [0.01, 0.99])
        y_clean = np.clip(y_clean, lo, hi)

        # Figure
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        ax.scatter(x, y_clean, s=12, alpha=0.35, edgecolors="none")

        # Binwise ribbon (10–90%) + median curve
        mids, (q10, q50, q90) = _binned_summary(x, y_clean, bins=16, qs=(0.10, 0.50, 0.90))
        if mids.size:
            ax.fill_between(mids, q10, q90, alpha=0.20, linewidth=0)
            ax.plot(mids, q50, lw=2.2, alpha=0.95)

        # Robust global trend: Theil–Sen line + 95% slope band
        try:
            slope, intercept, lo_slope, hi_slope = theilslopes(y_clean, x)
            xfit = np.linspace(np.min(x), np.max(x), 200)
            yfit = slope * xfit + intercept
            ylo  = lo_slope * xfit + intercept
            yhi  = hi_slope * xfit + intercept
            ax.plot(xfit, yfit, ls="--", lw=1.6)
            ax.fill_between(xfit, ylo, yhi, alpha=0.12, linewidth=0)
        except Exception:
            pass  # fall back gracefully if degenerate

        # Log-y if highly skewed
        y_pos = y_clean[y_clean > 0]
        if y_pos.size and (y_pos.max() / max(y_pos.min(), 1e-16) > 30):
            ax.set_yscale("log")

        # Rank-correlation annotation
        r_s, p_s = spearmanr(x, y)
        r_k, p_k = kendalltau(x, y, variant="b")
        ax.text(0.02, 0.98, f"Spearman {r_s:.2f} (p={p_s:.2g})\nKendall {r_k:.2f} (p={p_k:.2g})",
                transform=ax.transAxes, va="top", ha="left", fontsize=9)

        ax.set_xlabel(xlabel + " (higher = worse)")
        ax.set_ylabel(ykey)
        ax.grid(True, linestyle="--", alpha=0.45, which="both")
        ax.set_title(f"{xlabel} vs {ykey}")
        fig.tight_layout()
        fig.savefig(outdir / f"{k}_bad_vs_{ykey}_{tag}.png", dpi=150)
        plt.close(fig)


def scatter_plots(df: pd.DataFrame, ykey: str, outdir: pathlib.Path, tag: str):
    pairs = [
        ("x_krylov",  "σ_min(K_n)"),
        ("x_inv_mu",  "1 / mu_min (left-eig overlap)"),
        ("x_inv_pbh", "1 / PBH structured margin"),
    ]
    for x, xlabel in pairs:
        xv = df[x].to_numpy(); yv = df[ykey].to_numpy()
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        ax.scatter(xv, yv, s=18)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"REE({ykey.replace('err_', '')})")
        ax.set_title(f"{xlabel} vs. {ykey}")
        fig.savefig(outdir / f"{x}_vs_{ykey}_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# --- Additions for stratified sampling, cohorts, and partial correlations ---

def _quantile_edges(x, bins=10):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return np.unique(np.quantile(x, np.linspace(0,1,bins+1)))

def _bin_index(x, edges):
    # last bin closed on right
    return np.minimum(np.searchsorted(edges, x, side="right")-1, len(edges)-2)

def collect_trials_stratified(
    Ad, Bd, n, m, T, rng, 
    target_key="pbh", n_per_bin=10, n_bins=10,
    z_control=None, z_q=(0.3, 0.7),
    noise_std=0.0, max_draws=20000,
):
    """
    Draw x0 and PRBS until each quantile bin (on target_key badness) has ~n_per_bin trials.
    Optionally keep only trials whose regressor-on-V conditioning ZV_kappa lies
    within quantiles z_q to break confounding by excitation.
    Returns a dataframe exactly like run_trials() builds.
    """
    rows = []
    # precompute badness label
    def _bad(dfrow):
        x, _ = metric_badness(pd.DataFrame([dfrow]), target_key)
        return float(x[0])

    # first pass to get edges (use a quick probe of random x0)
    probe = []
    for _ in range(200):
        x0 = sample_unit_sphere(n, rng)
        u  = prbs(T, m, rng, dwell=1)
        X  = simulate_dt(Ad, Bd, u, x0)
        if noise_std > 0: X = X + noise_std * rng.standard_normal(X.shape)
        Xtrain, Xp, Utrain = X[:, :-1], X[:, 1:], u.T
        crit = compute_core_metrics(Ad, Bd, x0)
        probe.append(crit)
    edges = _quantile_edges([_bad(r) for r in probe], bins=n_bins)

    # optional z-control quantile bands from probe
    if z_control is not None:
        z_vals = np.log([max(r[z_control], 1.0) for r in probe])
        z_lo, z_hi = np.quantile(z_vals, z_q)
    else:
        z_lo = z_hi = None

    # bin counters
    counts = np.zeros(n_bins, dtype=int)

    kept = 0
    draws = 0
    while kept < n_bins * n_per_bin and draws < max_draws:
        draws += 1
        x0 = sample_unit_sphere(n, rng)
        u  = prbs(T, m, rng, dwell=1)
        X  = simulate_dt(Ad, Bd, u, x0)
        if noise_std > 0: X = X + noise_std * rng.standard_normal(X.shape)
        Xtrain, Xp, Utrain = X[:, :-1], X[:, 1:], u.T

        # estimators
        A_dmdc, B_dmdc = dmdc_pinv(Xtrain, Xp, Utrain)
        A_moesp, B_moesp = moesp_fit(Xtrain, Xp, Utrain, n=n)

        # errors (same as run_trials)
        err_pair_dmdc = relative_error_fro(A_dmdc, B_dmdc, Ad, Bd)
        P, _ = visible_subspace(Ad, Bd, x0)
        dA_V_dmdc, dB_V_dmdc = projected_errors(A_dmdc, B_dmdc, Ad, Bd, P)
        err_V_dmdc   = 0.5 * (dA_V_dmdc + dB_V_dmdc)
        Z = np.vstack([Xtrain, Utrain])
        Th_true = np.hstack([Ad, Bd]); Th_dmdc = np.hstack([A_dmdc, B_dmdc])
        Th_true_id  = project_identifiable(Th_true, Z)
        Th_dmdc_id  = project_identifiable(Th_dmdc, Z)
        err_ident_dmdc = np.linalg.norm(Th_dmdc_id - Th_true_id, "fro")
        err_unident_dmdc = np.linalg.norm(Th_dmdc - Th_dmdc_id, "fro")

        # metrics
        crit = compute_core_metrics(Ad, Bd, x0)
        row = dict(
            **crit,
            err_pair_dmdc=err_pair_dmdc, err_V_dmdc=err_V_dmdc,
            err_ident_dmdc=err_ident_dmdc, err_unident_dmdc=err_unident_dmdc,
        )

        # z-control filtering
        if z_control is not None:
            zval = np.log(max(row[z_control], 1.0))
            if not (z_lo <= zval <= z_hi):
                continue

        # bin by target badness
        b = _bad(row)
        j = _bin_index(b, edges)
        if counts[j] >= n_per_bin:  # bin full
            continue

        # accept
        counts[j] += 1
        rows.append(row)
        kept += 1

    df = pd.DataFrame(rows)
    return df

def partial_spearman(x, y, ctrl):
    """
    Spearman partial correlation of x and y controlling for ctrl (one covariate),
    implemented by rank-transform + Pearson on residuals.
    """
    from scipy.stats import rankdata, pearsonr
    xr = rankdata(x); yr = rankdata(y); zr = rankdata(ctrl)
    Z = np.column_stack([np.ones_like(zr), zr])
    bx, *_ = np.linalg.lstsq(Z, xr, rcond=None)
    by, *_ = np.linalg.lstsq(Z, yr, rcond=None)
    rx = xr - Z @ bx; ry = yr - Z @ by
    r, p = pearsonr(rx, ry)
    return r, p

def cohort_boxplot(df, key_x, ykey, outpath, z_name=None, z_mask=None, title=None):
    """
    Box/violin summary for discrete cohorts (e.g., controls A/B/C or tertiles of a metric).
    If z_mask is provided, it’s used to filter (e.g., middle 40% of log ZV_kappa).
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    d = df.copy()
    if z_mask is not None:
        d = d[z_mask]

    # tertiles of key_x (already in df)
    q = np.quantile(d[key_x], [0, 1/3, 2/3, 1])
    bins = np.digitize(d[key_x], q[1:-1], right=True)
    labels = ["low", "mid", "high"]
    data = [d[ykey].to_numpy()[bins==i] for i in range(3)]

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.violinplot(data, showextrema=False)
    ax.boxplot(data, widths=0.2)
    ax.set_xticks([1,2,3]); ax.set_xticklabels(labels)
    ax.set_ylabel(ykey)
    ttl = title or f"{ykey} across {key_x} tertiles"
    if z_name:
        ttl += f"\n(filtered by {z_name})"
    ax.set_title(ttl)
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--trials", type=int, default=64)
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=31415)
    ap.add_argument("--outdir", type=str, default="ident_vs_error_out")
    ap.add_argument("--tag", type=str, default="demo")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    plotdir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)

    df, sysinfo = run_trials(
        n=args.n, m=args.m, T=args.T, dt=args.dt,
        trials=args.trials, noise_std=args.noise_std, seed=args.seed
    )
    df = add_transforms(df)

    # Save trial-wise results
    df.to_csv(outdir / f"results_{args.tag}.csv", index=False)

    # Spearman table (just for a number)
    stab = spearman_table(df)
    stab.to_csv(outdir / f"spearman_{args.tag}.csv", index=False)

    # Plots for both algos
    scatter_plots(df, "err_dmdc",  plotdir, args.tag)
    scatter_plots(df, "err_moesp", plotdir, args.tag)

    # Minimal metadata
    meta_path = outdir / f"system_{args.tag}.npz"
    np.savez(meta_path, A=sysinfo["A"], B=sysinfo["B"], Ad=sysinfo["Ad"], Bd=sysinfo["Bd"])

    print("Saved:")
    print("  ", outdir / f"results_{args.tag}.csv")
    print("  ", outdir / f"spearman_{args.tag}.csv")
    print("  ", meta_path)
    print("  plots ->", plotdir)


if __name__ == "__main__":
    main()
