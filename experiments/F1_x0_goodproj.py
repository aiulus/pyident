from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy.stats import pearsonr, spearmanr

from ..config import ExperimentConfig
from ..ensembles import draw_with_ctrb_rank
from ..metrics import cont2discrete_zoh, pbh_margin_structured, unified_generator, krylov_smin_norm
from ..simulation import simulate_dt, prbs
from ..estimators import dmdc_tls
from ..projectors import build_projected_x0, left_uncontrollable_subspace

EPS = 1e-18

# --- scoring helpers ---

def _log_pbh(A: np.ndarray, B: np.ndarray, x0: np.ndarray) -> float:
    return float(np.log(max(float(pbh_margin_structured(A, B, x0)), EPS)))

def _log_krylov(A: np.ndarray, B: np.ndarray, x0: np.ndarray) -> float:
    return float(np.log(max(float(krylov_smin_norm(A, B, x0)), EPS)))

# --------------------------
# Core experiment per d
# --------------------------

def run_experiment_for_d(cfg: ExperimentConfig, d: int, score_thr: float | None = None,
                         k_off: int = 1) -> Dict[str, Any]:
    rng = np.random.default_rng(int(cfg.seed) + int(d) * 997)

    A, B, meta = draw_with_ctrb_rank(
        n=cfg.n, m=cfg.m, r=max(0, cfg.n - d), rng=rng, base_c="stable", base_u="stable"
    )
    Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)

    U = prbs(cfg.T, cfg.m, scale=cfg.u_scale, dwell=cfg.dwell, rng=rng)

    def trial(x0: np.ndarray) -> float:
        X = simulate_dt(x0, Ad, Bd, U, noise_std=cfg.noise_std, rng=rng)
        X0, X1 = X[:, :-1], X[:, 1:]
        Ahat, Bhat = dmdc_tls(X0, X1, U)
        nA = np.linalg.norm(Ad, 'fro') + EPS
        nB = np.linalg.norm(Bd, 'fro') + EPS
        errA = np.linalg.norm(Ahat - Ad, 'fro') / nA
        errB = np.linalg.norm(Bhat - Bd, 'fro') / nB
        return float(0.5 * (errA + errB))

    # RANDOM pool
    mses_rand: List[float] = []
    logpbh_rand: List[float] = []
    logkry_rand: List[float] = []
    x0_rand: List[np.ndarray] = []
    for _ in range(cfg.n_trials):
        x0 = rng.standard_normal(cfg.n); x0 /= np.linalg.norm(x0) + EPS
        x0_rand.append(x0)
        logpbh_rand.append(_log_pbh(A, B, x0))
        logkry_rand.append(_log_krylov(A, B, x0))
        mses_rand.append(trial(x0))

    # thresholds (upper tail) either from quantile or hard log cutoff
    tau_logpbh_quant = float(np.quantile(logpbh_rand, cfg.q_filter))
    tau_logkry_quant = float(np.quantile(logkry_rand, cfg.q_filter))

    if score_thr is not None:
        tau_logpbh = float(score_thr)
        tau_logkry = float(score_thr)
    else:
        tau_logpbh = tau_logpbh_quant
        tau_logkry = tau_logkry_quant

    pbh_cutoff    = float(np.exp(tau_logpbh))
    krylov_cutoff = float(np.exp(tau_logkry))

    # FILTERED (upper-tail AND-gate)
    mses_filt: List[float] = []
    logpbh_filt: List[float] = []
    logkry_filt: List[float] = []
    seeds: List[np.ndarray] = []

    for x0, lp, lk in zip(x0_rand, logpbh_rand, logkry_rand):
        if (lp >= tau_logpbh) and (lk >= tau_logkry):
            seeds.append(x0)
            logpbh_filt.append(lp)
            logkry_filt.append(lk)
            mses_filt.append(trial(x0))
        if len(seeds) >= cfg.n_trials:
            break

    if len(seeds) < cfg.n_trials:
        # top up by sampling until enough seeds satisfy the gate
        while len(seeds) < cfg.n_trials:
            x0 = rng.standard_normal(cfg.n); x0 /= np.linalg.norm(x0) + EPS
            lp = _log_pbh(A, B, x0); lk = _log_krylov(A, B, x0)
            if (lp >= tau_logpbh) and (lk >= tau_logkry):
                seeds.append(x0)
                logpbh_filt.append(lp)
                logkry_filt.append(lk)
                mses_filt.append(trial(x0))

    # PROJECTED: darken k_off uncontrollable directions starting from filtered seeds
    W_all = left_uncontrollable_subspace(A, B)
    dark_dim = int(W_all.shape[1])

    mses_proj: List[float] = []
    logpbh_proj: List[float] = []
    logkry_proj: List[float] = []

    for x0_seed in seeds:
        x0_proj, W_off, _ = build_projected_x0(A, B, x0_seed, k_off=k_off, rng=rng)
        if W_off.shape[1] > 0 and np.linalg.norm(W_off.T @ x0_proj) > 1e-8:
            continue  # safety
        logpbh_proj.append(_log_pbh(A, B, x0_proj))
        logkry_proj.append(_log_krylov(A, B, x0_proj))
        mses_proj.append(trial(x0_proj))

    # ensure equal number of trials across groups
    n_use = min(len(mses_rand), len(mses_filt), len(mses_proj))

    return {
        "d": d,
        "dark_dim": dark_dim,
        "mse_random": np.asarray(mses_rand[:n_use], float),
        "mse_filtered": np.asarray(mses_filt[:n_use], float),
        "mse_projected": np.asarray(mses_proj[:n_use], float),
        "logpbh_random": np.asarray(logpbh_rand[:n_use], float),
        "logkry_random": np.asarray(logkry_rand[:n_use], float),
        "logpbh_filtered": np.asarray(logpbh_filt[:n_use], float),
        "logkry_filtered": np.asarray(logkry_filt[:n_use], float),
        "logpbh_projected": np.asarray(logpbh_proj[:n_use], float),
        "logkry_projected": np.asarray(logkry_proj[:n_use], float),
        "tau_logpbh": float(tau_logpbh),
        "tau_logkry": float(tau_logkry),
        "tau_logpbh_quantile": float(tau_logpbh_quant),
        "tau_logkry_quantile": float(tau_logkry_quant),
        "pbh_cutoff": float(pbh_cutoff),
        "krylov_cutoff": float(krylov_cutoff),
    }

# --------------------------
# Plot helpers (now generic over groups)
# --------------------------

def _gather_all(results: List[Dict[str, Any]], keys: List[str]) -> np.ndarray:
    return np.concatenate([np.concatenate([r[k] for k in keys]) for r in results])

GROUPS = [
    ("random",   "mse_random"),
    ("filtered", "mse_filtered"),
    ("projected","mse_projected"),
]

LABEL = {
    "random":    "random $x_0$",
    "filtered":  "filtered $x_0$",
    "projected": "projected–dark $x_0$",
}


def _global_limits(results: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    all_mse = _gather_all(results, [g for _, g in GROUPS])
    xmin, xmax = float(np.min(all_mse)), float(np.max(all_mse))
    # For shift/effect-size, compute pairwise differences (projected − filtered) as representative
    q = np.linspace(0.1, 0.9, 9)
    diffs = []
    for r in results:
        qa = np.quantile(r["mse_filtered"], q)
        qb = np.quantile(r["mse_projected"], q)
        diffs.append(qb - qa)
    diffs = np.concatenate(diffs) if len(diffs) else np.array([0.0])
    dmin, dmax = float(np.min(diffs)), float(np.max(diffs))
    return {"mse": (xmin, xmax), "diff": (dmin, dmax)}


def plot_box_grid(results: List[Dict[str, Any]], out_png: Path):
    lims = _global_limits(results)["mse"]
    nrows = len(results)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(7.2, 2.8*nrows), sharey=True)
    if nrows == 1:
        axes = [axes]
    for ax, out in zip(axes, sorted(results, key=lambda o: o["d"])):
        data = [out[g] for _, g in GROUPS]
        ax.boxplot(data, whis=(5,95))
        ax.set_xticks([1,2,3], [LABEL[k] for k,_ in GROUPS])
        ax.set_ylabel('Relative error')
        ax.set_title(f"Boxplot — deficiency $d={out['d']}$ (dark_dim={out['dark_dim']})")
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.set_ylim(lims[0] - 0.02*(lims[1]-lims[0]), lims[1] + 0.02*(lims[1]-lims[0]))
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

# (Other plotters — ECDF/violin/effect-size/shift — can be updated analogously;

# --------------------------
# Correlations (extend to 3 groups)
# --------------------------

def save_correlations(results: List[Dict[str, Any]], out_csv: Path) -> None:
    rows: List[Dict[str, Any]] = []
    def _corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
        rP, pP = pearsonr(x, y); rS, pS = spearmanr(x, y)
        return float(rP), float(pP), float(rS), float(pS)

    for out in sorted(results, key=lambda o: o["d"]):
        d = int(out["d"])
        for grp_name, mse_key in GROUPS:
            err = out[mse_key]
            lp_key = f"logpbh_{grp_name}"
            lk_key = f"logkry_{grp_name}"
            if lp_key not in out or lk_key not in out:
                continue
            lp, lk = out[lp_key], out[lk_key]
            rP_lp, pP_lp, rS_lp, pS_lp = _corr(lp, err)
            rP_lk, pP_lk, rS_lk, pS_lk = _corr(lk, err)
            rows.append({
                "d": d, "group": grp_name, "n": int(err.size),
                "pearson_logPBH": rP_lp, "pearson_logPBH_p": pP_lp,
                "spearman_logPBH": rS_lp, "spearman_logPBH_p": pS_lp,
                "pearson_logSigmaK": rP_lk, "pearson_logSigmaK_p": pP_lk,
                "spearman_logSigmaK": rS_lk, "spearman_logSigmaK_p": pS_lk,
            })
        # combined row
        err = _gather_all([out], [g for _, g in GROUPS])
        lp  = _gather_all([out], [f"logpbh_{nm}" for nm,_ in GROUPS if f"logpbh_{nm}" in out])
        lk  = _gather_all([out], [f"logkry_{nm}" for nm,_ in GROUPS if f"logkry_{nm}" in out])
        if err.size and lp.size and lk.size:
            rP_lp, pP_lp, rS_lp, pS_lp = _corr(lp, err)
            rP_lk, pP_lk, rS_lk, pS_lk = _corr(lk, err)
            rows.append({
                "d": d, "group": "all", "n": int(err.size),
                "pearson_logPBH": rP_lp, "pearson_logPBH_p": pP_lp,
                "spearman_logPBH": rS_lp, "spearman_logPBH_p": pS_lp,
                "pearson_logSigmaK": rP_lk, "pearson_logSigmaK_p": pP_lk,
                "spearman_logSigmaK": rS_lk, "spearman_logSigmaK_p": pS_lk,
            })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[correlations] saved to {out_csv}")

# --------------------------
# CLI
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxplot", action="store_true", help="Grid: boxplots across d")
    ap.add_argument("--outdir", type=str, default="out_x0_sweep", help="Directory to save figures")
    ap.add_argument("--dmax", type=int, default=3, help="Max deficiency d to include (starts at 0)")
    ap.add_argument("--score-thr", type=float, default=None,
                    help="If set, use this *hard* log cutoff for BOTH metrics (AND-gate). Use --thr-linear for linear scale.")
    ap.add_argument("--thr-linear", action="store_true",
                    help="Interpret --score-thr in linear scale (PBH and σ_min(K_n,norm)).")
    ap.add_argument("--k-off", type=int, default=1, help="# uncontrollable directions to darken in projection")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True, parents=True)

    cfg = ExperimentConfig()
    d_vals = [d for d in range(0, min(args.dmax, cfg.n-1) + 1)]

    if args.score_thr is not None:
        score_thr = float(np.log(args.score_thr)) if args.thr_linear else float(args.score_thr)
    else:
        score_thr = None

    print(f"Running sweep over d = {d_vals} …")
    results = []
    for d in d_vals:
        out = run_experiment_for_d(cfg, d, score_thr=score_thr, k_off=args.k_off)
        results.append(out)
        print(
            "d={d}: dark_dim={dd}, τ_logPBH={tLP:.3e} (⇒ PBH ≥ {LP:.3e}), "
            "τ_logσmin(K)={tLK:.3e} (⇒ σ_min(K) ≥ {LK:.3e})".format(
                d=out["d"], dd=out["dark_dim"],
                tLP=out["tau_logpbh"], LP=out["pbh_cutoff"],
                tLK=out["tau_logkry"], LK=out["krylov_cutoff"],
            )
        )

    if not args.boxplot:
        args.boxplot = True

    print(f"Saving grid figures to {outdir} …")
    if args.boxplot:
        plot_box_grid(results, outdir / "f1_box_grid.png")

    save_correlations(results, outdir / "correlations_by_d.csv")
    print("Done.")

if __name__ == "__main__":
    main()
