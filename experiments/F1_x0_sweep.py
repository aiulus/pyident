"""
Sweep d in {0,1,2,3} and plot vertical grids (shared axes) comparing RANDOM vs FILTERED x0.
Saves one grid figure per plot type.

Dependencies: numpy, scipy (via metrics), matplotlib, pandas
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy.stats import pearsonr, spearmanr

from ..config import ExperimentConfig
from ..ensembles import draw_with_ctrb_rank, controllability_rank
from ..metrics import cont2discrete_zoh, pbh_margin_structured, unified_generator, krylov_smin_norm
from ..simulation import simulate_dt, prbs
from ..estimators import dmdc_tls
from ..projectors import left_uncontrollable_subspace


# --- scoring helpers ---
EPS = 1e-18


def _log_pbh(A: np.ndarray, B: np.ndarray, x0: np.ndarray) -> float:
    m = float(pbh_margin_structured(A, B, x0))  # CT pair
    return float(np.log(max(m, EPS)))

def _log_krylov(A: np.ndarray, B: np.ndarray, x0: np.ndarray) -> float:
    smin = float(krylov_smin_norm(A, B, x0))    # normalized K_n on CT pair
    return float(np.log(max(smin, EPS)))

def _pbh_krylov_badness(Ad, Bd, x0, eps=1e-12):
    pbh = float(pbh_margin_structured(Ad, Bd, x0))
    K   = unified_generator(Ad, Bd, x0, mode="unrestricted")
    sK  = np.linalg.svd(K, compute_uv=False) if K.size else np.array([0.0])
    smin = float(sK.min()) if sK.size else 0.0
    return pbh, smin

# --------------------------
# Core experiment per d
# --------------------------
def run_experiment_for_d(cfg: ExperimentConfig, ensemble_type: str, seed: int, d: int, score_thr: float | None = None) -> Dict[str, Any]:
    """
    Run the x0-filtering experiment for a fixed deficiency d (so rank ctrb = n-d).
    Returns dict with REE arrays etc.
    """
    # Per-d RNG so runs are reproducible but distinct
    if seed is None:
        rng = np.random.default_rng(int(seed) + int(d) * 997)
    else:
        rng = np.random.default_rng(int(cfg.seed) + int(d) * 997)

    # Fixed continuous-time pair with controllability rank r = n - d
    A, B, meta = draw_with_ctrb_rank(
        n=cfg.n, m=cfg.m, r=max(0, cfg.n - d), rng=rng, ensemble_type=ensemble_type
    )
    Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)
    W_all = left_uncontrollable_subspace(A, B)
    rCT, _ = controllability_rank(A, B, order=A.shape[0], rtol=1e-8)
    print(f"[debug] d={d}: CT controllability rank r={rCT} (expect {A.shape[0]-d}), dark_dim={W_all.shape[1]}")

    # Shared input across all trials at this d (PE-ish PRBS)
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

    # RANDOM x0
    mses_rand: List[float] = []
    logpbh_rand: List[float] = []
    for _ in range(cfg.n_trials):
        x0 = rng.standard_normal(cfg.n)
        x0 /= np.linalg.norm(x0) + EPS
        lp = _log_pbh(A, B, x0)     
        logpbh_rand.append(lp)
        mses_rand.append(trial(x0))

    # thresholds learned from the random pool (upper-tail gate)
    tau_logpbh_quant = float(np.quantile(logpbh_rand, cfg.q_filter))
    # if a hard cutoff is provided, use it (in log-space) for BOTH metrics
    if score_thr is not None:
        tau_logpbh = float(score_thr)
    else:
        tau_logpbh = tau_logpbh_quant
    # Linear-scale cutoffs for readability:
    pbh_cutoff    = float(np.exp(tau_logpbh))

    # FILTERED x0 using theory-only AND-gate: logPBH >= τ_logPBH AND logσmin(K) >= τ_logK
    mses_filt: List[float] = []
    logpbh_filt: List[float] = []
    accepted, tries = 0, 0
    max_tries = 1000 * cfg.n_trials
    while accepted < cfg.n_trials and tries < max_tries:
        x0 = rng.standard_normal(cfg.n)
        x0 /= np.linalg.norm(x0) + EPS
        lp = _log_pbh(A, B, x0)    
        tries += 1
        if lp >= tau_logpbh:
            logpbh_filt.append(lp)
            mses_filt.append(trial(x0))
            accepted += 1

    if accepted < cfg.n_trials:
        raise RuntimeError(f"[d={d}] Could not find enough filtered x0 after {max_tries} tries")


    return {
        "d": d,
        "mse_random": np.asarray(mses_rand, float),
        "mse_filtered": np.asarray(mses_filt, float),

        # theory metrics (random + filtered) and thresholds
        "logpbh_random": np.asarray(logpbh_rand, float),
        "logpbh_filtered": np.asarray(logpbh_filt, float),

        "tau_logpbh": float(tau_logpbh),

        "tau_logpbh_quantile": float(tau_logpbh_quant),

        "pbh_cutoff": float(pbh_cutoff),                 # PBH ≥ val

        "meta": {
            "tries": int(tries),
            "max_tries": int(max_tries)
        }
    }



# --------------------------
# Plot helpers (grid versions)
# --------------------------
def _global_limits(results: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    """Compute global axis limits across all d for REE and for shift/ECDF."""
    all_mse = np.concatenate(
        [np.concatenate([out["mse_random"], out["mse_filtered"]]) for out in results]
    )
    # ECDF/box/violin use REE
    xmin, xmax = float(np.min(all_mse)), float(np.max(all_mse))
    ymin, ymax = xmin, xmax  # not used; but keep structure if needed

    # Shift-function (quantile differences)
    qs = np.linspace(0.1, 0.9, 9)
    all_diffs = []
    for out in results:
        a, b = out["mse_random"], out["mse_filtered"]
        qa = np.quantile(a, qs)
        qb = np.quantile(b, qs)
        all_diffs.append(qb - qa)
    diffs = np.concatenate(all_diffs)
    dmin, dmax = float(np.min(diffs)), float(np.max(diffs))

    return {
        "mse": (xmin, xmax),
        "diff": (dmin, dmax)
    }


def plot_box_grid(results: List[Dict[str, Any]], out_png: Path):
    lims = _global_limits(results)["mse"]
    nrows = len(results)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(6, 2.6*nrows), sharey=True)
    if nrows == 1:
        axes = [axes]
    for ax, out in zip(axes, sorted(results, key=lambda o: o["d"])):
        data = [out["mse_random"], out["mse_filtered"]]
        #ax.boxplot(data, showfliers=False)
        ax.boxplot(data, whis=(5,95))
        ax.set_xticks([1, 2], ['random $x_0$', 'filtered $x_0$'])
        ax.set_ylabel('Relative error')
        ax.set_title(f'Boxplot — deficiency $d={out["d"]}$')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        # Share y range across panels
        ax.set_ylim(lims[0] - 0.02*(lims[1]-lims[0]), lims[1] + 0.02*(lims[1]-lims[0]))
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_ecdf_grid(results: List[Dict[str, Any]], out_png: Path):
    xmin, xmax = _global_limits(results)["mse"]
    nrows = len(results)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(6, 2.6*nrows), sharex=True)
    if nrows == 1:
        axes = [axes]
    for ax, out in zip(axes, sorted(results, key=lambda o: o["d"])):
        for data, label in [(out["mse_random"], 'random $x_0$'),
                            (out["mse_filtered"], 'filtered $x_0$')]:
            x = np.sort(data)
            y = np.linspace(0, 1, x.size, endpoint=True)
            ax.step(x, y, where='post', label=label)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylabel('ECDF')
        ax.set_title(f'ECDF — deficiency $d={out["d"]}$')
        ax.legend(loc='lower right', fontsize=9)
    axes[-1].set_xlabel('Relative error')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_violin_grid(results: List[Dict[str, Any]], out_png: Path):
    ymin, ymax = _global_limits(results)["mse"]
    nrows = len(results)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(6, 2.6*nrows), sharey=True)
    if nrows == 1:
        axes = [axes]
    for ax, out in zip(axes, sorted(results, key=lambda o: o["d"])):
        parts = ax.violinplot(
            [out["mse_random"], out["mse_filtered"]],
            positions=[1, 2],
            showmeans=True,
            showextrema=False
        )
        # (no custom colors per instructions)
        ax.set_xticks([1, 2], ['random $x_0$', 'filtered $x_0$'])
        ax.set_ylabel('Relative error')
        ax.set_title(f'Violin — deficiency $d={out["d"]}$')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.set_ylim(ymin - 0.02*(ymax-ymin), ymax + 0.02*(ymax-ymin))
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, B: int = 2000, rng=None) -> Tuple[float, Tuple[float,float]]:
    rng = np.random.default_rng() if rng is None else rng
    diffs = np.empty(B, float)
    nA, nB = a.size, b.size
    for i in range(B):
        sa = a[rng.integers(0, nA, nA)]
        sb = b[rng.integers(0, nB, nB)]
        diffs[i] = np.mean(sb) - np.mean(sa)
    diffs.sort()
    md = float(np.mean(b) - np.mean(a))
    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    return md, (lo, hi)


def plot_effect_size_grid(results: List[Dict[str, Any]], out_png: Path):
    # Gardner–Altman style: left swarm (light), right mean diff + 95% CI
    # Shared y for raw REE (left), shared y for diff panel (right) — here we keep a single axis per row for simplicity.
    dmin, dmax = _global_limits(results)["diff"]
    nrows = len(results)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(6, 2.6*nrows), sharex=False)
    if nrows == 1:
        axes = [axes]
    for ax, out in zip(axes, sorted(results, key=lambda o: o["d"])):
        a, b = out["mse_random"], out["mse_filtered"]
        # Mean difference + 95% bootstrap CI
        md, (lo, hi) = _bootstrap_mean_diff(a, b, B=2000)
        # Plot difference as a point with CI bar along x=1
        ax.errorbar([1], [md], yerr=[[md-lo], [hi-md]], fmt='o', capsize=4)
        ax.axhline(0.0, linestyle='--', linewidth=1)
        ax.set_xlim(0.5, 1.5)
        ax.set_ylabel('Mean difference (filtered − random)')
        ax.set_title(f'Effect size — deficiency $d={out["d"]}$')
        ax.set_ylim(dmin - 0.02*(dmax-dmin), dmax + 0.02*(dmax-dmin))
        ax.set_xticks([1], ['Δ mean'])
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_shift_grid(results: List[Dict[str, Any]], out_png: Path):
    # Quantile shift function: q in [0.1,..,0.9], plot (Q_b - Q_a)(q)
    qs = np.linspace(0.1, 0.9, 9)
    dmin, dmax = _global_limits(results)["diff"]
    nrows = len(results)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(6, 2.6*nrows), sharex=True)
    if nrows == 1:
        axes = [axes]
    for ax, out in zip(axes, sorted(results, key=lambda o: o["d"])):
        a, b = out["mse_random"], out["mse_filtered"]
        da = np.quantile(a, qs)
        db = np.quantile(b, qs)
        diff = db - da
        ax.plot(qs, diff, marker='o', linewidth=1)
        ax.axhline(0.0, linestyle='--', linewidth=1)
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(dmin - 0.02*(dmax-dmin), dmax + 0.02*(dmax-dmin))
        ax.set_ylabel('Shift (filtered − random)')
        ax.set_title(f'Shift function — deficiency $d={out["d"]}$')
        ax.grid(True, linestyle='--', alpha=0.5)
    axes[-1].set_xlabel('Quantile q')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_crit_vs_err_grid(results: List[Dict[str, Any]], out_png: Path):
    # gather global ranges
    all_mse = np.concatenate([np.r_[r["mse_random"], r["mse_filtered"]] for r in results])
    all_logpbh = np.concatenate([np.r_[r["logpbh_random"], r["logpbh_filtered"]] for r in results])
    
    ypad = 0.05 * (all_mse.max() - all_mse.min() + 1e-12)
    xpad1 = 0.05 * (all_logpbh.max() - all_logpbh.min() + 1e-12)
    nrows = len(results)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(10, 2.6*nrows), sharey=True)
    if nrows == 1:
        axes = np.array([axes])

    for i, out in enumerate(sorted(results, key=lambda o: o["d"])):
        # left: log PBH vs REE
        ax = axes[i, 0]
        ax.scatter(out["logpbh_random"], out["mse_random"], s=14, alpha=0.6, label="random")
        ax.scatter(out["logpbh_filtered"], out["mse_filtered"], s=14, alpha=0.6, label="filtered")
        ax.set_title(f"log PBH vs REE — $d={out['d']}$")
        ax.set_xlabel("log PBH")

        ax.set_ylabel("Relative error")
        ax.grid(True, linestyle="--", alpha=0.6)
        if i == 0:
            ax.legend(frameon=False, fontsize=9)

        # right: log σ_min(K) vs REE
        ax = axes[i, 1]
        ax.set_title(f"log $\\sigma_{{\\min}}(K)$ vs REE — $d={out['d']}$")
        ax.set_xlabel("log $\\sigma_{\\min}(K)$")
        ax.grid(True, linestyle="--", alpha=0.6)

    for j in range(2):
        axes[0, j].set_ylim(all_mse.min()-ypad, all_mse.max()+ypad)
    axes[-1, 0].set_xlim(all_logpbh.min()-xpad1, all_logpbh.max()+xpad1)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_correlations(results: List[Dict[str, Any]], out_csv: Path) -> None:
    """
    Compute Pearson & Spearman correlations between criteria and relative error.
    Produces a tidy CSV with rows grouped by d and group (random/filtered/all).
    """
    rows: List[Dict[str, Any]] = []
    def _corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
        rP, pP = pearsonr(x, y)
        rS, pS = spearmanr(x, y)
        return (
            float(np.asarray(rP)), 
            float(np.asarray(pP)), 
            float(np.asarray(rS)), 
            float(np.asarray(pS))
        )
    for out in sorted(results, key=lambda o: o["d"]):
        d = int(out["d"])
        for grp in ("random", "filtered"):
            err = out[f"mse_{grp}"]
            lp  = out[f"logpbh_{grp}"]
            rP_lp, pP_lp, rS_lp, pS_lp = _corr(lp, err)
            rows.append({
                "d": d, "group": grp,
                "n": int(err.size),
                "pearson_logPBH": rP_lp, "pearson_logPBH_p": pP_lp,
                "spearman_logPBH": rS_lp, "spearman_logPBH_p": pS_lp,
            })
        # combined group per d
        err = np.r_[out["mse_random"], out["mse_filtered"]]
        lp  = np.r_[out["logpbh_random"], out["logpbh_filtered"]]
        rP_lp, pP_lp, rS_lp, pS_lp = _corr(lp, err)
        rows.append({
            "d": d, "group": "all",
            "n": int(err.size),
            "pearson_logPBH": rP_lp, "pearson_logPBH_p": pP_lp,
            "spearman_logPBH": rS_lp, "spearman_logPBH_p": pS_lp,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[correlations] saved to {out_csv}")
# --------------------------
# CLI
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxplot", action="store_true", help="Grid: boxplots across d")
    ap.add_argument("--ecdf", action="store_true", help="Grid: ECDFs across d")
    ap.add_argument("--violin", action="store_true", help="Grid: violins across d")
    ap.add_argument("--efs", action="store_true", help="Grid: effect-size (mean diff + 95% CI)")
    ap.add_argument("--shift", action="store_true", help="Grid: shift function (quantile diffs)")
    ap.add_argument("--outdir", type=str, default="out_x0_sweep", help="Directory to save figures")
    ap.add_argument("--dmax", type=int, default=3, help="Max deficiency d to include (starts at 0)")
    # optional hard cutoff track
    ap.add_argument("--score-thr", type=float, default=None,
                    help="If set, use this *hard* cutoff (applied to BOTH metrics with an AND-gate). "
                         "By default interpreted in LOG-space; add --thr-linear to pass a linear cutoff.")
    ap.add_argument("--thr-linear", action="store_true",
                    help="Interpret --score-thr in linear scale (PBH and σ_min(K_n,norm)).")
    ap.add_argument("--ensemble-type", type=str, default="ginibre", help="Ensemble type for the experiment.")
    ap.add_argument("--seed", type=int, help="Random seed for the experiment")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    cfg = ExperimentConfig()
    # Sanity: ensure n > dmax
    d_vals = [d for d in range(0, min(args.dmax, cfg.n-1) + 1)]

    print(f"Running sweep over d = {d_vals} …")
    results = []
    # Precompute log-space hard cutoff if requested
    if args.score_thr is not None:
        if args.thr_linear:
            if args.score_thr <= 0.0:
                raise ValueError("--score-thr must be > 0 when --thr-linear is used.")
            score_thr = float(np.log(args.score_thr))
        else:
            score_thr = float(args.score_thr)
    else:
        score_thr = None

    for d in d_vals:
        out = run_experiment_for_d(cfg, args.ensemble_type, args.seed, d, score_thr=score_thr)
        results.append(out)
        print(
            "d={d}: τ_logPBH={tLP:.3e} (⇒ PBH ≥ {LP:.3e}), "
            "quantiles: (PBH {qLP:.3e}), tries={tr}".format(
                d=out["d"],
                tLP=out["tau_logpbh"], LP=out["pbh_cutoff"],
                qLP=out["tau_logpbh_quantile"],
                tr=out["meta"]["tries"],
            )
        )
        
    # Default: make all if none selected
    if not any([args.boxplot, args.ecdf, args.violin, args.efs, args.shift]):
        args.boxplot = args.ecdf = args.violin = args.efs = args.shift = True

    print(f"Saving grid figures to {outdir} …")

    if args.boxplot:
        plot_box_grid(results, outdir / "f1_box_grid.png")
    if args.ecdf:
        plot_ecdf_grid(results, outdir / "f2_ecdf_grid.png")
    if args.violin:
        plot_violin_grid(results, outdir / "f3_violin_grid.png")
    if args.efs:
        plot_effect_size_grid(results, outdir / "f4_effectsize_grid.png")
    if args.shift:
        plot_shift_grid(results, outdir / "f5_shift_grid.png")

    # criterion-vs-error grid
    plot_crit_vs_err_grid(results, outdir / "f6_crit_vs_err_grid.png")

    # correlations 
    save_correlations(results, outdir / "correlations_by_d.csv")

    print("Done.")


if __name__ == "__main__":
    main()
