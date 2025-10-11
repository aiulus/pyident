"""
F1_x0_good.py

Sweep d in {0,1,2,3} and, for each d, compare RANDOM x0 vs GOOD x0
(where "good" means the chosen score (PBH or Krylov) is above a high quantile).
Saves a single vertical grid of boxplots (shared y) across d.

Dependencies: numpy, matplotlib, pandas
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from ..config import ExperimentConfig
from ..ensembles import draw_with_ctrb_rank
from ..metrics import cont2discrete_zoh, pbh_margin_structured, krylov_smin_norm
from ..simulation import simulate_dt, prbs
from ..estimators import dmdc_tls

# --- scoring helpers (CT metrics; same orientation as the refactored scripts) ---
EPS = 1e-18


def _log_pbh(A: np.ndarray, B: np.ndarray, x0: np.ndarray) -> float:
    m = float(pbh_margin_structured(A, B, x0))  # CT pair
    return float(np.log(max(m, EPS)))


def _log_krylov(A: np.ndarray, B: np.ndarray, x0: np.ndarray) -> float:
    smin = float(krylov_smin_norm(A, B, x0))    # normalized K_n on CT pair
    return float(np.log(max(smin, EPS)))


# --------------------------
# Core experiment per d
# --------------------------

def run_experiment_for_d(
    cfg: ExperimentConfig,
    ensembletype: str,
    d: int,
    score_thr: float | None = None,
    *,
    filter_metric: str = "pbh",
) -> Dict[str, Any]:
    """
    Run the x0-filtering experiment for deficiency d (controllability rank r = n-d).
    Returns dict with MSE arrays for RANDOM vs GOOD (high-score criteria) x0.
    """
    rng = np.random.default_rng(int(cfg.seed) + int(d) * 997)

    # Fixed CT pair with controllability rank r = n - d
    A, B, _ = draw_with_ctrb_rank(
        n=cfg.n, m=cfg.m, r=max(0, cfg.n - d), rng=rng,
        ensemble_type=ensembletype
    )
    Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)

    # Shared PRBS input for this d
    U = prbs(cfg.T, cfg.m, scale=cfg.u_scale, dwell=cfg.dwell, rng=rng)

    def trial(x0: np.ndarray) -> float:
        X = simulate_dt(x0, Ad, Bd, U, noise_std=cfg.noise_std, rng=rng)
        X0, X1 = X[:, :-1], X[:, 1:]
        Ahat, Bhat = dmdc_tls(X0, X1, U)
        errA = np.linalg.norm(Ahat - Ad, 'fro')
        errB = np.linalg.norm(Bhat - Bd, 'fro')
        return float(0.5 * (errA + errB))

    # RANDOM x0 pool
    mses_rand: List[float] = []
    logpbh_rand: List[float] = []
    logkry_rand: List[float] = []
    for _ in range(cfg.n_trials):
        x0 = rng.standard_normal(cfg.n); x0 /= np.linalg.norm(x0) + EPS
        lp = _log_pbh(A, B, x0)
        lk = _log_krylov(A, B, x0)
        logpbh_rand.append(lp); logkry_rand.append(lk)
        mses_rand.append(trial(x0))

    # GOOD x0 thresholds (upper-tail).
    tau_logpbh_high_quant = float(np.quantile(logpbh_rand, cfg.q_filter))
    tau_logkry_high_quant = float(np.quantile(logkry_rand, cfg.q_filter))
    # If a hard cutoff is provided, it's interpreted in LOG-space (applied to BOTH metrics).
    # Otherwise, use the learned quantile highs.
    if score_thr is not None:
        tau_logpbh_high = float(score_thr)
        tau_logkry_high = float(score_thr)
    else:
        tau_logpbh_high = tau_logpbh_high_quant
        tau_logkry_high = tau_logkry_high_quant

    if filter_metric == "pbh":
        tau_selected = tau_logpbh_high
    elif filter_metric == "krylov":
        tau_selected = tau_logkry_high
    else:
        raise ValueError("filter_metric must be 'pbh' or 'krylov'")

    # Linear-scale cutoffs (for readability/printing)
    pbh_good_cutoff    = float(np.exp(tau_logpbh_high))
    krylov_good_cutoff = float(np.exp(tau_logkry_high))

    # GOOD x0 selection: score >= τ_high where score depends on filter_metric
    mses_good: List[float] = []
    logpbh_good: List[float] = []
    logkry_good: List[float] = []
    accepted, tries = 0, 0
    max_tries = 1000 * cfg.n_trials
    while accepted < cfg.n_trials and tries < max_tries:
        x0 = rng.standard_normal(cfg.n); x0 /= np.linalg.norm(x0) + EPS
        lp = _log_pbh(A, B, x0)
        lk = _log_krylov(A, B, x0)
        s = lp if filter_metric == "pbh" else lk
        tries += 1
        if s >= tau_selected:
            logpbh_good.append(lp); logkry_good.append(lk)
            mses_good.append(trial(x0))
            accepted += 1

    if accepted < cfg.n_trials:
        raise RuntimeError(f"[d={d}] Could not find enough GOOD x0 after {max_tries} tries")

    return {
        "d": d,
        "mse_random": np.asarray(mses_rand, float),
        "mse_good":     np.asarray(mses_good, float),
        "logpbh_random": np.asarray(logpbh_rand, float),
        "logkry_random": np.asarray(logkry_rand, float),
        "logpbh_good":    np.asarray(logpbh_good, float),
        "logkry_good":    np.asarray(logkry_good, float),
        "tau_logpbh_high": float(tau_logpbh_high),
        "tau_logkry_high": float(tau_logkry_high),
        "tau_selected": float(tau_selected),
        "tau_logpbh_high_quantile": float(tau_logpbh_high_quant),
        "tau_logkry_high_quantile": float(tau_logkry_high_quant),
        "pbh_good_cutoff": float(pbh_good_cutoff),
        "krylov_good_cutoff": float(krylov_good_cutoff),
        "filter_metric": filter_metric,
        "meta": {"tries": int(tries), "max_tries": int(max_tries)},
    }


# --------------------------
# Boxplot grid
# --------------------------

def _global_limits(results: List[Dict[str, Any]]) -> Tuple[float, float]:
    all_mse = np.concatenate(
        [np.concatenate([out["mse_random"], out["mse_good"]]) for out in results]
    )
    lo, hi = float(np.min(all_mse)), float(np.max(all_mse))
    pad = 0.02 * (hi - lo + 1e-12)
    return lo - pad, hi + pad


def plot_box_grid(results: List[Dict[str, Any]], out_png: Path):
    ylo, yhi = _global_limits(results)
    nrows = len(results)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(6, 2.6*nrows), sharey=True)
    if nrows == 1:
        axes = [axes]
    for ax, out in zip(axes, sorted(results, key=lambda o: o["d"])):
        data = [out["mse_random"], out["mse_good"]]
        ax.boxplot(data, whis=(5, 95))
        ax.set_xticks([1, 2], labels=['random $x_0$', 'good $x_0$'])
        ax.set_ylabel('MSE')
        ax.set_title(f'Boxplot — deficiency $d={out["d"]}$')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.set_ylim(ylo, yhi)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# --------------------------
# CLI
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="out_x0_good", help="Directory to save figure")
    ap.add_argument("--dmax", type=int, default=3, help="Max deficiency d to include (starts at 0)")
    ap.add_argument("--score-thr", type=float, default=None,
                    help="If set, use this *hard* cutoff for BOTH PBH and σ_min(K_n). "
                         "By default interpreted in LOG-space; add --thr-linear to pass a linear cutoff.")
    ap.add_argument("--thr-linear", action="store_true",
                    help="Interpret --score-thr in linear scale (PBH and σ_min(K_n,norm)).")
    ap.add_argument("--ensemble-type", type=str, default="ginibre")
    ap.add_argument(
        "--filter-metric",
        type=str,
        default="pbh",
        choices=("pbh", "krylov"),
        help="Metric used to score x0 when constructing the GOOD pool.",
    )

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    cfg = ExperimentConfig()
    d_vals = [d for d in range(0, min(args.dmax, cfg.n-1) + 1)]

    print(f"Running GOOD-x0 sweep over d = {d_vals} …")
    # Prepare log-space threshold if provided
    if args.score_thr is not None:
        if args.thr_linear:
            if args.score_thr <= 0.0:
                raise ValueError("--score-thr must be > 0 when --thr-linear is used.")
            thr_log = float(np.log(args.score_thr))
        else:
            thr_log = float(args.score_thr)
    else:
        thr_log = None

    results = []
    for d in d_vals:
        out = run_experiment_for_d(
            cfg,
            args.ensemble_type,
            d,
            score_thr=thr_log,
            filter_metric=args.filter_metric,
        )
        results.append(out)
        print(
            " d={d}: τ_high(logPBH)={tLP:.3e} (⇒ PBH ≥ {LP:.3e}), "
            "τ_high(logσmin(K_n))={tLK:.3e} (⇒ σ_min(K_n) ≥ {LK:.3e}), "
            "quantile highs: (PBH {qLP:.3e}, Krylov {qLK:.3e}), "
            "selected τ={ts:.3e} via {fm}, tries={tr}".format(
                d=d,
                tLP=out["tau_logpbh_high"], LP=out["pbh_good_cutoff"],
                tLK=out["tau_logkry_high"], LK=out["krylov_good_cutoff"],
                qLP=out["tau_logpbh_high_quantile"], qLK=out["tau_logkry_high_quantile"],
                ts=out["tau_selected"], fm=out["filter_metric"], tr=out["meta"]["tries"],
            )
        )

    print(f"Saving boxplot grid to {outdir} …")
    plot_box_grid(results, outdir / "f1_good_box_grid.png")
    print("Done.")


if __name__ == "__main__":
    main()