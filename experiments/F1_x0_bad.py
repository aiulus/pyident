"""
F1_x0_bad.py

Sweep d in {0,1,2,3} and, for each d, compare RANDOM x0 vs BAD x0
(where "bad" means BOTH PBH margin and σ_min(K_n) are near zero).
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
def run_experiment_for_d(cfg: ExperimentConfig, d: int, score_thr: float | None = None) -> Dict[str, Any]:
    """
    Run the x0-filtering experiment for deficiency d (controllability rank r = n-d).
    Returns dict with MSE arrays for RANDOM vs BAD (near-zero criteria) x0.
    """
    rng = np.random.default_rng(int(cfg.seed) + int(d) * 997)

    # Fixed CT pair with controllability rank r = n - d
    A, B, _ = draw_with_ctrb_rank(
        n=cfg.n, m=cfg.m, r=max(0, cfg.n - d), rng=rng,
        base_c="stable", base_u="stable"
    )
    Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)

    # Shared PRBS input for this d
    U = prbs(cfg.m, cfg.T, scale=cfg.u_scale, dwell=cfg.dwell, rng=rng)

    def trial(x0: np.ndarray) -> float:
        X = simulate_dt(cfg.T, x0, Ad, Bd, U, noise_std=cfg.noise_std, rng=rng)
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

    # BAD x0 thresholds (lower-tail). If cfg.q_filter=0.80 (for "good"), a natural
    q_bad = 0.10
    tau_logpbh_low_quant = float(np.quantile(logpbh_rand, q_bad))
    tau_logkry_low_quant = float(np.quantile(logkry_rand, q_bad))
    # If a hard cutoff is provided, it's interpreted in LOG-space (applied to BOTH metrics).
    # Otherwise, use the learned quantile lows.
    if score_thr is not None:
        tau_logpbh_low = float(score_thr)
        tau_logkry_low = float(score_thr)
    else:
        tau_logpbh_low = tau_logpbh_low_quant
        tau_logkry_low = tau_logkry_low_quant
    # Linear-scale cutoffs (for readability/printing)
    pbh_bad_cutoff    = float(np.exp(tau_logpbh_low))
    krylov_bad_cutoff = float(np.exp(tau_logkry_low))

    # BAD x0 selection: logPBH <= τ_low  AND  logσmin(K_n) <= τ_low
    mses_bad: List[float] = []
    logpbh_bad: List[float] = []
    logkry_bad: List[float] = []
    accepted, tries = 0, 0
    max_tries = 1000 * cfg.n_trials
    while accepted < cfg.n_trials and tries < max_tries:
        x0 = rng.standard_normal(cfg.n); x0 /= np.linalg.norm(x0) + EPS
        lp = _log_pbh(A, B, x0)
        lk = _log_krylov(A, B, x0)
        tries += 1
        if (lp <= tau_logpbh_low) and (lk <= tau_logkry_low):
            logpbh_bad.append(lp); logkry_bad.append(lk)
            mses_bad.append(trial(x0))
            accepted += 1

    if accepted < cfg.n_trials:
        raise RuntimeError(f"[d={d}] Could not find enough BAD x0 after {max_tries} tries")

    return {
        "d": d,
        "mse_random": np.asarray(mses_rand, float),
        "mse_bad":     np.asarray(mses_bad, float),
        "logpbh_random": np.asarray(logpbh_rand, float),
        "logkry_random": np.asarray(logkry_rand, float),
        "logpbh_bad":    np.asarray(logpbh_bad, float),
        "logkry_bad":    np.asarray(logkry_bad, float),
        "tau_logpbh_low": float(tau_logpbh_low),
        "tau_logkry_low": float(tau_logkry_low),
        "tau_logpbh_low_quantile": float(tau_logpbh_low_quant),
        "tau_logkry_low_quantile": float(tau_logkry_low_quant),
        "pbh_bad_cutoff": float(pbh_bad_cutoff),        
        "krylov_bad_cutoff": float(krylov_bad_cutoff),
        "meta": {"tries": int(tries), "max_tries": int(max_tries)},
    }


# --------------------------
# Boxplot grid
# --------------------------
def _global_limits(results: List[Dict[str, Any]]) -> Tuple[float, float]:
    all_mse = np.concatenate(
        [np.concatenate([out["mse_random"], out["mse_bad"]]) for out in results]
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
        data = [out["mse_random"], out["mse_bad"]]
        ax.boxplot(data, whis=(5, 95))
        ax.set_xticks([1, 2], labels=['random $x_0$', 'bad $x_0$'])
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
    ap.add_argument("--outdir", type=str, default="out_x0_bad", help="Directory to save figure")
    ap.add_argument("--dmax", type=int, default=3, help="Max deficiency d to include (starts at 0)")
    ap.add_argument("--score-thr", type=float, default=None,
                    help="If set, use this *hard* cutoff for BOTH PBH and σ_min(K_n). "
                         "By default interpreted in LOG-space; add --thr-linear to pass a linear cutoff.")
    ap.add_argument("--thr-linear", action="store_true",
                    help="Interpret --score-thr in linear scale (PBH and σ_min(K_n,norm)).")
     
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    cfg = ExperimentConfig()
    d_vals = [d for d in range(0, min(args.dmax, cfg.n-1) + 1)]

    print(f"Running BAD-x0 sweep over d = {d_vals} …")
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
        out = run_experiment_for_d(cfg, d, score_thr=thr_log)
        results.append(out)
        print(
            " d={d}: τ_low(logPBH)={tLP:.3e} (⇒ PBH ≤ {LP:.3e}), "
            "τ_low(logσmin(K_n))={tLK:.3e} (⇒ σ_min(K_n) ≤ {LK:.3e}), "
            "quantile lows: (PBH {qLP:.3e}, Krylov {qLK:.3e}), tries={tr}".format(
                d=d,
                tLP=out["tau_logpbh_low"], LP=out["pbh_bad_cutoff"],
                tLK=out["tau_logkry_low"], LK=out["krylov_bad_cutoff"],
                qLP=out["tau_logpbh_low_quantile"], qLK=out["tau_logkry_low_quantile"],
                tr=out["meta"]["tries"],
            )
         )

    print(f"Saving boxplot grid to {outdir} …")
    plot_box_grid(results, outdir / "f1_bad_box_grid.png")
    print("Done.")

if __name__ == "__main__":
    main()
