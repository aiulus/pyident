"""
F1: Effect of initial-state filtering under noise (DT, sweep over d)
--------------------------------------------------------------------
Mirrors the clean pipeline but injects controlled noise and uses
noise-aware estimators only: TLS and IV.

We sweep d in --d-list, where controllability rank r = max(0, n - d).

Noise model:
  Process:    w_t ~ N(0, sigma_w^2 I)
  Input:      u_t_obs = u_t + eta_t,   eta_t ~ N(0, sigma_u^2 I)
  Measurement x_t_obs = x_t + v_t,     v_t   ~ N(0, sigma_x^2 I)

Estimator:
  --estimator {tls, iv}; IV uses --iv-lags L.

Outputs:
  One figure per plot type (box, ecdf, violin), with vertical subplots
  (one row per d) and shared axes.

Dependencies: numpy, matplotlib, pandas; uses pyident.*
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from ..config import ExperimentConfig
from ..ensembles import draw_with_ctrb_rank
from ..metrics import cont2discrete_zoh, pbh_margin_structured, x0_score, unified_generator
from ..estimators import dmdc_tls, dmdc_iv
from ..signals import prbs  

# --- scoring helpers (theory-only; larger log is better) ---
EPS = 1e-18

def _log_pbh(Ad: np.ndarray, Bd: np.ndarray, x0: np.ndarray) -> float:
    m = float(pbh_margin_structured(Ad, Bd, x0))
    return float(np.log(max(m, EPS)))

def _log_krylov(Ad: np.ndarray, Bd: np.ndarray, x0: np.ndarray) -> float:
    K = unified_generator(Ad, Bd, x0, mode="unrestricted")
    if K.size == 0:
        return -np.inf
    s = np.linalg.svd(K, compute_uv=False)
    smin = float(s.min()) if s.size else 0.0
    return float(np.log(max(smin, EPS)))


# -------------------------------
# Helpers
# -------------------------------

def _simulate_dt_with_noise(Ad: np.ndarray, Bd: np.ndarray, x0: np.ndarray,
                            U_clean: np.ndarray, rng: np.random.Generator,
                            sigma_w: float = 0.0) -> np.ndarray:
    """x_{t+1} = Ad x_t + Bd u_t + w_t, w_t ~ N(0, sigma_w^2 I). Returns X (n, T+1)."""
    n, m = Ad.shape[0], Bd.shape[1]
    if U_clean.ndim != 2 or U_clean.shape[1] != m:
        raise ValueError(f"U_clean must have shape (T, m); got {U_clean.shape}")
    T = U_clean.shape[0]
    X = np.zeros((n, T + 1), dtype=float)
    X[:, 0] = x0
    for t in range(T):
        w = sigma_w * rng.standard_normal(n) if sigma_w > 0 else 0.0
        X[:, t+1] = Ad @ X[:, t] + Bd @ U_clean[t, :] + w
    return X

def _trial_once(Ad: np.ndarray, Bd: np.ndarray, x0: np.ndarray, U_clean: np.ndarray,
                rng: np.random.Generator, estimator: str, iv_lags: int,
                sigma_x: float, sigma_u: float, sigma_w: float) -> float:
    """
    Run one noisy trial; return param MSE = 0.5(||Ahat-Ad||_F + ||Bhat-Bd||_F).
    """
    # Process noise in state dynamics
    X = _simulate_dt_with_noise(Ad, Bd, x0, U_clean, rng, sigma_w=sigma_w)

    # Input noise (actuator / sensing)
    U_obs = U_clean + (sigma_u * rng.standard_normal(U_clean.shape) if sigma_u > 0 else 0.0)

    # Measurement noise on states
    X_obs = X + (sigma_x * rng.standard_normal(X.shape) if sigma_x > 0 else 0.0)

    X0, X1 = X_obs[:, :-1], X_obs[:, 1:]

    if estimator == "tls":
        Ahat, Bhat = dmdc_tls(X0, X1, U_obs)
    elif estimator == "iv":
        Ahat, Bhat = dmdc_iv(X0, X1, U_obs, L=iv_lags)
    else:
        raise ValueError("estimator must be one of {'tls','iv'}")

    errA = float(np.linalg.norm(Ahat - Ad, "fro"))
    errB = float(np.linalg.norm(Bhat - Bd, "fro"))
    return 0.5 * (errA + errB)

def run_experiment_for_d_noisy(cfg: ExperimentConfig, d: int, *,
                               estimator: str, iv_lags: int,
                               sigma_x: float, sigma_u: float, sigma_w: float) -> Dict[str, Any]:
    """
    One sweep level at fixed d. Builds r = max(0, n - d), draws (A,B),
    discretizes, builds U, and compares random vs filtered x0.
    """
    rng = np.random.default_rng(cfg.seed + d)  # per-d deterministic shift

    r = max(0, cfg.n - d)
    A, B, meta = draw_with_ctrb_rank(
        n=cfg.n, m=cfg.m, r=r, rng=rng,
        ensemble_type="stable", base_u="stable"
    )
    Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)

    # Persistently exciting input (clean)
    U_clean = prbs(cfg.T, cfg.m, rng=rng)

    def draw_x0_unit():
        x0 = rng.standard_normal(cfg.n); x0 /= (np.linalg.norm(x0) + 1e-18)
        return x0

    mses_rand, logpbh_rand, logkry_rand = [], [], []
    for _ in range(cfg.n_trials):
        x0 = draw_x0_unit()
        lp = _log_pbh(Ad, Bd, x0)
        lk = _log_krylov(Ad, Bd, x0)
        logpbh_rand.append(lp)
        logkry_rand.append(lk)

        mses_rand.append(_trial_once(Ad, Bd, x0, U_clean, rng,
                                     estimator, iv_lags, sigma_x, sigma_u, sigma_w))

    # thresholds
    tau_logpbh = float(np.quantile(logpbh_rand, cfg.q_filter))
    tau_logkry = float(np.quantile(logkry_rand, cfg.q_filter))

    mses_filt, logpbh_filt, logkry_filt = [], [], []
    accepted = tries = 0
    max_tries = 1000 * cfg.n_trials
    while accepted < cfg.n_trials and tries < max_tries:
        x0 = draw_x0_unit()
        lp = _log_pbh(Ad, Bd, x0)
        lk = _log_krylov(Ad, Bd, x0)
        tries += 1

        if (lp >= tau_logpbh) and (lk >= tau_logkry):
            logpbh_filt.append(lp)
            logkry_filt.append(lk)
            mses_filt.append(_trial_once(Ad, Bd, x0, U_clean, rng,
                                         estimator, iv_lags, sigma_x, sigma_u, sigma_w))
            accepted += 1

    if accepted < cfg.n_trials:
        raise RuntimeError(f"[d={d}] Filtered: could not accept {cfg.n_trials} x0 within {max_tries} tries.")

    return {
        "d": int(d), "r": int(max(0, cfg.n - d)),
        "mse_random": np.asarray(mses_rand),
        "mse_filtered": np.asarray(mses_filt),
        "logpbh_random": np.asarray(logpbh_rand),
        "logpbh_filtered": np.asarray(logpbh_filt),
        "logkry_random": np.asarray(logkry_rand),
        "logkry_filtered": np.asarray(logkry_filt),
        "tau_logpbh": tau_logpbh,
        "tau_logkry": tau_logkry,
        "meta": {"tries": tries, "seed": cfg.seed,
                 "sigma_x": sigma_x, "sigma_u": sigma_u, "sigma_w": sigma_w}
    }




# -------------------------------
# Plotting (vertical grids, shared axes)
# -------------------------------

def _shared_limits(results: List[Dict[str, Any]]) -> Tuple[Tuple[float,float], Tuple[float,float]]:
    """Compute shared x/y limits for MSE (used as y in box/violin, x in ECDF)."""
    vals = []
    for out in results:
        vals.extend(out["mse_random"].tolist())
        vals.extend(out["mse_filtered"].tolist())
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    # small padding
    pad = 0.05 * (vmax - vmin + 1e-12)
    return (vmin - pad, vmax + pad), (vmin - pad, vmax + pad)

def _plot_box_grid(results: List[Dict[str, Any]], out_png: Path):
    ylims, _ = _shared_limits(results)
    fig, axes = plt.subplots(nrows=len(results), ncols=1, figsize=(6, 2.6*len(results)), sharex=False, sharey=True)
    if len(results) == 1:
        axes = [axes]
    for ax, out in zip(axes, results):
        ax.boxplot([out["mse_random"], out["mse_filtered"]], showfliers=False)
        ax.set_xticks([1, 2], labels=['random $x_0$', 'filtered $x_0$'])
        ax.set_ylabel('MSE')
        ax.set_title(f'MSE (d={out["d"]}, r={out["r"]})')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    axes[0].set_ylim(*ylims)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def _plot_ecdf_grid(results: List[Dict[str, Any]], out_png: Path):
    xlims, _ = _shared_limits(results)
    fig, axes = plt.subplots(nrows=len(results), ncols=1, figsize=(6, 2.6*len(results)), sharex=True, sharey=True)
    if len(results) == 1:
        axes = [axes]
    for ax, out in zip(axes, results):
        for arr, label in [(out["mse_random"], "random $x_0$"), (out["mse_filtered"], "filtered $x_0$")]:
            x = np.sort(arr)
            y = np.linspace(0, 1, x.size, endpoint=True)
            ax.plot(x, y, lw=1.6, label=label)
        ax.set_xlim(*xlims)
        ax.set_ylabel('ECDF')
        ax.set_title(f'ECDF (d={out["d"]}, r={out["r"]})')
        ax.grid(True, linestyle='--', alpha=0.6)
    axes[-1].set_xlabel('MSE')
    axes[0].legend(loc='lower right', frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def _plot_violin_grid(results: List[Dict[str, Any]], out_png: Path):
    ylims, _ = _shared_limits(results)
    fig, axes = plt.subplots(nrows=len(results), ncols=1, figsize=(6, 2.6*len(results)), sharex=False, sharey=True)
    if len(results) == 1:
        axes = [axes]
    for ax, out in zip(axes, results):
        parts = ax.violinplot([out["mse_random"], out["mse_filtered"]],
                              positions=[1, 2], showmeans=True, showextrema=False, widths=0.9)
        # cosmetic: centers as ticks
        ax.set_xticks([1, 2], labels=['random $x_0$', 'filtered $x_0$'])
        ax.set_ylabel('MSE')
        ax.set_title(f'Violin (d={out["d"]}, r={out["r"]})')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    axes[0].set_ylim(*ylims)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

# -------------------------------
# CLI
# -------------------------------

def parse_d_list(raw: str) -> List[int]:
    if raw is None or raw.strip() == "":
        return [0, 1, 2, 3]
    return [int(s) for s in raw.replace(",", " ").split()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="out_x0_sweep_noisy", help="Directory to save figures")
    ap.add_argument("--d-list", type=str, default="0,1,2,3", help="List of d values, e.g. '0,1,2,4'")
    ap.add_argument("--estimator", type=str, choices=["tls", "iv"], default="tls", help="Estimator to use")
    ap.add_argument("--iv-lags", type=int, default=2, help="Number of lags for IV (if estimator=iv)")
    ap.add_argument("--sigma-x", type=float, default=0.02, help="Std of measurement noise on states")
    ap.add_argument("--sigma-u", type=float, default=0.0, help="Std of input noise")
    ap.add_argument("--sigma-w", type=float, default=0.0, help="Std of process noise")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--T", type=int, default=None, help="Override cfg.T if provided")
    ap.add_argument("--u-scale", type=float, default=None, help="Override cfg.u_scale if provided")
    ap.add_argument("--q-filter", type=float, default=None, help="Quantile threshold in [0,1] for filtering")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True, parents=True)
    d_list = parse_d_list(args.d_list)

    cfg = ExperimentConfig()
    cfg.seed = args.seed
    cfg.n_trials = int(args.n_trials)
    if args.T is not None: cfg.T = int(args.T)
    if args.u_scale is not None: cfg.u_scale = float(args.u_scale)
    if args.q_filter is not None: cfg.q_filter = float(args.q_filter)

    results = []
    print(f"Running noisy sweep with estimator={args.estimator}  d_list={d_list}  "
          f"sigmas: x={args.sigma_x}, u={args.sigma_u}, w={args.sigma_w}")
    for d in d_list:
        out = run_experiment_for_d_noisy(
            cfg, d,
            estimator=args.estimator, iv_lags=args.iv_lags,
            sigma_x=args.sigma_x, sigma_u=args.sigma_u, sigma_w=args.sigma_w
        )
        results.append(out)
        print(f"  d={d}: τ_logPBH={out['tau_logpbh']:.3e} | τ_logσmin(K)={out['tau_logkry']:.3e} | "
              f"MSE medians random={np.median(out['mse_random']):.3e} "
              f"filtered={np.median(out['mse_filtered']):.3e}")

    # Grid plots with shared axes
    print(f"Saving figures to {outdir} ...")
    _plot_box_grid(results, outdir / "f1_noisy_box_grid.png")
    _plot_ecdf_grid(results, outdir / "f1_noisy_ecdf_grid.png")
    _plot_violin_grid(results, outdir / "f1_noisy_violin_grid.png")
    print("Done.")

if __name__ == "__main__":
    main()
