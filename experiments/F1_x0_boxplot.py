"""
Experiment: effect of initial-state selection on one-step ID accuracy (DT)
-----------------------------------------------------------------------------
We construct a *fixed* continuous-time pair (A,B) with controllability rank n-2,
embed it randomly, discretize by ZOH to (Ad,Bd), and compare the parameter MSE of
least-squares identification from a single trajectory under two designs for x0:

  (1) RANDOM  : x0 ~ uniform on unit sphere S^{n-1}
  (2) FILTERED: draw many x0 ~ S^{n-1}, keep those with large structured PBH
                 margin m(x0) := min_{λ∈spec(Ad)} σ_min([λ I − Ad, [x0  Bd]])

We report the distribution (boxplots) of parameter MSE across trials.

Dependencies: numpy, scipy (expm, null_space), matplotlib
"""
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
import argparse

from ..config import ExperimentConfig
from ..ensembles import draw_with_ctrb_rank
from ..metrics import cont2discrete_zoh, pbh_margin_structured, unified_generator
from ..simulation import simulate_dt, prbs
from ..estimators import dmdc_tls
from ..plots import (
    plot_ecdf,
    plot_violin_swarm,
    plot_effect_size,
    plot_shift_function
)

EPS = 1e-18

def _log_pbh(Ad: np.ndarray, Bd: np.ndarray, x0: np.ndarray) -> float:
    m = float(pbh_margin_structured(Ad, Bd, x0))
    return float(np.log(max(m, EPS)))

def _log_krylov(Ad: np.ndarray, Bd: np.ndarray, x0: np.ndarray) -> float:
    # K = [ [x0 B], A[x0 B], ..., A^{n-1}[x0 B] ]
    K = unified_generator(Ad, Bd, x0, mode="unrestricted")
    if K.size == 0:
        return -np.inf
    svals = np.linalg.svd(K, compute_uv=False)
    smin = float(svals.min()) if svals.size else 0.0
    return float(np.log(max(smin, EPS)))


def run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Run initial state filtering experiment."""
    rng = np.random.default_rng(cfg.seed)
    
    # Generate system with rank(R) = n-2
    A, B, meta = draw_with_ctrb_rank(
        cfg.n, cfg.m, cfg.n - 4, rng,
        base_c="stable", base_u="stable"
    )

    Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)
    
    # Generate input
    U = prbs(cfg.m, cfg.T, scale=cfg.u_scale, dwell=cfg.dwell, rng=rng)
    
    def trial(x0):
        """Run single trial and return MSE."""
        X = simulate_dt(cfg.T, x0, Ad, Bd, U, 
                       noise_std=cfg.noise_std, rng=rng)
        X0, X1 = X[:, :-1], X[:, 1:]
        Ahat, Bhat = dmdc_tls(X0, X1, U)
        errA = np.linalg.norm(Ahat - Ad, 'fro')
        errB = np.linalg.norm(Bhat - Bd, 'fro')
        return 0.5 * (errA + errB)
    
    # Random initial states
    mses_rand = []
    margins_rand = []
    for _ in range(cfg.n_trials):
        x0 = rng.standard_normal(cfg.n)
        x0 /= np.linalg.norm(x0)
        margin = pbh_margin_structured(Ad, Bd, x0)
        margins_rand.append(margin)
        mses_rand.append(trial(x0))
    
    # Compute threshold
    tau = np.quantile(margins_rand, cfg.q_filter)
    
    # Filtered initial states with timeout
    mses_filt = []
    margins_filt = []
    accepted = 0
    tries = 0
    max_tries = 1000 * cfg.n_trials
    
    while accepted < cfg.n_trials and tries < max_tries:
        x0 = rng.standard_normal(cfg.n)
        x0 /= np.linalg.norm(x0)
        margin = pbh_margin_structured(Ad, Bd, x0)
        tries += 1
        if margin >= tau:
            margins_filt.append(margin)
            mses_filt.append(trial(x0))
            accepted += 1
    
    if accepted < cfg.n_trials:
        raise RuntimeError(f"Could not find enough filtered x0 after {max_tries} tries")
    
    return {
        "mse_random": np.array(mses_rand),
        "mse_filtered": np.array(mses_filt),
        "margins_random": np.array(margins_rand),
        "margins_filtered": np.array(margins_filt),
        "tau": tau,
        "meta": {
            "tries": tries,
            "max_tries": max_tries,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxplot", action="store_true", help="Generate a boxplot")
    ap.add_argument("--ecdf", action="store_true", help="Generate an ECDF plot")
    ap.add_argument("--violin", action="store_true", help="Generate a violin plot")
    ap.add_argument("--efs", action="store_true", help="Generate an effect-size plot")
    ap.add_argument("--shift", action="store_true", help="Generate a shift function plot")
    ap.add_argument("--outdir", type=str, default="out_x0_boxplot", help="Directory to save figures")
    args = ap.parse_args()

    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    
    cfg = ExperimentConfig()
    out = run_experiment(cfg)
    
    # If no plot type is specified, generate all of them.
    if not any([args.boxplot, args.ecdf, args.violin, args.efs, args.shift]):
        args.boxplot = args.ecdf = args.violin = args.efs = args.shift = True
    
    print(f"Saving figures to {outdir}...")

    if args.boxplot:
        fig, ax = plt.subplots()
        ax.boxplot([out["mse_random"], out["mse_filtered"]])
        ax.set_xticks([1, 2], ['random $x_0$', 'filtered $x_0$'])
        ax.set_ylabel('MSE')
        ax.set_title('MSE Comparison: Random vs. Filtered x0')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        fig.tight_layout()
        fig.savefig(outdir / "f1_x0_comp_boxplot.png", dpi=150)
        plt.close(fig)
    
    # The following functions use `out_png` to save and close the figure.
    if args.ecdf:
        plot_ecdf([out["mse_random"], out["mse_filtered"]], 
                 ["random $x_0$", "filtered $x_0$"],
                 out_png=outdir / "f1_x0_comp_ecdf.png")
    
    if args.violin:
        plot_violin_swarm(out["mse_random"], out["mse_filtered"],
                          out_png=outdir / "f1_x0_comp_violin.png")
    
    if args.efs:
        plot_effect_size(out["mse_random"], out["mse_filtered"],
                         out_png=outdir / "f1_x0_comp_efs.png")
    
    if args.shift:
        plot_shift_function(out["mse_random"], out["mse_filtered"],
                            out_png=outdir / "f1_x0_comp_shift.png")

    print("Done.")

if __name__ == "__main__":
    main()
