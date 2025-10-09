#!/usr/bin/env python3
"""
Unidentifiability vs. input dimension m (continuous-time, sparse ensemble).

(1) For each m in {1,2,...,10}, generate N=200 random (A,B) with the
    **sparse_continuous** ensemble at fixed density p=0.8 and state dim n=5.
    Sample x0 ~ Unif(S^{n-1}).
(2) Compute binary criteria with tolerance `--tol`:
      - PBH margin: d_pbh = pbh_margin_structured(A,B,x0) → unident_pbh = (d_pbh <= tol)
      - Left-eigen: mu_min = min_i μ_i(A,[x0 B])          → unident_mu  = (mu_min <= tol)
      - Union: unident_any = unident_pbh OR unident_mu
(3) Plot % unidentifiable pairs (y-axis) vs input dimension m (x-axis),
    with the union as the main line and the two individual criteria as dotted lines.
"""
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import argparse

from ..config import ExperimentConfig
from ..ensembles import draw_with_ctrb_rank
from ..metrics import cont2discrete_zoh, pbh_margin_structured
from ..simulation import simulate_dt, prbs
from ..plots import plot_margin_scatter
from ..estimators import dmdc_tls

@dataclass
class UnderaConfig(ExperimentConfig):
    """Configuration for underactuated system experiments."""
    m_min: int = 1
    m_max: int = 10
    trials_per_m: int = 200
    density: float = 0.3
    tol: float = 1e-12
    outdir: str = "out_undera"

    def __post_init__(self):
        super().__post_init__()
        if self.m_max < self.m_min:
            raise ValueError(f"m_max ({self.m_max}) must be >= m_min ({self.m_min})")
        if self.trials_per_m < 1:
            raise ValueError(f"trials_per_m must be positive, got {self.trials_per_m}")

def run_one_trial(n: int, m: int, cfg: UnderaConfig, rng: np.random.Generator) -> Dict[str, Any]:
    """Run single trial with given input dimension."""
    # Generate system
    A, B, meta = draw_with_ctrb_rank(n, m, n-2, rng)
    Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)
    
    # Generate input and simulate
    U = prbs(m, cfg.T, scale=cfg.u_scale, dwell=cfg.dwell, rng=rng)
    x0 = rng.standard_normal(n)
    x0 /= np.linalg.norm(x0)
    
    X = simulate_dt(cfg.T, x0, Ad, Bd, U, noise_std=cfg.noise_std, rng=rng)
    
    # Identify
    X0, X1 = X[:, :-1], X[:, 1:]
    Ahat, Bhat = dmdc_tls(X0, X1, U)
    
    return {
        "n": n,
        "m": m,
        "errA": np.linalg.norm(Ahat - Ad, 'fro'),
        "errB": np.linalg.norm(Bhat - Bd, 'fro'),
        "margin": pbh_margin_structured(Ad, Bd, x0),
        "meta": {
            "cond_A": np.linalg.cond(A),
            "cond_Ad": np.linalg.cond(Ad),
            "timestamp": pd.Timestamp.now().isoformat()
        }
    }

def run_experiment(cfg: UnderaConfig) -> pd.DataFrame:
    """Run complete experiment sweep over input dimensions."""
    rng = np.random.default_rng(cfg.seed)
    results = []
    
    for m in range(cfg.m_min, cfg.m_max + 1):
        for _ in range(cfg.trials_per_m):
            result = run_one_trial(cfg.n, m, cfg, rng)
            results.append(result)
    
    df = pd.DataFrame(results)
    
    # Save results
    outdir = Path(cfg.outdir)
    outdir.mkdir(exist_ok=True)
    df.to_csv(outdir / "undera_sweep.csv", index=False)
    
    # Generate plots
    plot_margin_scatter(
        df["errA"], df["errB"], 
        df["margin"], df["m"],
        cfg.tol
    )
    
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--m-min", type=int, default=1)
    ap.add_argument("--m-max", type=int, default=10)
    ap.add_argument("--trials-per-m", type=int, default=200)
    ap.add_argument("--density", type=float, default=0.3)
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="out_undera")
    args = ap.parse_args()
    
    cfg = UnderaConfig(
        n=args.n,
        m_min=args.m_min,
        m_max=args.m_max,
        trials_per_m=args.trials_per_m,
        density=args.density,
        tol=args.tol,
        seed=args.seed,
        outdir=args.outdir
    )
    
    run_experiment(cfg)

if __name__ == "__main__":
    main()
