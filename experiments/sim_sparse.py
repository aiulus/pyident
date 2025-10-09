"""
Unidentifiability vs. sparsity density (continuous-time ensemble).

(1) Generate random (A,B) with the **sparse_continuous** ensemble (density p).
    If `ensembles.sparse_continuous` is unavailable, we fall back to a compatible
    sparse CT generator that enforces stability of A by spectral shift.
(2) For each (A,B) and x0~Unif(S^{n-1}), compute:
      - PBH structured margin: d_pbh = pbh_margin_structured(A,B,x0)
      - Left-eigen criterion:  mu_min = min_i ||w_i^T [x0 B]||_2 / ||w_i||_2
    Treat them as **binary** with tolerance `--tol`:
      unident_pbh = (d_pbh <= tol)
      unident_mu  = (mu_min <= tol)
      unident_any = unident_pbh OR unident_mu
(3) For densities p ∈ {0.05, 0.10, …, 0.95, 1.0} (20 levels), with N=200 per level,
    report the **percentage of unidentifiable pairs** vs. p and plot it.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from dataclasses import dataclass, field

from ..config import ExperimentConfig
from ..ensembles import sparse_continuous
from ..metrics import cont2discrete_zoh, pbh_margin_structured
from ..simulation import simulate_dt, prbs
from ..estimators import dmdc_tls

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SparseConfig(ExperimentConfig):
    """Configuration for sparse system experiments."""
    density_range: List[float] = field(
        default_factory=lambda: np.linspace(0.1, 0.9, 20).tolist()
    )
    trials_per_density: int = 50
    outdir: str = "out_sparse"
    
    def __post_init__(self):
        super().__post_init__()
        if not self.density_range:
            raise ValueError("density_range cannot be empty")
        if min(self.density_range) <= 0 or max(self.density_range) >= 1:
            raise ValueError("density must be in (0,1)")
        if self.trials_per_density < 1:
            raise ValueError("trials_per_density must be positive")

def run_one_trial(density: float, cfg: SparseConfig, rng: np.random.Generator) -> Dict[str, Any]:
    # Generate sparse system
    A = sparse_continuous(cfg.n, int(density), rng)
    B = rng.standard_normal((cfg.n, cfg.m))
    Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)  # Note: order is Ad, Bd not Bd, Ad
    Bd, Ad = cont2discrete_zoh(A, B, cfg.dt)
    
    # Generate input and simulate
    U = prbs(cfg.m, cfg.T, scale=cfg.u_scale, dwell=cfg.dwell, rng=rng)
    x0 = rng.standard_normal(cfg.n)
    x0 /= np.linalg.norm(x0)
    
    X = simulate_dt(cfg.T, x0, Ad, Bd, U, noise_std=cfg.noise_std, rng=rng)
    
    # Identify
    X0, X1 = X[:, :-1], X[:, 1:]
    Ahat, Bhat = dmdc_tls(X0, X1, U)
    
    return {
        "density": density,
        "errA": np.linalg.norm(Ahat - Ad, 'fro'),
        "errB": np.linalg.norm(Bhat - Bd, 'fro'),
        "margin": pbh_margin_structured(Ad, Bd, x0),
        "sparsity_A": np.sum(np.abs(np.asarray(A)) > 1e-10) / np.asarray(A).size,
        "sparsity_Ahat": np.sum(np.abs(Ahat) > 1e-10) / Ahat.size,
        "meta": {
            "cond_A": np.linalg.cond(A),
            "timestamp": pd.Timestamp.now().isoformat()
        }
    }

def run_experiment(cfg: SparseConfig) -> pd.DataFrame:
    """Run complete sparsity sweep experiment."""
    rng = np.random.default_rng(cfg.seed)
    results = []
    
    for density in cfg.density_range:
        logger.info(f"Running density = {density:.3f}")
        for trial in range(cfg.trials_per_density):
            result = run_one_trial(density, cfg, rng)
            results.append(result)
    
    df = pd.DataFrame(results)
    
    outdir = Path(cfg.outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    
    csv_path = outdir / "sparse_sweep.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")
    
    plot_path = outdir / "sparse_summary.png"
    plot_sparsity_summary(df, save_path=plot_path)
    logger.info(f"Saved plot to {plot_path}")
    
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--trials-per-density", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="out_sparse")
    args = ap.parse_args()
    
    cfg = SparseConfig(
        n=args.n,
        m=args.m,
        trials_per_density=args.trials_per_density,
        seed=args.seed,
        outdir=args.outdir
    )
    
    run_experiment(cfg)

if __name__ == "__main__":
    main()
