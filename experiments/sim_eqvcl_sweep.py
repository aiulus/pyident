from __future__ import annotations
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import argparse
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from ..config import ExperimentConfig
from ..plots import plot_consistency_sweep

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SweepConfig(ExperimentConfig):
    """Configuration for equivalence class sweep experiment."""
    noise_range: List[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.1, 0.5]
    )
    dwell_range: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8]
    )
    T_range: List[int] = field(
        default_factory=lambda: [100, 200, 400, 800]
    )
    outdir: str = "out_consistency_sweep"
    
    def __post_init__(self):
        super().__post_init__()
        if not self.noise_range:
            raise ValueError("noise_range cannot be empty")
        if not self.dwell_range:
            raise ValueError("dwell_range cannot be empty")
        if not self.T_range:
            raise ValueError("T_range cannot be empty")

def run_one(cfg: SweepConfig, noise: float, dwell: int, T: int) -> Optional[Dict[str, Any]]:
    """Run single experiment with given parameters."""
    try:
        cmd = [
            "python", "-m", "pyident.experiments.sim_eqv_cl",
            "--n", str(cfg.n),
            "--m", str(cfg.m),
            "--seed", str(cfg.seed),
            "--noise", str(noise),
            "--dwell", str(dwell),
            "--T", str(T)
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        df = pd.read_csv("out_eqv_cl/consistency_summary.csv")
        return {
            "df": df, 
            "params": {
                "noise": noise, 
                "dwell": dwell, 
                "T": T
            }
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run sim_eqv_cl: {e}")
        return None
    except FileNotFoundError:
        logger.error("Could not find output CSV file")
        return None

def run_experiment(cfg: SweepConfig) -> pd.DataFrame:
    """Run full parameter sweep experiment."""
    results = []
    
    for noise in cfg.noise_range:
        for dwell in cfg.dwell_range:
            for T in cfg.T_range:
                out = run_one(cfg, noise, dwell, T)
                if out is not None:
                    results.append(out["df"])
    
    if not results:
        raise RuntimeError("No successful trials")
    
    df = pd.concat(results, ignore_index=True)
    
    outdir = Path(cfg.outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    
    csv_path = outdir / "sweep_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")
    
    plot_path = outdir / "sweep_summary.png"
    plot_consistency_sweep(str(csv_path), save_path=str(plot_path))
    logger.info(f"Saved plot to {plot_path}")
    
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="out_consistency_sweep")
    args = ap.parse_args()
    
    cfg = SweepConfig(
        n=args.n,
        m=args.m,
        seed=args.seed,
        outdir=args.outdir
    )
    
    run_experiment(cfg)

if __name__ == "__main__":
    main()
