# sim_eqv_cl.py  (drop-in replacement of your current file)

from __future__ import annotations
from typing import Dict, Any

import argparse, pathlib, math, json
import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..ensembles import draw_with_ctrb_rank
from ..metrics import (
    cont2discrete_zoh,
    pbh_margin_structured,
    same_equiv_class_dt_rel,   # relative version of EC check
    regressor_stats,           # PE / regressor health proxy
)
from ..simulation import simulate_dt, prbs
from ..estimators import (
    dmdc_pinv,
    dmdc_tls,
    moesp_fit,
    sindy_fit,
    node_fit,
)

# ---------------------------------------------------------------------

def unit(v: np.ndarray, eps: float = 0.0) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n if n > eps else 1.0)

# --------------------------- main run --------------------------------

def run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Run single equivalence class experiment."""
    rng = np.random.default_rng(cfg.seed)
    
    # Generate system
    A, B, meta = draw_with_ctrb_rank(cfg.n, cfg.m, cfg.n-2, rng)
    Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)
    
    # Generate input
    U = prbs(cfg.T, cfg.m, scale=cfg.u_scale, dwell=cfg.dwell, rng=rng)
    
    # Simulate and identify
    x0 = rng.standard_normal(cfg.n)
    x0 /= np.linalg.norm(x0)
    
    X = simulate_dt(x0, Ad, Bd, U, noise_std=cfg.noise_std, rng=rng)
    
    # Identify
    X0, X1, Utr = X[:, :-1], X[:, 1:], U.T
    zstats = regressor_stats(X0, Utr, rtol_rank=1e-12)

    Ahat, Bhat = dmdc_tls(X0, X1, Utr)

    ok, info = same_equiv_class_dt_rel(Ad, Bd, Ahat, Bhat, x0,
                                        rtol_eq=cfg.rtol_eq_rel, rtol_rank=1e-12, use_leak=True)

    rows = [dict(
        trial=0, algo="dmdc_tls", ok=int(ok),
        dim_V=info.get("dim_V", None),
        dA_V=info.get("dA_V_rel", None),
        leak=info.get("leak_rel", None),
        dB=info.get("dB_rel", None),
        noise_std=cfg.noise_std, T=cfg.T, dwell=cfg.dwell, u_scale=cfg.u_scale,
        **zstats
    )]

    df = pd.DataFrame(rows)
    outdir = pathlib.Path("out_eqv_cl")
    df.to_csv(outdir / "consistency.csv", index=False)

    # small summary
    s = df.groupby("algo")["ok"].mean().reset_index().rename(columns={"ok": "ok_rate"})
    s["noise_std"] = cfg.noise_std; s["T"] = cfg.T; s["dwell"] = cfg.dwell; s["u_scale"] = cfg.u_scale
    s.to_csv(outdir / "consistency_summary.csv", index=False)
    print("[DT] OK rates per algorithm:\n", s)

    return {
        "A": A, "B": B,
        "Ad": Ad, "Bd": Bd,
        "Ahat": Ahat, "Bhat": Bhat,
        "X": X, "U": U,
        "meta": meta
    }

# ----------------------------- CLI -----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    cfg = ExperimentConfig(n=args.n, m=args.m)
    out = run_experiment(cfg)
    
    # Save results
    np.savez("eqvcl_results.npz", **out)

if __name__ == "__main__":
    main()