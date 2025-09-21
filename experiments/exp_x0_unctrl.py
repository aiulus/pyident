# exp_x0_unctrl.py
import json, numpy as np
from ..config import ExpConfig, SolverOpts
from ..run_single import run_single

def sample_unctrl_AB(rng, n=8, m=2, dt=0.05):
    # Keep drawing until clearly uncontrollable (rank < n)
    while True:
        cfg = ExpConfig(n=n, m=m, T=1, dt=dt, ensemble="sparse",
                        sparse_which="B", p_density=0.8, p_density_B=0.05,
                        signal="prbs", sigPE=12, light=True)
        # single run just to get (A,B) via internal factory
        # Use a tiny T to be cheap; we won't use outputs from this call
        out = run_single(cfg, seed=int(rng.integers(1e9)), sopts=SolverOpts(), algs=("dmdc",), use_jax=False)
        # Extract A,B through K-based rank: K_rank < n is a good proxy for lack of reachability
        if out["K_rank"] < n:
            return out  # keep A,B implicitly fixed by the seed used internally

def run_grid(n_x0=200, seed=0, out_path="out/x0_unctrl.json"):
    rng = np.random.default_rng(seed)
    # Fix a problematic (A,B) instance
    base = sample_unctrl_AB(rng)
    # Reuse its exact config but vary x0 by varying the top-level seed
    rows = []
    for s in rng.integers(1e9, size=n_x0):
        cfg = ExpConfig(**{**base, "T": 600})  # ensure T=600 richness
        out = run_single(cfg, seed=int(s), sopts=SolverOpts(), algs=("dmdc","moesp"), use_jax=False)
        rows.append({
            "seed": int(s),
            "delta_pbh": out["delta_pbh"],
            "K_rank": out["K_rank"],
            "gram_min": out["gram_min"],
            "estimators": out["estimators"],
        })
    rows.sort(key=lambda r: r["delta_pbh"])
    lo = rows[:n_x0//10]
    hi = rows[-n_x0//10:]
    res = {
        "n": int(base["n"]), "m": int(base["m"]), "T": 600, "dt": base["dt"],
        "mode": "uncontrollable_fixed_AB_var_x0", "N": len(rows),
        "deciles": {"low": lo, "high": hi}
    }
    import os; os.makedirs("out", exist_ok=True)
    with open(out_path, "w") as f: json.dump(res, f, indent=2)

if __name__ == "__main__":
    run_grid()
