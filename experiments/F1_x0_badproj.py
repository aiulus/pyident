from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from ..config import ExperimentConfig
from ..ensembles import draw_with_ctrb_rank, controllability_rank
from ..metrics import cont2discrete_zoh, pbh_margin_structured, krylov_smin_norm
from ..simulation import simulate_dt, prbs
from ..estimators import dmdc_tls
from ..projectors import build_projected_x0, left_uncontrollable_subspace

EPS = 1e-18

# --- scoring helpers (CT metrics, log-protected) ---
def _log_pbh(A: np.ndarray, B: np.ndarray, x0: np.ndarray) -> float:
    m = float(pbh_margin_structured(A, B, x0))
    return float(np.log(max(m, EPS)))

def _log_krylov(A: np.ndarray, B: np.ndarray, x0: np.ndarray) -> float:
    smin = float(krylov_smin_norm(A, B, x0))
    return float(np.log(max(smin, EPS)))

# --------------------------
# Core experiment per d
# --------------------------

def run_experiment_for_d(cfg: ExperimentConfig, d: int, q_high: float, k_off: int) -> Dict[str, Any]:
    rng = np.random.default_rng(int(cfg.seed) + int(d) * 997)

    # CT base pair with rank r = n - d
    A, B, _ = draw_with_ctrb_rank(
        n=cfg.n, m=cfg.m, r=max(0, cfg.n - d), rng=rng,
        base_c="stable", base_u="stable"
    )
    Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)
    # --- debug: controllability rank & dark dimension ---
    W_all = left_uncontrollable_subspace(A, B)
    rCT, _ = controllability_rank(A, B, order=A.shape[0], rtol=1e-8)
    print(f"[debug] d={d}: CT controllability rank r={rCT} (expect {A.shape[0]-d}), dark_dim={W_all.shape[1]}")


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
    x0_rand_store: List[np.ndarray] = []
    for _ in range(cfg.n_trials):
        x0 = rng.standard_normal(cfg.n); x0 /= np.linalg.norm(x0) + EPS
        lp = _log_pbh(A, B, x0); lk = _log_krylov(A, B, x0)
        x0_rand_store.append(x0)
        logpbh_rand.append(lp); logkry_rand.append(lk)
        mses_rand.append(trial(x0))

    # Select high-scoring seeds (away from 0) by upper quantiles
    tau_lp = float(np.quantile(logpbh_rand, q_high))
    tau_lk = float(np.quantile(logkry_rand, q_high))

    seeds: List[np.ndarray] = []
    for x0, lp, lk in zip(x0_rand_store, logpbh_rand, logkry_rand):
        if (lp >= tau_lp) and (lk >= tau_lk):
            seeds.append(x0)
    if len(seeds) == 0:
        raise RuntimeError(f"[d={d}] No high-scoring seeds found; try smaller q_high")

    # Project each selected seed to kill at least k_off uncontrollable directions
    mses_proj: List[float] = []
    logpbh_proj: List[float] = []
    logkry_proj: List[float] = []
    used = 0
    W_all = left_uncontrollable_subspace(A, B)
    dark_dim = int(W_all.shape[1])

    for x0_seed in seeds:
        x0_proj, W_off, P_dark = build_projected_x0(A, B, x0_seed, k_off=k_off, rng=rng)
        # verify darkness for at least one direction
        assert W_off.shape[1] == min(k_off, dark_dim)
        if W_off.shape[1] > 0:
            if np.linalg.norm(W_off.T @ x0_proj) > 1e-8:
                continue  # numerical issue; skip
        logpbh_proj.append(_log_pbh(A, B, x0_proj))
        logkry_proj.append(_log_krylov(A, B, x0_proj))
        mses_proj.append(trial(x0_proj))
        used += 1
        if used >= cfg.n_trials:
            break

    if used < cfg.n_trials:
        # top up by re-seeding randomly (rare)
        while used < cfg.n_trials:
            x0_seed = rng.standard_normal(cfg.n); x0_seed /= np.linalg.norm(x0_seed) + EPS
            x0_proj, W_off, _ = build_projected_x0(A, B, x0_seed, k_off=k_off, rng=rng)
            if W_off.shape[1] > 0 and np.linalg.norm(W_off.T @ x0_proj) > 1e-8:
                continue
            logpbh_proj.append(_log_pbh(A, B, x0_proj))
            logkry_proj.append(_log_krylov(A, B, x0_proj))
            mses_proj.append(trial(x0_proj))
            used += 1

    return {
        "d": d,
        "dark_dim": dark_dim,
        "q_high": float(q_high),
        "k_off": int(k_off),
        "mse_random": np.asarray(mses_rand, float),
        "mse_projected": np.asarray(mses_proj, float),
        "logpbh_random": np.asarray(logpbh_rand, float),
        "logkry_random": np.asarray(logkry_rand, float),
        "logpbh_projected": np.asarray(logpbh_proj, float),
        "logkry_projected": np.asarray(logkry_proj, float),
        "tau_logpbh_high": float(tau_lp),
        "tau_logkry_high": float(tau_lk),
    }

# --------------------------
# Plot
# --------------------------

def _global_limits(results: List[Dict[str, Any]]) -> Tuple[float, float]:
    all_mse = np.concatenate([np.concatenate([o["mse_random"], o["mse_projected"]]) for o in results])
    lo, hi = float(np.min(all_mse)), float(np.max(all_mse))
    pad = 0.02 * (hi - lo + 1e-12)
    return lo - pad, hi + pad


def plot_box_grid(results: List[Dict[str, Any]], out_png: Path) -> None:
    ylo, yhi = _global_limits(results)
    nrows = len(results)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(6, 2.6*nrows), sharey=True)
    if nrows == 1:
        axes = [axes]
    for ax, out in zip(axes, sorted(results, key=lambda o: o["d"])):
        data = [out["mse_random"], out["mse_projected"]]
        ax.boxplot(data, whis=(5, 95))
        ax.set_xticks([1, 2], labels=["random $x_0$", "projected–dark $x_0$"])
        ax.set_ylabel("MSE")
        ax.set_title(f"Boxplot — deficiency $d={out['d']}$ (dark_dim={out['dark_dim']})")
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.set_ylim(ylo, yhi)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

# --------------------------
# CLI
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="out_x0_darkproj", help="Directory to save figure")
    ap.add_argument("--dmax", type=int, default=3, help="Max deficiency d to include (starts at 0)")
    ap.add_argument("--q-high", type=float, default=0.80, help="Upper-quantile for seed selection")
    ap.add_argument("--k-off", type=int, default=1, help="# of uncontrollable directions to darken")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True, parents=True)

    cfg = ExperimentConfig()
    d_vals = [d for d in range(0, min(args.dmax, cfg.n-1) + 1)]

    print(f"Running dark-projection sweep over d = {d_vals} …")
    results = []
    for d in d_vals:
        out = run_experiment_for_d(cfg, d, q_high=args.q_high, k_off=args.k_off)
        results.append(out)
        print(
            " d={d}: dark_dim={dd}, τ_high(logPBH)={tLP:.3e}, τ_high(logσmin(K))={tLK:.3e}".format(
                d=d, dd=out["dark_dim"], tLP=out["tau_logpbh_high"], tLK=out["tau_logkry_high"],
            )
        )

    print(f"Saving boxplot grid to {outdir} …")
    plot_box_grid(results, outdir / "f1_darkproj_box_grid.png")
    print("Done.")

if __name__ == "__main__":
    main()
