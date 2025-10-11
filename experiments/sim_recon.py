from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import argparse

import matplotlib.pyplot as plt
import numpy as np

from ..config import ExperimentConfig
from ..estimators import dmdc_pinv, dmdc_tls, moesp_fit
from ..metrics import eta0, unified_generator
from ..signals import estimate_pe_order
from ..simulation import prbs, simulate_dt
from .visible_sampling import (
    VisibleDrawConfig,
    construct_x0_with_dim_visible,
    prepare_system_with_visible_dim,
)


Estimator = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def _available_estimators() -> Dict[str, Estimator]:
    return {
        "dmdc_tls": dmdc_tls,
        "dmdc_pinv": dmdc_pinv,
        "moesp": moesp_fit,
    }


def _identifiability_summary(Ad: np.ndarray, Bd: np.ndarray, x0: np.ndarray) -> Dict[str, float]:
    generator = unified_generator(Ad, Bd, x0, mode="unrestricted")
    if generator.size:
        svals = np.linalg.svd(generator, compute_uv=False)
        sigma_min = float(svals[-1])
        cond = float(svals[0] / (svals[-1] + 1e-15))
    else:
        sigma_min = 0.0
        cond = np.inf

    eta = float(eta0(Ad, Bd, x0, rtol=1e-12))

    return {
        "sigma_min": sigma_min,
        "sigma_cond": cond,
        "eta0": eta,
    }


@dataclass
class TrajectoryReconConfig(ExperimentConfig):
    estimator: str = "dmdc_tls"
    outdir: Path = field(default_factory=lambda: Path("out_trajectory_recon"))
    force_hurwitz: bool = True
    stability_margin: float = 0.05


def _select_estimator(name: str) -> Estimator:
    estimators = _available_estimators()
    if name not in estimators:
        raise ValueError(f"Unknown estimator '{name}'. Choose from {list(estimators)}.")
    return estimators[name]


def _visible_dims(n: int) -> List[int]:
    max_drop = min(5, n - 1)
    return [n - d for d in range(1, max_drop + 1)]


def run_experiment(cfg: TrajectoryReconConfig) -> Dict[str, object]:
    rng = np.random.default_rng(int(cfg.seed))
    estimator = _select_estimator(cfg.estimator)

    # Shared PE input signal for all visibility levels.
    U = prbs(cfg.T, cfg.m, scale=cfg.u_scale, dwell=cfg.dwell, rng=rng)
    pe_order = estimate_pe_order(U, s_max=min(cfg.T // 2, cfg.n + cfg.m))

    dims = _visible_dims(cfg.n)

    time = np.arange(cfg.T + 1) * cfg.dt

    results: Dict[str, object] = {
        "cfg": cfg,
        "dims": dims,
        "pe_order": pe_order,
        "trials": [],
    }

    base_ensemble = "stable" if cfg.ensemble == "A_stbl_B_ctrb" else cfg.ensemble

    for dim_visible in dims:
        draw_cfg = VisibleDrawConfig(
            n=cfg.n,
            m=cfg.m,
            dt=cfg.dt,
            dim_visible=dim_visible,
            ensemble=base_ensemble,
            tol=1e-12,
            force_hurwitz=cfg.force_hurwitz,
            stability_margin=cfg.stability_margin,
        )

        _, _, Ad, Bd, Rbasis = prepare_system_with_visible_dim(draw_cfg, rng)

        x0, Vbasis = construct_x0_with_dim_visible(
            Ad,
            Bd,
            Rbasis,
            dim_visible,
            rng,
            tol=1e-12,
        )

        X_true = simulate_dt(x0, Ad, Bd, U, noise_std=0.0)
        if not np.all(np.isfinite(X_true)):
            raise RuntimeError("True trajectory produced non-finite values. Try enabling force_hurwitz or increasing the stability margin.")
        true_norm = float(np.linalg.norm(X_true, ord="fro") + 1e-15)
        true_ident = _identifiability_summary(Ad, Bd, x0)

        recon_trajectories: List[np.ndarray] = []
        traj_errors: List[float] = []
        sigma_vals: List[float] = []
        eta_vals: List[float] = []

        for trial in range(cfg.n_trials):
            trial_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
            X_noisy = simulate_dt(x0, Ad, Bd, U, noise_std=cfg.noise_std, rng=trial_rng)
            X0, X1 = X_noisy[:, :-1], X_noisy[:, 1:]

            try:
                Ahat, Bhat = estimator(X0, X1, U)
            except Exception as exc:  # pragma: no cover - safeguards rare numerical issues
                # Skip rare failures without aborting the sweep.
                print(f"[warn] estimator '{cfg.estimator}' failed on trial {trial}: {exc}")
                continue

            X_recon = simulate_dt(x0, Ahat, Bhat, U, noise_std=0.0)
            if not np.all(np.isfinite(X_recon)):
                print(f"[warn] skipping unstable reconstruction on trial {trial}: non-finite trajectory")
                continue
            recon_trajectories.append(X_recon)

            err = float(np.linalg.norm(X_recon - X_true, ord="fro") / true_norm)
            if not np.isfinite(err):
                print(f"[warn] skipping unstable reconstruction on trial {trial}: error is not finite")
                recon_trajectories.pop()
                continue
            traj_errors.append(err)

            ident_hat = _identifiability_summary(Ahat, Bhat, x0)
            sigma_vals.append(ident_hat["sigma_min"])
            eta_vals.append(ident_hat["eta0"])

        if not traj_errors:
            raise RuntimeError(
                f"No successful trials recorded for dim V(x0)={dim_visible}. "
                "Check estimator stability or noise level."
            )

        results["trials"].append(
            {
                "dim_visible": dim_visible,
                "deficiency": cfg.n - dim_visible,
                "x0": x0,
                "Vbasis": Vbasis,
                "X_true": X_true,
                "recon": recon_trajectories,
                "errors": np.asarray(traj_errors, float),
                "sigma_min": np.asarray(sigma_vals, float),
                "eta0": np.asarray(eta_vals, float),
                "true_ident": true_ident,
                "time": time,
            }
        )

    results["trials"] = sorted(results["trials"], key=lambda item: item["dim_visible"], reverse=True)
    return results


def _plot_trajectories(trials: Sequence[Dict[str, object]], outdir: Path) -> None:
    rows = len(trials)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.0 * rows), sharex=True)
    if rows == 1:
        axes = np.asarray([axes])
    colors = plt.cm.tab10(np.linspace(0, 1, trials[0]["X_true"].shape[0]))

    for idx, trial in enumerate(trials):
        ax_true = axes[idx, 0]
        ax_recon = axes[idx, 1]

        time = trial["time"]
        X_true = trial["X_true"]
        recon_list = trial["recon"]
        dim_visible = trial["dim_visible"]

        for state_idx, color in enumerate(colors):
            ax_true.plot(time, X_true[state_idx, :], color=color, label=f"$x_{state_idx+1}$")

        ax_true.set_title(f"True trajectory — dim V(x0) = {dim_visible}")
        ax_true.grid(True, linestyle="--", alpha=0.5)
        if idx == rows - 1:
            ax_true.set_xlabel("Time")
        ax_true.set_ylabel("State value")

        for X_hat in recon_list:
            for state_idx, color in enumerate(colors):
                ax_recon.plot(time, X_hat[state_idx, :], color=color, alpha=0.08)

        mean_recon = np.mean(np.stack(recon_list, axis=0), axis=0)
        for state_idx, color in enumerate(colors):
            ax_recon.plot(time, mean_recon[state_idx, :], color=color, linewidth=1.5)
            ax_recon.plot(time, X_true[state_idx, :], color=color, linestyle="--", linewidth=1.0)

        ax_recon.set_title("Reconstructed trajectories")
        ax_recon.grid(True, linestyle="--", alpha=0.5)
        if idx == rows - 1:
            ax_recon.set_xlabel("Time")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outdir / "trajectories_overlay.png", dpi=150)
    plt.close(fig)


def _plot_error_vs_deficiency(trials: Sequence[Dict[str, object]], outdir: Path) -> None:
    deficiencies = [trial["deficiency"] for trial in trials]
    means = [float(np.mean(trial["errors"])) for trial in trials]
    stds = [float(np.std(trial["errors"])) for trial in trials]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(deficiencies, means, yerr=stds, fmt="o-", capsize=4)
    ax.set_xlabel(r"$n - \dim V(x_0)$")
    ax.set_ylabel("Mean relative trajectory error")
    ax.set_title("Trajectory reconstruction error vs. visible deficiency")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(outdir / "error_vs_deficiency.png", dpi=150)
    plt.close(fig)


def _plot_error_vs_identifiability(trials: Sequence[Dict[str, object]], outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for trial in trials:
        deficiency = trial["deficiency"]
        mean_error = float(np.mean(trial["errors"]))
        sigma = trial["sigma_min"]
        eta_vals = trial["eta0"]

        axes[0].scatter(np.mean(sigma), mean_error, label=f"d={deficiency}")
        axes[0].set_xlabel(r"Mean $\sigma_{\min}$ of generator")

        axes[1].scatter(np.mean(eta_vals), mean_error, label=f"d={deficiency}")
        axes[1].set_xlabel(r"Mean $\eta_0$")

    for ax in axes:
        ax.set_ylabel("Mean relative trajectory error")
        ax.grid(True, linestyle="--", alpha=0.6)
    axes[0].set_title(r"Error vs. generator $\sigma_{\min}$")
    axes[1].set_title(r"Error vs. $\eta_0$")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(outdir / "error_vs_identifiability.png", dpi=150)
    plt.close(fig)


def save_figures(results: Dict[str, object]) -> None:
    outdir = Path(results["cfg"].outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    trials: Sequence[Dict[str, object]] = results["trials"]

    _plot_trajectories(trials, outdir)
    _plot_error_vs_deficiency(trials, outdir)
    _plot_error_vs_identifiability(trials, outdir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Trajectory reconstruction using estimated models")
    ap.add_argument("--outdir", type=str, default="out_trajectory_recon", help="Output directory for figures")
    ap.add_argument("--n", type=int, default=8, help="State dimension")
    ap.add_argument("--m", type=int, default=2, help="Input dimension")
    ap.add_argument("--T", type=int, default=200, help="Trajectory horizon")
    ap.add_argument("--dt", type=float, default=0.1, help="Sampling time")
    ap.add_argument("--noise-std", type=float, default=0.02, help="Process noise std. dev. during data generation")
    ap.add_argument("--n-trials", type=int, default=200, help="Number of estimation trials")
    ap.add_argument("--u-scale", type=float, default=3.0, help="PRBS input scaling")
    ap.add_argument("--dwell", type=int, default=1, help="PRBS dwell time")
    ap.add_argument("--seed", type=int, default=12345, help="Random seed")
    ap.add_argument("--estimator", type=str, default="dmdc_tls", choices=list(_available_estimators().keys()))
    ap.add_argument("--ensemble", type=str, default="ginibre", help="Ensemble used to draw (A,B)")
    ap.add_argument("--no-force-hurwitz", dest="force_hurwitz", action="store_false",
                    help="Disable shifting random A draws to ensure stability")
    ap.add_argument("--stability-margin", type=float, default=0.05,
                    help="Real-part margin enforced when stabilising random A")

    ap.set_defaults(force_hurwitz=True)

    args = ap.parse_args()

    cfg = TrajectoryReconConfig(
        n=args.n,
        m=args.m,
        T=args.T,
        dt=args.dt,
        noise_std=args.noise_std,
        n_trials=args.n_trials,
        u_scale=args.u_scale,
        dwell=args.dwell,
        seed=args.seed,
        estimator=args.estimator,
        ensemble=args.ensemble,
        outdir=Path(args.outdir),
        force_hurwitz=args.force_hurwitz,
        stability_margin=args.stability_margin,
    )

    results = run_experiment(cfg)
    save_figures(results)

    dims = ", ".join(str(item["dim_visible"]) for item in results["trials"])
    print(f"Completed trajectory reconstruction sweep. dim V(x0): {dims}")
    print(f"Estimated PE order of shared input: {results['pe_order']}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()