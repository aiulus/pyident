"""Simplified identifiability/error correlation experiment (stochastic engine).

This script mirrors the stochastic mode of ``sim_cor`` but keeps the
core metrics from the historical ``sim1`` script.  Each sampled
system/state pair is evaluated with a configurable list of estimators
and both the discrete-time identifiability proxies (``σ_min``,
``η_0``, ``dim V``) and the continuous-time metrics (PBH margin,
Krylov σ_min, μ_min) are recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..ensembles import sample_system_instance
from ..estimators import dmdc_pinv, dmdc_tls, moesp_fit, node_fit, sindy_fit
from ..metrics import cont2discrete_zoh, pair_distance
from ..simulation import simulate_dt
from .simcor_common import (
    CoreExperimentConfig,
    available_ident_scores,
    compute_core_metrics,
    compute_identifiability_metrics,
    ident_score_label,
    prbs_with_order,
    relative_errors,
    sample_unit_sphere,
    target_pe_order,
)


# ---------------------------------------------------------------------------
# Configuration & CLI
# ---------------------------------------------------------------------------


@dataclass
class SimpleCoreConfig(CoreExperimentConfig):
    """Configuration for the simplified stochastic experiment."""

    estimators: tuple[str, ...] = ("dmdc_pinv", "moesp")
    outdir: Path = Path("out_simpcore")
    crit: str = "sigma_min"

    def __post_init__(self) -> None:  # type: ignore[override]
        super().__post_init__()
        if not self.estimators:
            raise ValueError("At least one estimator must be specified.")
        if isinstance(self.outdir, str):
            self.outdir = Path(self.outdir)


def _available_estimators() -> Dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, float], tuple[np.ndarray, np.ndarray]]]:
    return {
        "dmdc_tls": lambda X0, X1, U_cm, dt: dmdc_tls(X0, X1, U_cm),
        "dmdc_pinv": lambda X0, X1, U_cm, dt: dmdc_pinv(X0, X1, U_cm),
        "moesp": lambda X0, X1, U_cm, dt: moesp_fit(X0, X1, U_cm),
        "sindy": lambda X0, X1, U_cm, dt: sindy_fit(X0, X1, U_cm, dt),
        "node": lambda X0, X1, U_cm, dt: node_fit(X0, X1, U_cm, dt, epochs=200),
    }


def _parse_args(argv: Sequence[str] | None = None) -> SimpleCoreConfig:
    estimators = _available_estimators()

    parser = argparse.ArgumentParser(
        description="Correlate identifiability metrics with estimation error (stochastic engine)."
    )
    parser.add_argument("--n", type=int, default=6, help="State dimension.")
    parser.add_argument("--m", type=int, default=3, help="Input dimension.")
    parser.add_argument("--T", type=int, default=400, help="Trajectory horizon (minimum enforced automatically).")
    parser.add_argument("--dt", type=float, default=0.05, help="Sampling time for discretisation.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    parser.add_argument(
        "--ensemble",
        type=str,
        default="ginibre",
        choices=["ginibre", "sparse", "stable", "binary", "A_stbl_B_ctrb"],
        help="Base ensemble for (A, B).",
    )
    parser.add_argument("--p-density", type=float, default=0.8, dest="p_density", help="Density for sparse ensemble.")
    parser.add_argument(
        "--sparse-which",
        "--sparse_which",
        dest="sparse_which",
        type=str,
        default="both",
        choices=["A", "B", "both"],
        help="Which matrices are sparsified when ensemble=sparse.",
    )
    parser.add_argument(
        "--p-density-B",
        "--p_density_B",
        dest="p_density_B",
        type=float,
        default=None,
        help="Optional density for B when ensemble=sparse.",
    )

    parser.add_argument("--ens-size", type=int, default=24, dest="ens_size", help="Number of systems in the ensemble.")
    parser.add_argument("--x0-samples", type=int, default=8, dest="x0_samples", help="Initial states per system.")
    parser.add_argument(
        "--x0amp",
        type=float,
        default=1.0,
        help="Uniform scaling factor applied to the unit-sphere initial states.",
    )
    parser.add_argument("--noise-std", type=float, default=0.0, dest="noise_std", help="Process noise standard deviation.")
    parser.add_argument("--u-scale", type=float, default=3.0, dest="u_scale", help="PRBS amplitude.")
    parser.add_argument("--dwell", type=int, default=1, help="PRBS dwell time (samples per draw).")

    parser.add_argument(
        "--estimators",
        type=str,
        nargs="+",
        default=["dmdc_pinv", "moesp"],
        choices=sorted(estimators.keys()),
        help="Estimators evaluated for each trial.",
    )

    parser.add_argument(
        "--crit",
        type=str,
        default="sigma_min",
        choices=sorted(available_ident_scores()),
        help="Identifiability score to display on the x-axis.",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="out_simpcore",
        help="Directory for CSV outputs and plots.",
    )

    args = parser.parse_args(argv)

    cfg = SimpleCoreConfig(
        n=args.n,
        m=args.m,
        T=args.T,
        dt=args.dt,
        seed=args.seed,
        ensemble=args.ensemble,
        p_density=args.p_density,
        sparse_which=args.sparse_which,
        p_density_B=args.p_density_B,
        ens_size=args.ens_size,
        x0_samples=args.x0_samples,
        x0_amp=args.x0amp,
        noise_std=args.noise_std,
        u_scale=args.u_scale,
        dwell=args.dwell,
        estimators=tuple(args.estimators),
        outdir=Path(args.outdir),
        crit=args.crit,
    )
    return cfg


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------


def _select_estimators(
    names: Iterable[str],
) -> Dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, float], tuple[np.ndarray, np.ndarray]]]:
    available = _available_estimators()
    selected: Dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, float], tuple[np.ndarray, np.ndarray]]] = {}
    for name in names:
        if name not in available:
            raise ValueError(f"Unknown estimator '{name}'. Available: {sorted(available)}")
        selected[name] = available[name]
    return selected


def _scatter_plot(df: pd.DataFrame, cfg: SimpleCoreConfig, outfile: Path) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    estimators = sorted(df["estimator"].unique())
    n_plots = len(estimators)
    ncols = min(3, max(1, n_plots))
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for ax in axes[n_plots:]:
        ax.axis("off")

    x_label = ident_score_label(cfg.crit)

    for ax, name in zip(axes, estimators):
        sub = df[df["estimator"] == name]
        colors = sub["deficiency"].to_numpy()
        sc = ax.scatter(
            sub["crit_value"],
            sub["err_mean_rel"],
            c=colors,
            cmap="viridis",
            alpha=0.7,
            s=20,
        )
        ax.set_title(f"{name}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("mean relative error")
        ax.grid(True, linestyle="--", alpha=0.4)
    if n_plots:
        fig.colorbar(
            sc,
            ax=axes[:n_plots],
            shrink=0.85,
            label="deficiency",
            pad=0.02,
        )

    fig.suptitle(
        f"Identifiability vs error (stochastic, ensemble={cfg.ensemble})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.03, 0.95, 0.95])
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def _compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (estimator, deficiency), sub in df.groupby(["estimator", "deficiency"]):
        if len(sub) < 2:
            pearson = np.nan
            spearman = np.nan
        else:
            x = sub["crit_value"].to_numpy()
            y = sub["err_mean_rel"].to_numpy()
            if np.allclose(x.std(ddof=0), 0.0) or np.allclose(y.std(ddof=0), 0.0):
                pearson = np.nan
            else:
                pearson = float(np.corrcoef(x, y)[0, 1])

            rx = pd.Series(x).rank(method="average").to_numpy()
            ry = pd.Series(y).rank(method="average").to_numpy()
            if np.allclose(rx.std(ddof=0), 0.0) or np.allclose(ry.std(ddof=0), 0.0):
                spearman = np.nan
            else:
                spearman = float(np.corrcoef(rx, ry)[0, 1])

        rows.append(
            {
                "estimator": estimator,
                "deficiency": int(deficiency),
                "pearson": pearson,
                "spearman": spearman,
                "count": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def _run_stochastic(cfg: SimpleCoreConfig, rng: np.random.Generator) -> pd.DataFrame:
    estimators = _select_estimators(cfg.estimators)

    U, order_est = prbs_with_order(cfg, rng)
    if order_est < target_pe_order(cfg):
        print(
            f"[warn] PRBS achieved order {order_est} < target {target_pe_order(cfg)}. Results may be conservative."
        )

    records: List[Dict[str, object]] = []
    U_cm = U.T

    for sys_idx in range(cfg.ens_size):
        A, B = sample_system_instance(cfg, rng)
        Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)
        for x_idx in range(cfg.x0_samples):
            x0 = sample_unit_sphere(cfg.n, rng, radius=cfg.x0_amp)
            ident = compute_identifiability_metrics(Ad, Bd, x0)
            core = compute_core_metrics(A, B, x0)

            X = simulate_dt(x0, Ad, Bd, U, noise_std=cfg.noise_std, rng=rng)
            X0, X1 = X[:, :-1], X[:, 1:]

            for name, estimator in estimators.items():
                try:
                    Ahat, Bhat = estimator(X0, X1, U_cm, cfg.dt)
                except Exception:
                    continue

                rel = relative_errors(Ahat, Bhat, Ad, Bd)
                err_pair = float(pair_distance(Ahat, Bhat, Ad, Bd))
                dim_visible = int(round(ident["dim_visible"]))

                records.append(
                    {
                        "engine": "stoch",
                        "estimator": name,
                        "system_index": sys_idx,
                        "x0_index": x_idx,
                        "deficiency": int(cfg.n - dim_visible),
                        "dim_visible": dim_visible,
                        "sigma_min": ident["sigma_min"],
                        "eta0": ident["eta0"],
                        "pbh_struct": core["pbh_struct"],
                        "krylov_smin": core["krylov_smin"],
                        "mu_min": core["mu_min"],
                        "errA_rel": rel["errA_rel"],
                        "errB_rel": rel["errB_rel"],
                        "err_mean_rel": rel["err_mean_rel"],
                        "err_pair": err_pair,
                        "pe_order": order_est,
                        "crit_value": ident[cfg.crit],
                    }
                )

    if not records:
        raise RuntimeError("No successful trials were recorded. Check estimator stability or configuration.")

    return pd.DataFrame.from_records(records)


def main(argv: Sequence[str] | None = None) -> None:
    cfg = _parse_args(argv)
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(cfg.seed))
    df = _run_stochastic(cfg, rng)

    scatter_path = cfg.outdir / "scatter_stoch.png"
    _scatter_plot(df, cfg, scatter_path)

    corr_df = _compute_correlations(df)

    csv_path = cfg.outdir / "results_stoch.csv"
    df.to_csv(csv_path, index=False)

    corr_path = cfg.outdir / "correlations_stoch.csv"
    corr_df.to_csv(corr_path, index=False)

    meta = {
        "config": {
            "n": cfg.n,
            "m": cfg.m,
            "dt": cfg.dt,
            "T": cfg.T,
            "ensemble": cfg.ensemble,
            "estimators": list(cfg.estimators),
            "ens_size": cfg.ens_size,
            "x0_samples": cfg.x0_samples,
            "x0_amp": cfg.x0_amp,
            "noise_std": cfg.noise_std,
            "u_scale": cfg.u_scale,
            "dwell": cfg.dwell,
            "seed": cfg.seed,
            "crit": cfg.crit,
        },
        "records": int(len(df)),
    }
    with open(cfg.outdir / "summary_stoch.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Saved scatter plot to {scatter_path}")
    print(f"Saved results to {csv_path}")
    print(f"Saved correlations to {corr_path}")


if __name__ == "__main__":
    main()