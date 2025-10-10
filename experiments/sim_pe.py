"""Experiments: PE order vs. visible subspace dimension.

This script implements the Chapter 5 experiment that tests the claim
"visible-subspace dimension is the ceiling, PE order is the floor".

We fix two discrete-time systems derived from stable continuous pairs:

* A partially identifiable pair with n=5 and dim V(x0)=3.
* A fully visible control system with n=5 and dim V(x0)=5.

For each pseudo-random binary sequence (PRBS) input, we vary the (block)
persistency of excitation order between 1 and 10 by adjusting the horizon
length so that the block Hankel matrix of depth r has full row rank while
higher depths do not.  We run DMDc (total-least-squares variant) to recover
the system matrices from a single trajectory and measure errors both in the
ambient space and restricted to the visible subspace.  The resulting CSV and
plots let us verify whether estimation errors drop once the PE order exceeds
the visible dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..ensembles import stable
from ..estimators import dmdc_tls
from ..metrics import cont2discrete_zoh, projected_errors, visible_subspace
from ..signals import estimate_moment_pe_order, estimate_pe_order
from ..simulation import prbs, simulate_dt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PEVisibleConfig(ExperimentConfig):
    """Configuration for the PE-order versus visible-subspace study."""

    n: int = 5
    m: int = 2
    T: int = 200
    dt: float = 0.1
    dwell: int = 1
    u_scale: float = 3.0
    noise_std: float = 0.0

    pe_orders: Sequence[int] = tuple(range(1, 11))
    trials_per_order: int = 25
    T_padding: int = 10
    T_min: int = 40
    max_system_attempts: int = 500
    visible_tol: float = 1e-8
    eps_norm: float = 1e-12
    outdir: str = "out_pe_vs_visible"

    def __post_init__(self) -> None:  # type: ignore[override]
        super().__post_init__()
        if self.trials_per_order < 1:
            raise ValueError("trials_per_order must be positive.")
        if not self.pe_orders:
            raise ValueError("pe_orders cannot be empty.")
        if min(self.pe_orders) <= 0:
            raise ValueError("pe_orders must be positive integers.")
        if self.T_min <= 0:
            raise ValueError("T_min must be positive.")
        if self.T_padding < 0:
            raise ValueError("T_padding must be non-negative.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vector(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm <= 0.0:
        raise ValueError("x0 cannot be the zero vector.")
    return x / norm


def _draw_system_with_visible_dim(
    cfg: PEVisibleConfig,
    target_dim: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw (A, B, Ad, Bd, x0, P) with dim span(P) == target_dim."""

    for _ in range(cfg.max_system_attempts):
        A, B = stable(cfg.n, cfg.m, rng)
        x0 = _unit_vector(rng.standard_normal(cfg.n))
        Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)
        P, dim = visible_subspace(Ad, Bd, x0, tol=cfg.visible_tol)
        if dim == target_dim:
            return A, B, Ad, Bd, x0, P
    raise RuntimeError(
        f"Unable to draw system with dim V(x0) == {target_dim} after {cfg.max_system_attempts} tries."
    )


def _horizon_for_order(order: int, cfg: PEVisibleConfig) -> int:
    """Choose a trajectory length whose Hankel depth matches ``order``."""

    # For block-Hankel H_r (m*r rows) to be valid we need at least 2r samples.
    # Add padding to avoid degenerate one-column Hankel matrices.
    return max(cfg.T_min, 2 * order + cfg.T_padding)


def _relative_norm(err: float, ref: float, eps: float) -> float:
    return float(err / (max(ref, eps)))


def _aggregate_stat(
    series: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (orders, median, q1, q3) for plotting."""

    grouped = series.groupby(level=0)
    orders = grouped.median().index.to_numpy()
    med = grouped.median().to_numpy()
    q1 = grouped.quantile(0.25).to_numpy()
    q3 = grouped.quantile(0.75).to_numpy()
    return orders, med, q1, q3


def _plot_summary(df: pd.DataFrame, cfg: PEVisibleConfig, outfile: pathlib.Path) -> None:
    """Plot relative errors vs PE order for each scenario."""

    outfile.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    metrics = [("errA_rel", "‖Â−A‖₍F₎ / ‖A‖₍F₎"), ("errA_V_rel", "Visible-projected error")]

    for ax, (col, label) in zip(axes, metrics):
        for scenario, linestyle, color in [("partial", "-", "C0"), ("full", "--", "C1")]:
            sub = df[df["scenario"] == scenario]
            grouped = sub.set_index("pe_order_actual")[col]
            orders, med, q1, q3 = _aggregate_stat(grouped)
            ax.plot(orders, med, linestyle=linestyle, color=color, label=scenario)
            ax.fill_between(orders, q1, q3, color=color, alpha=0.2)

        ax.set_xlabel("Estimated PE order")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.axvline(3, color="k", linestyle=":", linewidth=1.0, label="dim V(x0)=3" if ax is axes[0] else None)
        ax.axvline(5, color="k", linestyle="-.", linewidth=1.0, label="dim V(x0)=5" if ax is axes[0] else None)

    axes[0].legend(title="Scenario")
    fig.suptitle(
        "Estimation error vs PE order (n=5): visible dimension as ceiling"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outfile, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------


def _run_trial(
    label: str,
    U: np.ndarray,
    pe_target: int,
    pe_actual: int,
    pe_moment: int,
    system: Dict[str, np.ndarray],
    cfg: PEVisibleConfig,
):
    """Simulate and identify for one scenario."""

    x0 = system["x0"]
    Ad = system["Ad"]
    Bd = system["Bd"]
    P = system["P"]

    X = simulate_dt(x0, Ad, Bd, U, noise_std=cfg.noise_std)
    X0, X1 = X[:, :-1], X[:, 1:]
    Ahat, Bhat = dmdc_tls(X0, X1, U)

    errA = float(np.linalg.norm(Ahat - Ad, ord="fro"))
    errB = float(np.linalg.norm(Bhat - Bd, ord="fro"))
    normA = float(np.linalg.norm(Ad, ord="fro"))
    normB = float(np.linalg.norm(Bd, ord="fro"))

    dA_V, dB_V = projected_errors(Ahat, Bhat, Ad, Bd, P)
    A_vis = float(np.linalg.norm(P.T @ Ad @ P, ord="fro"))
    B_vis = float(np.linalg.norm(P.T @ Bd, ord="fro"))

    return {
        "scenario": label,
        "pe_order_target": pe_target,
        "pe_order_actual": pe_actual,
        "pe_order_moment": pe_moment,
        "errA_rel": _relative_norm(errA, normA, cfg.eps_norm),
        "errB_rel": _relative_norm(errB, normB, cfg.eps_norm),
        "errA_V_rel": _relative_norm(dA_V, A_vis, cfg.eps_norm),
        "errB_V_rel": _relative_norm(dB_V, B_vis, cfg.eps_norm),
        "dim_visible": system["dim_visible"],
        "T": U.shape[0],
    }


def run_experiment(cfg: PEVisibleConfig) -> pd.DataFrame:
    """Run the PE order vs visible-subspace experiment."""

    base_rng = np.random.default_rng(cfg.seed)
    sys_rng = np.random.default_rng(base_rng.integers(0, 2**32))
    input_rng = np.random.default_rng(base_rng.integers(0, 2**32))

    # Draw systems for the partially visible and fully visible scenarios.
    partial = {}
    partial_keys = ("A", "B", "Ad", "Bd", "x0", "P")
    values = _draw_system_with_visible_dim(cfg, target_dim=3, rng=sys_rng)
    partial.update(dict(zip(partial_keys, values)))
    partial["dim_visible"] = 3

    full = {}
    values = _draw_system_with_visible_dim(cfg, target_dim=cfg.n, rng=sys_rng)
    full.update(dict(zip(partial_keys, values)))
    full["dim_visible"] = cfg.n

    records: List[Dict[str, float]] = []
    max_order = max(cfg.pe_orders)

    for order in cfg.pe_orders:
        horizon = _horizon_for_order(order, cfg)
        for trial in range(cfg.trials_per_order):
            U = prbs(horizon, cfg.m, scale=cfg.u_scale, dwell=cfg.dwell, rng=input_rng)
            pe_block = int(estimate_pe_order(U, s_max=max_order))
            pe_moment = int(estimate_moment_pe_order(U, r_max=max_order, dt=cfg.dt))
            records.append(_run_trial("partial", U, order, pe_block, pe_moment, partial, cfg))
            records.append(_run_trial("full", U, order, pe_block, pe_moment, full, cfg))

    df = pd.DataFrame.from_records(records)

    outdir = pathlib.Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "pe_vs_visible.csv", index=False)
    _plot_summary(df, cfg, outdir / "pe_vs_visible.png")

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials-per-order", type=int, default=25)
    parser.add_argument("--outdir", type=str, default="out_pe_vs_visible")
    args = parser.parse_args(argv)

    cfg = PEVisibleConfig(
        seed=args.seed,
        trials_per_order=args.trials_per_order,
        outdir=args.outdir,
    )

    df = run_experiment(cfg)
    summary = (
        df.groupby(["scenario", "pe_order_actual"])["errA_V_rel"]
        .median()
        .reset_index()
    )
    print(summary)


if __name__ == "__main__":
    main()