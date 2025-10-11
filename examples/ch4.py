from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D plots

from pyident.metrics import (
    cont2discrete_zoh,
    gramian_dt_finite,
    krylov_generator,
    pbh_margin_structured,
)
from pyident.simulation import simulate_dt
from pyident.signals import estimate_pe_order, multisine, prbs


# ---------------------------------------------------------------------------
# Problem data from the example in the manuscript
# ---------------------------------------------------------------------------
A = np.array(
    [
        [1.0, 1.0, 0.0],
        [0.0, 2.0, 1.0],
        [0.0, 0.0, 3.0],
    ]
)
B = np.array(
    [
        [1.0, 0.0],
        [2.0, 1.0],
        [0.0, 0.0],
    ]
)
A_TILDE = np.array(
    [
        [1.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 4.0],
    ]
)
B_TILDE = B.copy()

X0_GOOD = np.array([0.0, 0.0, 1.0])
X0_BAD = np.array([1.0, -1.0, 0.0])


@dataclass
class Metrics:
    krylov_rank: int
    mu_min: float
    pbh_margin: float
    gramian_lambda_min: float


# ---------------------------------------------------------------------------
# Signal generation helpers
# ---------------------------------------------------------------------------
def generate_signal(
    shape: str,
    length: int,
    dt: float,
    frequency: float,
    amplitude: float,
    prbs_period: int,
    seed: int | None,
) -> np.ndarray:
    """Generate a 2-channel excitation sequence."""

    rng = np.random.default_rng(seed)
    t = np.arange(length) * dt

    if shape == "sinusoid":
        u1 = amplitude * np.sin(2.0 * np.pi * frequency * t)
        # phase shift the second channel for visibility
        u2 = amplitude * np.sin(2.0 * np.pi * frequency * t + 0.5 * np.pi)
        u = np.column_stack((u1, u2))
    elif shape == "prbs":
        base = prbs(length, m=2, rng=rng, period=prbs_period)
        u = amplitude * base
    elif shape == "multisine":
        base = multisine(length, m=2, rng=rng, k_lines=max(4, int(np.ceil(frequency))))
        u = amplitude * base
    else:
        raise ValueError(f"Unsupported signal shape '{shape}'.")

    return u


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------
def compute_mu_min(A_mat: np.ndarray, B_mat: np.ndarray, x0: np.ndarray) -> float:
    """Compute μ_min = min_i ||w_i^T [x0 B]||^2 / ||w_i||^2."""

    _, eigvecs = np.linalg.eig(A_mat.T)
    generator = np.column_stack((x0.reshape(-1, 1), B_mat))
    mu_vals = []
    for idx in range(eigvecs.shape[1]):
        w = eigvecs[:, idx]
        row = w.conj().T
        numerator = np.linalg.norm(row @ generator) ** 2
        denominator = np.linalg.norm(row) ** 2
        mu_vals.append(float(numerator / (denominator + 1e-18)))
    return float(np.min(mu_vals)) if mu_vals else 0.0


def compute_metrics(
    A_mat: np.ndarray,
    B_mat: np.ndarray,
    x0: np.ndarray,
    Ad: np.ndarray,
    Bd: np.ndarray,
    horizon: int,
) -> Metrics:
    K3 = krylov_generator(A_mat, np.column_stack((x0.reshape(-1, 1), B_mat)), depth=A_mat.shape[0])
    krylov_rank = int(np.linalg.matrix_rank(K3))
    mu_min = compute_mu_min(A_mat, B_mat, x0)
    pbh = pbh_margin_structured(A_mat, B_mat, x0)
    gramian = gramian_dt_finite(Ad, np.column_stack((x0.reshape(-1, 1), Bd)), horizon)
    sym_gram = 0.5 * (gramian + gramian.T)
    gramian_lambda_min = float(np.min(np.linalg.eigvalsh(sym_gram)))
    return Metrics(krylov_rank, mu_min, pbh, gramian_lambda_min)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _default_markevery(length: int) -> int:
    return max(1, length // 12)


def plot_state_trajectories(
    X_ref: np.ndarray,
    X_tilde: np.ndarray,
    title: str,
    ax: plt.Axes,
) -> None:
    steps = X_ref.shape[1]
    markevery = _default_markevery(steps)
    ax.plot(
        X_ref[0, :],
        X_ref[1, :],
        X_ref[2, :],
        label="(A, B)",
        color="tab:blue",
        linewidth=2.0,
        linestyle="-",
        marker="o",
        markevery=markevery,
        markersize=4,
    )
    ax.plot(
        X_tilde[0, :],
        X_tilde[1, :],
        X_tilde[2, :],
        label="(Ã, B̃)",
        color="tab:orange",
        linewidth=2.0,
        linestyle="--",
        marker="s",
        markevery=markevery,
        markersize=4,
    )
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_zlabel("x₃")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True)


def plot_signal_3d(time: np.ndarray, U: np.ndarray, ax: plt.Axes) -> None:
    markevery = _default_markevery(len(time))
    ax.plot(
        time,
        U[:, 0],
        U[:, 1],
        color="tab:green",
        linewidth=2.0,
        linestyle="-",
        marker="^",
        markevery=markevery,
        markersize=4,
    )
    ax.set_xlabel("time")
    ax.set_ylabel("u₁")
    ax.set_zlabel("u₂")
    ax.set_title("Input signal trajectory")
    ax.grid(True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate the single-trajectory x0-equivalence example with configurable inputs.",
    )
    parser.add_argument("--trajectory-length", type=int, default=120, help="Number of simulation steps (T).")
    parser.add_argument(
        "--input-length",
        type=int,
        default=None,
        help="Length of the generated input sequence (defaults to trajectory length).",
    )
    parser.add_argument("--dt", type=float, default=0.05, help="Sampling step for ZOH discretization.")
    parser.add_argument(
        "--input-shape",
        choices=["sinusoid", "prbs", "multisine"],
        default="sinusoid",
        help="Type of excitation signal to use.",
    )
    parser.add_argument("--frequency", type=float, default=0.5, help="Base frequency parameter for the signal.")
    parser.add_argument("--amplitude", type=float, default=1.0, help="Amplitude applied to the control signal.")
    parser.add_argument("--prbs-period", type=int, default=31, help="PRBS period when using the prbs signal.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for stochastic signals.")
    parser.add_argument(
        "--pe-order-max",
        type=int,
        default=10,
        help="Maximum block-Hankel depth considered when estimating the PE order.",
    )
    parser.add_argument(
        "--pe-tol",
        type=float,
        default=1e-8,
        help="Absolute tolerance passed to the PE order estimator.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="If provided, save figures using this prefix (suffixes _good.png, _bad.png, _input.png).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="If set, do not open an interactive window with the plots.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
def main(args: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    opts = parser.parse_args(args=args)

    T = opts.trajectory_length
    input_len = T if opts.input_length is None else opts.input_length
    if input_len < T:
        raise ValueError("Input length must be at least as large as the trajectory length.")

    U_full = generate_signal(
        shape=opts.input_shape,
        length=input_len,
        dt=opts.dt,
        frequency=opts.frequency,
        amplitude=opts.amplitude,
        prbs_period=opts.prbs_period,
        seed=opts.seed,
    )
    U = U_full[:T, :]

    time_grid = np.arange(T) * opts.dt

    Ad, Bd = cont2discrete_zoh(A, B, opts.dt)
    Ad_tilde, Bd_tilde = cont2discrete_zoh(A_TILDE, B_TILDE, opts.dt)

    X_good = simulate_dt(X0_GOOD, Ad, Bd, U)
    X_good_tilde = simulate_dt(X0_GOOD, Ad_tilde, Bd_tilde, U)
    X_bad = simulate_dt(X0_BAD, Ad, Bd, U)
    X_bad_tilde = simulate_dt(X0_BAD, Ad_tilde, Bd_tilde, U)

    pe_order = estimate_pe_order(U, s_max=opts.pe_order_max, tol=opts.pe_tol)

    metrics_good = compute_metrics(A, B, X0_GOOD, Ad, Bd, T)
    metrics_bad = compute_metrics(A, B, X0_BAD, Ad, Bd, T)

    # Console report
    print("=== Persistency of excitation ===")
    print(f"Estimated PE order (block-Hankel): {pe_order}")
    print()

    def _print_metrics(label: str, metrics: Metrics) -> None:
        print(f"--- {label} ---")
        print(f"rank(K_3)           : {metrics.krylov_rank}")
        print(f"mu_min              : {metrics.mu_min:.6g}")
        print(f"PBH margin          : {metrics.pbh_margin:.6g}")
        print(f"lambda_min(W_T)     : {metrics.gramian_lambda_min:.6g}")
        print()

    _print_metrics("good x0", metrics_good)
    _print_metrics("bad x0", metrics_bad)

    # Plotting
    fig_good = plt.figure(figsize=(6, 5))
    ax_good = fig_good.add_subplot(111, projection="3d")
    plot_state_trajectories(X_good, X_good_tilde, "Good initial state", ax_good)

    fig_bad = plt.figure(figsize=(6, 5))
    ax_bad = fig_bad.add_subplot(111, projection="3d")
    plot_state_trajectories(X_bad, X_bad_tilde, "Bad initial state", ax_bad)

    fig_input = plt.figure(figsize=(6, 5))
    ax_input = fig_input.add_subplot(111, projection="3d")
    plot_signal_3d(time_grid, U, ax_input)

    if opts.output_prefix is not None:
        prefix = opts.output_prefix
        fig_good.savefig(prefix.with_name(prefix.name + "_good.png"), dpi=200, bbox_inches="tight")
        fig_bad.savefig(prefix.with_name(prefix.name + "_bad.png"), dpi=200, bbox_inches="tight")
        fig_input.savefig(prefix.with_name(prefix.name + "_input.png"), dpi=200, bbox_inches="tight")

    if not opts.no_show:
        plt.show()
    else:
        plt.close(fig_good)
        plt.close(fig_bad)
        plt.close(fig_input)


if __name__ == "__main__":
    main()