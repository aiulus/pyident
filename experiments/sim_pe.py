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

Earlier runs:
    python -m pyident.experiments.sim_pe --dt 0.01 --T 20 --n 8 --vdim 3
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
from typing_extensions import Literal

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..estimators import dmdc_tls
from ..metrics import projected_errors, build_visible_basis_dt
from ..signals import estimate_moment_pe_order, estimate_pe_order, multisine
from ..simulation import prbs, simulate_dt
from .visible_sampling import VisibleDrawConfig, draw_system_state_with_visible_dim
from .interpretation_aids import create_theory_validation_plots


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PEVisibleConfig(ExperimentConfig):
    """Configuration for the PE-order versus visible-subspace study."""

    n: int = 5
    visible_dim: int = 3
    m: int = 2
    T: int = 200  # Fixed horizon for all PE orders
    dt: float = 0.1
    signal_type: Literal["prbs", "multisine"] = "prbs"
    dwell: int = 1  # PRBS only
    u_scale: float = 3.0  # PRBS only
    k_lines: int = 8  # multisine only
    noise_std: float = 0.0

    pe_orders: Sequence[int] = tuple(range(1, 11))
    ntrials: int = 25
    n_systems: int = 100  # Number of systems per scenario
    T_min: int = 0
    max_system_attempts: int = 500
    max_x0_attempts: int = 256
    visible_tol: float = 1e-8
    eps_norm: float = 1e-12
    outdir: str = "out_pe_vs_visible"
    deterministic_x0: bool = False
    enforce_exact_block_pe: bool = True
    enhanced_plots: bool = False

    def __post_init__(self) -> None:  # type: ignore[override]
        super().__post_init__()
        if self.ntrials < 1:
            raise ValueError("ntrials must be positive.")
        if not self.pe_orders:
            raise ValueError("pe_orders cannot be empty.")
        if min(self.pe_orders) <= 0:
            raise ValueError("pe_orders must be positive integers.")
        if self.T_min < 0:
            raise ValueError("T_min must be non-negative.")
        if self.max_x0_attempts <= 0:
            raise ValueError("max_x0_attempts must be positive.")
        if not 0 < self.visible_dim <= self.n:
            raise ValueError("visible_dim must satisfy 0 < visible_dim <= n.")
        if self.signal_type not in ("prbs", "multisine"):
            raise ValueError("signal_type must be 'prbs' or 'multisine'.")
        if self.signal_type == "prbs" and self.dwell <= 0:
            raise ValueError("dwell must be positive for PRBS signals.")
        if self.signal_type == "multisine" and self.k_lines <= 0:
            raise ValueError("k_lines must be positive for multisine signals.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_system_dict(
    A: np.ndarray, B: np.ndarray, Ad: np.ndarray, Bd: np.ndarray, 
    x0: np.ndarray, P: np.ndarray, dim_visible: int, system_id: int
) -> Dict[str, float | int | np.ndarray]:
    """Create a system dictionary with proper typing."""
    return {
        "A": A, "B": B, "Ad": Ad, "Bd": Bd, "x0": x0, "P": P,
        "dim_visible": dim_visible, "system_id": system_id
    }

def _draw_system_with_visible_dim(
    cfg: PEVisibleConfig,
    target_dim: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw (A, B, Ad, Bd, x0, P) with dim span(P) == target_dim.
    
    Note: The visible basis P is computed in discrete time using (Ad, Bd, x0).
    While the theoretical framework develops V(x0) for continuous-time systems,
    we use the discretized version for practical computation. For stable systems
    with reasonable dt, these subspaces typically align, but near-aliasing effects
    could create mild degeneracies at specific dt values.
    """

    draw_cfg = VisibleDrawConfig(
        n=cfg.n,
        m=cfg.m,
        dt=cfg.dt,
        dim_visible=target_dim,
        ensemble="stable",
        max_system_attempts=cfg.max_system_attempts,
        max_x0_attempts=cfg.max_x0_attempts,
        tol=cfg.visible_tol,
        deterministic_x0=cfg.deterministic_x0,
    )
    for _ in range(cfg.max_system_attempts):
        A, B, Ad, Bd, x0, P = draw_system_state_with_visible_dim(draw_cfg, rng)
        Qv = build_visible_basis_dt(Ad, Bd, x0, tol=cfg.visible_tol)
        if Qv.shape[1] == target_dim:
            return A, B, Ad, Bd, x0, Qv

    raise RuntimeError(
        "Failed to draw a system whose discrete-time visible subspace matches the "
        f"requested dimension {target_dim}."
    )


def _horizon_for_order(order: int, cfg: PEVisibleConfig) -> int:
    """Use fixed T from config, but check if it's sufficient for target PE order."""
    # Full row rank at depth r requires T >= m*r + r - 1
    T_needed = cfg.m * order + order - 1
    if cfg.T < T_needed:
        raise ValueError(
            f"Fixed horizon T={cfg.T} is insufficient for PE order {order}. "
            f"Need at least T >= {T_needed} = m*r + r - 1."
        )
    return cfg.T

def _make_prbs_with_order(
    order: int,
    cfg: PEVisibleConfig,
    rng: np.random.Generator,
    max_tries: int = 20,
) -> Tuple[np.ndarray, int]:
    """Draw a PRBS sequence that achieves the requested block-PE order with normalized energy."""

    T = _horizon_for_order(order, cfg)
    best_U: np.ndarray | None = None
    best_r: int = -1
    best_score: Tuple[int, int, int] | None = None
    
    # Target energy for normalization (based on u_scale and T)
    target_energy = float(cfg.u_scale**2 * T * cfg.m)

    for _ in range(max_tries):
        U = prbs(T, cfg.m, scale=cfg.u_scale, dwell=cfg.dwell, rng=rng)
        
        # Normalize to constant total energy
        current_energy = float(np.sum(U**2))
        if current_energy > 0:
            U = U * np.sqrt(target_energy / current_energy)
        
        r = int(estimate_pe_order(U, s_max=order))
        r_plus = int(estimate_pe_order(U, s_max=order + 1))
        if r == order and r_plus < order + 1:
            return U, r

        excess = max(0, r_plus - order)
        score = (abs(r - order), excess, r_plus)
        if best_score is None or score < best_score:
            best_U = U
            best_r = r
            best_score = score

    if best_U is None:
        best_U = prbs(T, cfg.m, scale=cfg.u_scale, dwell=cfg.dwell, rng=rng)
        best_r = int(estimate_pe_order(best_U, s_max=order))
    
    # Apply energy normalization to final result
    final_U = best_U  # Type narrowing
    current_energy = float(np.sum(final_U**2))
    if current_energy > 0:
        final_U = final_U * np.sqrt(target_energy / current_energy)
    
    return final_U, best_r


def _make_multisine_with_order(
    order: int,
    cfg: PEVisibleConfig,
    rng: np.random.Generator,
    max_tries: int = 20,
) -> Tuple[np.ndarray, int]:
    """Draw a multisine sequence that achieves the requested block-PE order with normalized energy.
    
    Note: Multisine signals may have limited PE order capability depending on T and k_lines.
    This function implements early stopping when the target order appears unachievable.
    """
    
    T = _horizon_for_order(order, cfg)
    best_U: np.ndarray | None = None
    best_r: int = -1
    best_score: Tuple[int, int, int] | None = None
    
    # Target energy for normalization (for fair comparison with PRBS)
    target_energy = float(3.0**2 * T * cfg.m)  # Use consistent energy scaling like PRBS
    
    # Early stopping: if we don't improve after several tries, give up
    no_improvement_count = 0
    early_stop_threshold = max(5, max_tries // 4)

    for trial in range(max_tries):
        U = multisine(T, cfg.m, rng=rng, k_lines=cfg.k_lines)
        
        # Normalize to constant total energy
        current_energy = float(np.sum(U**2))
        if current_energy > 0:
            U = U * np.sqrt(target_energy / current_energy)
        
        r = int(estimate_pe_order(U, s_max=order))
        r_plus = int(estimate_pe_order(U, s_max=order + 1))
        if r == order and r_plus < order + 1:
            return U, r

        excess = max(0, r_plus - order)
        score = (abs(r - order), excess, r_plus)
        improved = best_score is None or score < best_score
        
        if improved:
            best_U = U
            best_r = r
            best_score = score
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        # Early stopping: if we've tried several times without improvement, 
        # the target PE order may be unachievable with current parameters
        if no_improvement_count >= early_stop_threshold and trial >= early_stop_threshold:
            break

    if best_U is None:
        best_U = multisine(T, cfg.m, rng=rng, k_lines=cfg.k_lines)
        best_r = int(estimate_pe_order(best_U, s_max=order))
    
    # Apply energy normalization to final result
    final_U = best_U  # Type narrowing
    current_energy = float(np.sum(final_U**2))
    if current_energy > 0:
        final_U = final_U * np.sqrt(target_energy / current_energy)
    
    return final_U, best_r


def _make_signal_with_order(
    order: int,
    cfg: PEVisibleConfig,
    rng: np.random.Generator,
    max_tries: int = 20,
) -> Tuple[np.ndarray, int]:
    """Generate a signal that achieves the requested block-PE order based on cfg.signal_type."""
    
    if cfg.signal_type == "prbs":
        return _make_prbs_with_order(order, cfg, rng, max_tries)
    elif cfg.signal_type == "multisine":
        return _make_multisine_with_order(order, cfg, rng, max_tries)
    else:
        raise ValueError(f"Unknown signal type: {cfg.signal_type}")


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

def _plot_standard_by_visibility(
    df: pd.DataFrame,
    cfg: PEVisibleConfig,
    outfile: pathlib.Path,
    x_col: str = "pe_order_actual",
    xlabel: str | None = "Estimated block-PE order (Hankel)",
) -> None:
    """Two panels: left k=--vdim (partial), right k=n (full), standard-basis errors only."""
    outfile.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    panels = [
        ("partial", cfg.visible_dim, f"k = {cfg.visible_dim} (partial)"),
        ("full", cfg.n, f"k = {cfg.n} (full)"),
    ]

    for ax, (scenario, k, title) in zip(axes, panels):
        sub = df[df["scenario"] == scenario]
        # A-error
        orders_A, med_A, q1_A, q3_A = _aggregate_stat(sub.set_index(x_col)["errA_rel"])
        ax.plot(orders_A, med_A, label="‖Â−A‖₍F₎ / ‖A‖₍F₎")
        ax.fill_between(orders_A, q1_A, q3_A, alpha=0.20)

        # B-error
        orders_B, med_B, q1_B, q3_B = _aggregate_stat(sub.set_index(x_col)["errB_rel"])
        ax.plot(orders_B, med_B, linestyle="--", label="‖ B̂−B ‖₍F₎ / ‖B‖₍F₎")
        ax.fill_between(orders_B, q1_B, q3_B, alpha=0.20)

        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axvline(k, color="k", linestyle=":", linewidth=1.0)  # expected elbow at k
        ax.set_xlabel(xlabel or x_col)

    axes[0].set_ylabel("Relative error (Frobenius)")
    axes[0].legend(title="Standard-basis errors")
    fig.suptitle("Estimation error vs. PEness (standard basis)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outfile, dpi=160)
    plt.close(fig)


def _plot_markov_parameters(
    df: pd.DataFrame,
    cfg: PEVisibleConfig,
    outfile: pathlib.Path,
) -> None:
    """Plot Markov parameter errors vs moment-PE order for direct theoretical validation."""
    outfile.parent.mkdir(parents=True, exist_ok=True)
    
    # Create subplots for each Markov parameter order k
    n_params = min(cfg.n, 6)  # Limit to first 6 for readability
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True)
    axes = axes.flat
    
    for k in range(n_params):
        ax = axes[k]
        markov_col = f"markov_err_{k}"
        
        if markov_col not in df.columns:
            continue
            
        for scenario, linestyle, color in [("partial", "-", "C0"), ("full", "--", "C1")]:
            sub = df[df["scenario"] == scenario]
            if sub.empty:
                continue
            grouped = sub.set_index("pe_order_moment")[markov_col]
            orders, med, q1, q3 = _aggregate_stat(grouped)
            ax.plot(orders, med, linestyle=linestyle, color=color, label=scenario if k == 0 else "")
            ax.fill_between(orders, q1, q3, color=color, alpha=0.2)
        
        ax.set_title(f"E_{k} = ||Â^{k}B̂ - A^{k}B||_F")
        ax.set_ylabel("Markov parameter error")
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at k+1 (theoretical threshold)
        ax.axvline(k + 1, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
        if k == 0:
            ax.text(k + 1.1, ax.get_ylim()[1] * 0.9, f"r={k+1}", rotation=90, 
                   verticalalignment='top', color="red", fontsize=10)
    
    # Hide unused subplots
    for k in range(n_params, len(axes)):
        axes[k].set_visible(False)
    
    # Common x-label
    for ax in axes[-3:]:
        ax.set_xlabel("Moment-PE order")
    
    axes[0].legend(title="Scenario")
    fig.suptitle("Markov Parameter Errors vs Moment-PE Order (Theory: E_k ↓ when r > k)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outfile, dpi=160)
    plt.close(fig)

def _plot_subspace_errors(
    df: pd.DataFrame, 
    cfg: PEVisibleConfig,
    outfile: pathlib.Path,
) -> None:
    """Plot subspace-resolved errors to show ceiling effect."""
    outfile.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    
    # Top row: A errors on V(x0) and V(x0)⊥
    # Bottom row: B errors on V(x0) and V(x0)⊥
    metrics = [
        ("errA_V_subspace_rel", "A errors on V(x₀)", 0, 0),
        ("errA_Vperp_subspace_rel", "A errors on V(x₀)⊥", 0, 1), 
        ("errB_V_subspace_rel", "B errors on V(x₀)", 1, 0),
        ("errB_Vperp_subspace_rel", "B errors on V(x₀)⊥", 1, 1),
    ]
    
    for col, title, row, column in metrics:
        ax = axes[row, column]
        
        for scenario, linestyle, color in [("partial", "-", "C0"), ("full", "--", "C1")]:
            sub = df[df["scenario"] == scenario]
            if sub.empty or col not in df.columns:
                continue
            grouped = sub.set_index("pe_order_actual")[col]
            orders, med, q1, q3 = _aggregate_stat(grouped)
            ax.plot(orders, med, linestyle=linestyle, color=color, label=scenario)
            ax.fill_between(orders, q1, q3, color=color, alpha=0.2)
        
        ax.set_title(title)
        ax.set_ylabel("Relative error")
        ax.grid(True, alpha=0.3)
        
        # Add vertical lines for visible dimensions
        if "V(x₀)⊥" not in title:  # Only for V(x0) subspace
            ax.axvline(cfg.visible_dim, color="C0", linestyle=":", label=f"k={cfg.visible_dim} (partial)")
            ax.axvline(cfg.n, color="C1", linestyle="-.", label=f"k={cfg.n} (full)")
    
    # Set common x-labels
    axes[1, 0].set_xlabel("PE order")
    axes[1, 1].set_xlabel("PE order")
    
    axes[0, 0].legend(title="Scenario")
    fig.suptitle("Subspace-Resolved Errors (Ceiling Effect: V(x₀) improves, V(x₀)⊥ stays high)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outfile, dpi=160)
    plt.close(fig)

def _plot_summary(
    df: pd.DataFrame,
    cfg: PEVisibleConfig,
    outfile: pathlib.Path,
    x_col: str = "pe_order_actual",
    xlabel: str | None = None,
) -> None:
    """Plot relative errors vs a chosen PE-order column for each scenario."""
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    metrics = [
        ("errA_rel", "‖Â−A‖₍F₎ / ‖A‖₍F₎"),
        ("errA_V_rel", "Visible-projected ‖Â−A‖₍F₎"),
        ("errB_rel", "‖ B̂−B ‖₍F₎ / ‖B‖₍F₎"),
        ("errB_V_rel", "Visible-projected ‖ B̂−B ‖₍F₎"),
    ]

    for ax, (col, label) in zip(axes.flat, metrics):
        for scenario, linestyle, color in [("partial", "-", "C0"), ("full", "--", "C1")]:
            sub = df[df["scenario"] == scenario]
            if sub.empty:
                continue
            grouped = sub.set_index(x_col)[col]
            orders, med, q1, q3 = _aggregate_stat(grouped)
            ax.plot(orders, med, linestyle=linestyle, color=color, label=scenario)
            ax.fill_between(orders, q1, q3, color=color, alpha=0.2)

        ax.set_xlabel(xlabel or x_col)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        line_specs = [
            (cfg.visible_dim, ":", f"dim V(x0) = {cfg.visible_dim} (partial)"),
            (cfg.n, "-.", f"dim V(x0) = {cfg.n} (full)")
        ]

        seen_positions: set[int] = set()
        for pos, style, label_text in line_specs:
            if pos in seen_positions:
                # Avoid duplicate labels/lines when dimensions coincide.
                continue
            seen_positions.add(pos)
            label = label_text if ax is axes[0, 0] else None
            ax.axvline(pos, color="k", linestyle=style, linewidth=1.0, label=label)

    axes[0, 0].legend(title="Scenario")
    fig.suptitle(f"Estimation error vs PE order (n={cfg.n}): ambient vs visible blocks")
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
    system: Dict[str, float | int | np.ndarray],
    cfg: PEVisibleConfig,
):
    """Simulate and identify for one scenario."""

    # Extract arrays with type assertions
    x0 = system["x0"]
    Ad = system["Ad"] 
    Bd = system["Bd"]
    P = system["P"]
    assert isinstance(x0, np.ndarray)
    assert isinstance(Ad, np.ndarray)
    assert isinstance(Bd, np.ndarray) 
    assert isinstance(P, np.ndarray)

    X = simulate_dt(x0, Ad, Bd, U, noise_std=cfg.noise_std)
    X0, X1 = X[:, :-1], X[:, 1:]
    Ahat, Bhat = dmdc_tls(X0, X1, U)

    errA = float(np.linalg.norm(Ahat - Ad, ord="fro"))
    errB = float(np.linalg.norm(Bhat - Bd, ord="fro"))
    normA = float(np.linalg.norm(Ad, ord="fro"))
    normB = float(np.linalg.norm(Bd, ord="fro"))

    Q, _ = np.linalg.qr(P)
    dA_V, dB_V = projected_errors(Ahat, Bhat, Ad, Bd, Q)
    A_vis = float(np.linalg.norm(Q.T @ Ad @ Q, ord="fro"))
    B_vis = float(np.linalg.norm(Q.T @ Bd, ord="fro"))
    
    # Check for small denominators and add robustness
    A_vis_safe = max(A_vis, cfg.eps_norm)
    B_vis_safe = max(B_vis, cfg.eps_norm)
    
    # Compute Markov parameter errors E_k = ||Â^k B̂ - A^k B||_F for k = 0..n-1
    markov_errors = {}
    A_power = np.eye(cfg.n)  # A^0
    Ahat_power = np.eye(cfg.n)  # Â^0
    
    for k in range(cfg.n):
        true_markov = A_power @ Bd  # A^k @ B
        est_markov = Ahat_power @ Bhat  # Â^k @ B̂
        markov_err = float(np.linalg.norm(est_markov - true_markov, ord="fro"))
        markov_errors[f"markov_err_{k}"] = markov_err
        
        # Update powers for next iteration
        if k < cfg.n - 1:  # Don't compute beyond what we need
            A_power = A_power @ Ad
            Ahat_power = Ahat_power @ Ahat
    
    # Compute subspace-resolved errors: V(x0) vs V(x0)⊥ (ceiling effect)
    # Q columns span V(x0), so Q⊥ spans V(x0)⊥
    dim_V = Q.shape[1]
    
    # Create orthogonal complement basis Q_perp for V(x0)⊥
    if dim_V < cfg.n:
        # Complete Q to full orthogonal basis
        Q_full, _ = np.linalg.qr(np.hstack([Q, np.random.randn(cfg.n, cfg.n - dim_V)]))
        Q_perp = Q_full[:, dim_V:]  # Orthogonal complement
    else:
        # If V(x0) = R^n, there's no orthogonal complement
        Q_perp = np.zeros((cfg.n, 0))
    
    # Compute errors restricted to each subspace
    A_err = Ahat - Ad
    B_err = Bhat - Bd
    
    # Error on V(x0): ||(Â-A)|_V||_F = ||Q^T(Â-A)Q||_F  
    A_err_V = Q.T @ A_err @ Q
    errA_V_subspace = float(np.linalg.norm(A_err_V, ord="fro"))
    
    # Error on V(x0)⊥: ||(Â-A)|_{V⊥}||_F = ||Q_⊥^T(Â-A)Q_⊥||_F
    if Q_perp.shape[1] > 0:
        A_err_Vperp = Q_perp.T @ A_err @ Q_perp
        errA_Vperp_subspace = float(np.linalg.norm(A_err_Vperp, ord="fro"))
    else:
        errA_Vperp_subspace = 0.0
    
    # B errors on subspaces: ||Q^T(B̂-B)||_F and ||Q_⊥^T(B̂-B)||_F
    B_err_V = Q.T @ B_err
    errB_V_subspace = float(np.linalg.norm(B_err_V, ord="fro"))
    
    if Q_perp.shape[1] > 0:
        B_err_Vperp = Q_perp.T @ B_err
        errB_Vperp_subspace = float(np.linalg.norm(B_err_Vperp, ord="fro"))
    else:
        errB_Vperp_subspace = 0.0
    
    # Relative versions (for normalization)
    A_V_norm_subspace = float(np.linalg.norm(Q.T @ Ad @ Q, ord="fro"))
    B_V_norm_subspace = float(np.linalg.norm(Q.T @ Bd, ord="fro"))
    
    if Q_perp.shape[1] > 0:
        A_Vperp_norm_subspace = float(np.linalg.norm(Q_perp.T @ Ad @ Q_perp, ord="fro"))
        B_Vperp_norm_subspace = float(np.linalg.norm(Q_perp.T @ Bd, ord="fro"))
    else:
        A_Vperp_norm_subspace = 0.0
        B_Vperp_norm_subspace = 0.0

    return {
        "scenario": label,
        "pe_order_target": pe_target,
        "pe_order_actual": pe_actual,
        "pe_order_moment": pe_moment,
        "errA_rel": _relative_norm(errA, normA, cfg.eps_norm),
        "errB_rel": _relative_norm(errB, normB, cfg.eps_norm),
        "errA_V_abs": float(dA_V),
        "errB_V_abs": float(dB_V),
        "errA_V_rel": _relative_norm(dA_V, A_vis_safe, cfg.eps_norm),
        "errB_V_rel": _relative_norm(dB_V, B_vis_safe, cfg.eps_norm),
        "A_vis_norm": A_vis,
        "B_vis_norm": B_vis,
        "dim_visible": system["dim_visible"],
        "T": U.shape[0],
        # Subspace-resolved errors (ceiling effect)
        "errA_V_subspace_abs": errA_V_subspace,
        "errB_V_subspace_abs": errB_V_subspace,  
        "errA_Vperp_subspace_abs": errA_Vperp_subspace,
        "errB_Vperp_subspace_abs": errB_Vperp_subspace,
        "errA_V_subspace_rel": _relative_norm(errA_V_subspace, A_V_norm_subspace, cfg.eps_norm),
        "errB_V_subspace_rel": _relative_norm(errB_V_subspace, B_V_norm_subspace, cfg.eps_norm),
        "errA_Vperp_subspace_rel": _relative_norm(errA_Vperp_subspace, A_Vperp_norm_subspace, cfg.eps_norm),
        "errB_Vperp_subspace_rel": _relative_norm(errB_Vperp_subspace, B_Vperp_norm_subspace, cfg.eps_norm),
        "A_V_norm_subspace": A_V_norm_subspace,
        "B_V_norm_subspace": B_V_norm_subspace,
        "A_Vperp_norm_subspace": A_Vperp_norm_subspace,
        "B_Vperp_norm_subspace": B_Vperp_norm_subspace,
        **markov_errors,  # Add all markov_err_0, markov_err_1, ..., markov_err_{n-1}
    }


def run_experiment(cfg: PEVisibleConfig) -> pd.DataFrame:
    """Run the PE order vs visible-subspace experiment."""

    base_rng = np.random.default_rng(cfg.seed)
    sys_rng = np.random.default_rng(base_rng.integers(0, 2**32))
    input_rng = np.random.default_rng(base_rng.integers(0, 2**32))

    # Draw multiple systems for each scenario (reused across all PE orders)
    print(f"Drawing {cfg.n_systems} systems for each scenario...")
    
    partial_systems = []
    for i in range(cfg.n_systems):
        A, B, Ad, Bd, x0, P = _draw_system_with_visible_dim(cfg, target_dim=cfg.visible_dim, rng=sys_rng)
        system = _create_system_dict(A, B, Ad, Bd, x0, P, cfg.visible_dim, i)
        partial_systems.append(system)
        if (i + 1) % 20 == 0:
            print(f"  Partial systems: {i + 1}/{cfg.n_systems}")

    full_systems = []
    for i in range(cfg.n_systems):
        A, B, Ad, Bd, x0, P = _draw_system_with_visible_dim(cfg, target_dim=cfg.n, rng=sys_rng)
        system = _create_system_dict(A, B, Ad, Bd, x0, P, cfg.n, i)
        full_systems.append(system)
        if (i + 1) % 20 == 0:
            print(f"  Full systems: {i + 1}/{cfg.n_systems}")

    records: List[Dict[str, float]] = []
    max_order = max(cfg.pe_orders)

    print("Running experiments...")
    for order in cfg.pe_orders:
        print(f"  PE order {order}:")
        for trial in range(cfg.ntrials):
            U, pe_block = _make_signal_with_order(order, cfg, input_rng)
            pe_moment = int(estimate_moment_pe_order(U, r_max=max_order, dt=cfg.dt))
            
            # Run trials on all systems for this PE order and input
            for sys_partial in partial_systems:
                result = _run_trial("partial", U, order, pe_block, pe_moment, sys_partial, cfg)
                result["system_id"] = sys_partial["system_id"]
                records.append(result)
                
            for sys_full in full_systems:
                result = _run_trial("full", U, order, pe_block, pe_moment, sys_full, cfg)
                result["system_id"] = sys_full["system_id"]
                records.append(result)
                
            if (trial + 1) % 5 == 0:
                print(f"    Trial {trial + 1}/{cfg.ntrials}")

    df = pd.DataFrame.from_records(records)

    outdir = pathlib.Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "pe_vs_visible.csv", index=False)

    # Two-panel plot: left k=3 partial, right k=5 full, standard basis, x = used PEness
    _plot_standard_by_visibility(
        df, cfg, outdir / "pe_vs_visible_standard_basis_blockPE.png",
        x_col="pe_order_actual",
        xlabel="Estimated block-PE order (Hankel)"
    )

    _plot_standard_by_visibility(
        df,
        cfg,
        outdir / "pe_vs_visible_standard_basis_momentPE.png",
        x_col="pe_order_moment",
        xlabel="Estimated moment-PE order",
    )

    _plot_summary(
        df,
        cfg,
        outdir / "pe_vs_visible_ambient_vs_visible_blockPE.png",
        x_col="pe_order_actual",
        xlabel="Estimated block-PE order (Hankel)",
    )
    
    # New theory-focused plots
    _plot_markov_parameters(df, cfg, outdir / "pe_vs_visible_markov_parameters.png")
    _plot_subspace_errors(df, cfg, outdir / "pe_vs_visible_subspace_ceiling_effect.png")
    
    # Enhanced interpretation aids (optional)
    if cfg.enhanced_plots:
        create_theory_validation_plots(df, cfg, outdir)
        # Also generate a human-readable summary
        _generate_results_summary(df, cfg, outdir)

    return df


def _generate_results_summary(df: pd.DataFrame, cfg: PEVisibleConfig, outdir: pathlib.Path) -> None:
    """Generate a human-readable summary of key findings."""
    
    summary_lines = [
        "=" * 60,
        "PE vs VISIBLE SUBSPACE EXPERIMENT SUMMARY",
        "=" * 60,
        f"Configuration: n={cfg.n}, vdim={cfg.visible_dim}, m={cfg.m}",
        f"Systems tested: {cfg.n_systems} per scenario",
        f"PE orders: {min(cfg.pe_orders)} to {max(cfg.pe_orders)}",
        f"Trials per order: {cfg.ntrials}",
        "",
        "KEY FINDINGS:",
        "-" * 20,
    ]
    
    # Main threshold effects
    for scenario in ['partial', 'full']:
        scenario_df = df[df['scenario'] == scenario]
        if scenario_df.empty:
            continue
            
        visible_dim = scenario_df['dim_visible'].iloc[0]
        
        # Test main effect
        before = scenario_df[scenario_df['pe_order_actual'] <= visible_dim]['errA_V_rel']
        after = scenario_df[scenario_df['pe_order_actual'] > visible_dim]['errA_V_rel']
        
        if len(before) > 0 and len(after) > 0:
            from scipy import stats
            _, p_val = stats.mannwhitneyu(after, before, alternative='less')
            effect = (before.median() - after.median()) / before.median() * 100
            
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            
            summary_lines.append(
                f"• {scenario.upper()}: {effect:+.1f}% error reduction when r > {visible_dim} "
                f"(p={p_val:.3f}{significance})"
            )
    
    # Markov parameter validation
    summary_lines.extend(["", "THEORETICAL VALIDATION:", "-" * 25])
    
    markov_successes = 0
    markov_total = 0
    
    for k in range(min(cfg.n, 4)):
        markov_col = f'markov_err_{k}'
        if markov_col not in df.columns:
            continue
            
        before = df[df['pe_order_moment'] <= k][markov_col]
        after = df[df['pe_order_moment'] > k][markov_col]
        
        if len(before) > 0 and len(after) > 0:
            from scipy import stats
            _, p_val = stats.mannwhitneyu(after, before, alternative='less')
            effect = (before.median() - after.median()) / before.median() * 100
            
            markov_total += 1
            if p_val < 0.05 and effect > 0:
                markov_successes += 1
                
            status = "✓" if p_val < 0.05 and effect > 0 else "✗"
            summary_lines.append(f"• Markov E_{k} improves when r > {k}: {status} ({effect:+.1f}%, p={p_val:.3f})")
    
    # Ceiling effect
    summary_lines.extend(["", "CEILING EFFECT:", "-" * 15])
    partial_df = df[df['scenario'] == 'partial']
    if not partial_df.empty:
        v_errors = partial_df['errA_V_subspace_rel']
        vperp_errors = partial_df['errA_Vperp_subspace_rel']
        
        if len(v_errors) > 0 and len(vperp_errors) > 0:
            ratio = vperp_errors.median() / v_errors.median() if v_errors.median() > 0 else np.inf
            from scipy import stats
            _, p_val = stats.mannwhitneyu(vperp_errors, v_errors, alternative='greater')
            
            status = "✓" if p_val < 0.05 and ratio > 10 else "✗"
            summary_lines.append(f"• V⊥ errors >> V errors: {status} ({ratio:.0f}x ratio, p={p_val:.3f})")
    
    # Overall assessment
    summary_lines.extend(["", "OVERALL ASSESSMENT:", "-" * 20])
    
    markov_success_rate = 0.0
    if markov_total > 0:
        markov_success_rate = markov_successes / markov_total * 100
        
        if markov_success_rate >= 75:
            theory_support = "STRONG"
        elif markov_success_rate >= 50:
            theory_support = "MODERATE"  
        else:
            theory_support = "WEAK"
            
        summary_lines.append(f"• Theory validation: {theory_support} ({markov_success_rate:.0f}% of Markov tests passed)")
    
    # Recommendations
    summary_lines.extend(["", "RECOMMENDATIONS:", "-" * 16])
    
    if markov_total > 0 and markov_success_rate < 50:
        summary_lines.append("• Consider increasing sample size (--n-systems, --n-trials)")
        summary_lines.append("• Try longer horizons (--T) for better moment-PE estimation")
    
    if cfg.n_systems < 50:
        summary_lines.append("• Increase --n-systems to 100+ for more reliable statistics")
        
    if cfg.ntrials < 20:
        summary_lines.append("• Increase --n-trials to 25+ for better coverage")
    
    summary_lines.extend(["", "=" * 60])
    
    # Write summary
    with open(outdir / "RESULTS_SUMMARY.txt", "w") as f:
        f.write("\n".join(summary_lines))
    
    # Also print key findings
    print("\n" + "\n".join(summary_lines[:15]) + "\n...")
    print(f"📊 Full summary saved to: {outdir / 'RESULTS_SUMMARY.txt'}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    
    # Basic experiment parameters
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--ntrials", "--n-trials", type=int, default=25, help="Number of trials per PE order")
    parser.add_argument("--n-systems", type=int, default=100, help="Number of systems per scenario")
    parser.add_argument("--outdir", type=str, default="out_pe_vs_visible", help="Output directory")
    parser.add_argument("--output-prefix", type=str, help="Output file prefix (alternative to --outdir)")
    
    # System dimensions
    parser.add_argument("--n", type=int, default=5, help="State dimension for the system")
    parser.add_argument("--vdim", type=int, default=3, help="Target visible subspace dimension dim(V(x0)) for the partial scenario")
    parser.add_argument("--m", type=int, default=2, help="Input dimension")
    
    # Simulation parameters
    parser.add_argument("--T", type=int, default=200, help="Fixed simulation horizon length for all PE orders")
    parser.add_argument("--dt", type=float, default=0.1, help="Discretization time step")
    parser.add_argument("--sigtype", "--signal-type", type=str, default="prbs", choices=["prbs", "multisine"], 
                        help="Signal type for excitation")
    parser.add_argument("--dwell", type=int, default=1, help="PRBS dwell time (PRBS only)")
    parser.add_argument("--u-scale", type=float, default=3.0, help="Input scaling factor (PRBS only)")
    parser.add_argument("--k-lines", type=int, default=8, help="Number of frequency lines (multisine only)")
    parser.add_argument("--noise-std", type=float, default=0.0, help="Measurement noise standard deviation")
    
    # PE order configuration
    parser.add_argument("--pe-orders", type=str, default="1,2,3,4,5,6,7,8,9,10", 
                        help="Comma-separated list of PE orders to test")
    parser.add_argument("--max-pe-order", type=int, help="Maximum PE order (generates range 1 to max-pe-order)")
    
    # Algorithm parameters
    parser.add_argument("--T-min", type=int, default=0, help="Minimum horizon length")
    parser.add_argument("--max-system-attempts", type=int, default=500, help="Maximum attempts to draw suitable system")
    parser.add_argument("--max-x0-attempts", type=int, default=256, help="Maximum attempts to draw suitable initial state")
    
    # Tolerance parameters
    parser.add_argument("--visible-tol", type=float, default=1e-8, help="Tolerance for visible subspace computation")
    parser.add_argument("--eps-norm", type=float, default=1e-12, help="Epsilon for relative norm computation")
    
    # Behavioral flags
    parser.add_argument("--det", action="store_true", help="Use deterministic x0 construction")
    parser.add_argument("--exact-pe", dest="exact_pe", action="store_true", default=True, help="Enforce exact block PE order")
    parser.add_argument("--no-exact-pe", dest="exact_pe", action="store_false", help="Disable exact block PE order enforcement")
    parser.add_argument("--enhanced-plots", action="store_true", help="Generate enhanced interpretation dashboard and statistical tests")
    
    args = parser.parse_args(argv)
    
    # Handle --max-pe-order vs --pe-orders
    if args.max_pe_order is not None:
        pe_orders = tuple(range(1, args.max_pe_order + 1))
    else:
        # Parse PE orders from comma-separated string
        pe_orders = tuple(int(x.strip()) for x in args.pe_orders.split(",") if x.strip())
    
    # Handle --output-prefix vs --outdir  
    if args.output_prefix is not None:
        outdir = args.output_prefix
    else:
        outdir = args.outdir

    cfg = PEVisibleConfig(
        seed=args.seed,
        ntrials=args.ntrials,
        n_systems=args.n_systems,
        outdir=outdir,
        n=args.n,
        visible_dim=args.vdim,
        m=args.m,
        T=args.T,
        dt=args.dt,
        signal_type=args.sigtype,
        dwell=args.dwell,
        u_scale=args.u_scale,
        k_lines=getattr(args, 'k_lines', 8),  # Handle hyphenated argument
        noise_std=args.noise_std,
        pe_orders=pe_orders,
        T_min=args.T_min,
        max_system_attempts=args.max_system_attempts,
        max_x0_attempts=args.max_x0_attempts,
        visible_tol=args.visible_tol,
        eps_norm=args.eps_norm,
        deterministic_x0=args.det,
        enforce_exact_block_pe=args.exact_pe,
        enhanced_plots=args.enhanced_plots,
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