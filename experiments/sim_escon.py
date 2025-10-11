from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import argparse  # NEW
import pathlib

import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..ensembles import draw_with_ctrb_rank
from ..metrics import (
    cont2discrete_zoh,
    eta0,
    projected_errors,
    regressor_stats,
    same_equiv_class_dt_rel,
    unified_generator,
)
from ..simulation import prbs as prbs_dt
from ..simulation import simulate_dt
from ..signals import estimate_pe_order
from ..estimators import (
    dmdc_pinv,
    dmdc_tls,
    moesp_fit,
    node_fit, 
    sindy_fit
)

def _sweep_estimators() -> Mapping[str, Callable[[np.ndarray, np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray]]]:
    """
    Return callables that accept (X0, X1, U_cm, dt) and return (Ahat, Bhat).
    """
    return {
        "SINDy": lambda X0, X1, U_cm, dt: sindy_fit(X0, X1, U_cm, dt),
        "MOESP": lambda X0, X1, U_cm, dt: moesp_fit(X0, X1, U_cm),
        "DMDc":  lambda X0, X1, U_cm, dt: dmdc_tls(X0, X1, U_cm),
        "NODE":  lambda X0, X1, U_cm, dt: node_fit(X0, X1, U_cm, dt, epochs=200),
    }


# ---------- NEW: PRBS helper ensuring reasonable richness ----------
def _draw_rich_prbs_for_dim(cfg: EstimatorConsistencyConfig, rng: np.random.Generator, dim_visible: int) -> Tuple[np.ndarray, int]:
    """
    Try to draw a reasonably rich PRBS (short dwell) for the given visible dimension.
    We keep T fixed (cfg.T) for reproducibility; dwell=1 maximizes variation.
    """
    dwell = 1
    U = prbs_dt(cfg.T, cfg.m, scale=cfg.u_scale, dwell=dwell, rng=rng)
    # Optional: you can check PE here and warn if it falls short
    try:
        pe_est = estimate_pe_order(U, s_max=min(cfg.pe_order_max, cfg.T // 2))
        if pe_est < dim_visible:
            # Still proceed; typically T and dwell=1 are enough in practice.
            pass
    except Exception:
        pass
    return U, dwell


# ---------- NEW: one sweep run for a single algorithm ----------
def _visibility_sweep_for_algo(
    cfg: EstimatorConsistencyConfig,
    algo_name: str,
    rng: np.random.Generator,
    ensemble_size: int = 200,
    dims: Optional[Sequence[int]] = None,
    out_dir: Optional[pathlib.Path] = None,
) -> None:
    """
    For a chosen estimator, build ensembles over dimV = n..5, simulate, fit, and save two figures:
      1) standard basis errors (A, B) as boxplots,
      2) adapted-basis errors (visible/dark) as boxplots for A and B.
    """
    algos = _sweep_estimators()
    if algo_name not in algos:
        raise ValueError(f"Unknown algorithm '{algo_name}'. Available: {list(algos.keys())}")
    estimator = algos[algo_name]

    n = cfg.n
    if n < 10:
        raise ValueError(f"This sweep assumes n=10; got n={n}. Increase cfg.n to 10.")
    # dimV in {n, n-1, ..., max(5, n-5)}
    if dims is None:
        dim_min = max(5, n - 5)
        dims = list(range(n, dim_min - 1, -1))

    # Storage per dimension
    by_dim_stdA: Dict[int, List[float]] = {k: [] for k in dims}
    by_dim_stdB: Dict[int, List[float]] = {k: [] for k in dims}
    by_dim_visA: Dict[int, List[float]] = {k: [] for k in dims}
    by_dim_visB: Dict[int, List[float]] = {k: [] for k in dims}
    by_dim_darkA: Dict[int, List[float]] = {k: [] for k in dims}
    by_dim_darkB: Dict[int, List[float]] = {k: [] for k in dims}
    by_dim_unified_std: Dict[int, List[float]] = {k: [] for k in dims}   # NEW
    by_dim_unified_vis: Dict[int, List[float]] = {k: [] for k in dims}   # NEW


    base = getattr(cfg, "partial_base_ensemble", None) or cfg.ensemble
    base = "stable" if base == "A_stbl_B_ctrb" else base

    for k in dims:
        # Build ensemble of size ensemble_size at this visible dimension
        count = 0
        while count < ensemble_size:
            # Draw a system with reachable rank = k
            try:
                A, B, _ = draw_with_ctrb_rank(
                    n=cfg.n, m=cfg.m, r=k, rng=rng, ensemble_type=base, embed_random_basis=True
                )
                Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)
                Rbasis = _reachable_basis(Ad, Bd, tol=1e-12)
                if Rbasis.shape[1] != k:
                    continue
            except Exception:
                continue

            # Build x0 either deterministically (Krylov) or via rejection (legacy)
            if getattr(cfg, "det", False):
                try:
                    x0, Vbasis = _sample_visible_initial_state_det(Ad, Bd, Rbasis, k, rng)
                except Exception:
                    # tiny fallback, try legacy once
                    try:
                        x0, Vbasis = _sample_visible_initial_state(Ad, Bd, Rbasis, k, rng, cfg.max_x0_draws)
                    except Exception:
                        continue
            else:
                try:
                    x0, Vbasis = _sample_visible_initial_state(Ad, Bd, Rbasis, k, rng, cfg.max_x0_draws)
                except Exception:
                    continue

            # Input and simulation
            U, _ = _draw_rich_prbs_for_dim(cfg, rng, k)
            X = simulate_dt(x0, Ad, Bd, U, noise_std=cfg.noise_std, rng=rng)
            X0, X1 = X[:, :-1], X[:, 1:]
            U_cm = U.T  # estimators accept (m,T)

            # Fit
            try:
                Ahat, Bhat = estimator(X0, X1, U_cm, cfg.dt)
            except Exception:
                # Skip rare failures
                continue

            # Errors
            errs = _estimation_errors(Ahat, Bhat, Ad, Bd, Vbasis)
            by_dim_stdA[k].append(errs["errA_rel"])
            by_dim_stdB[k].append(errs["errB_rel"])
            by_dim_visA[k].append(errs["errA_vis_block_rel"])
            by_dim_visB[k].append(errs["errB_vis_block_rel"])
            by_dim_darkA[k].append(errs["errA_dark_block_rel"])
            by_dim_darkB[k].append(errs["errB_dark_block_rel"])
            # Unified (A-B mean) errors  # NEW
            unified_std = 0.5 * (errs["errA_rel"] + errs["errB_rel"])
            unified_vis = 0.5 * (errs["errA_vis_block_rel"] + errs["errB_vis_block_rel"])
            by_dim_unified_std[k].append(unified_std)
            by_dim_unified_vis[k].append(unified_vis)

            count += 1

    # ---------- plotting: standard-basis ----------
    out_dir = out_dir or (cfg.save_dir / "vis_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    # ---------- Figure 1: A/B — Standard vs V(x0)-basis (visible) ----------

    dims_sorted = list(sorted(dims, reverse=True))  # n..down..5
    fig1, axes1 = plt.subplots(nrows=2, ncols=2, figsize=(12, 6), sharex=True)

    # A — standard basis (top-left)
    dataA_std = [np.asarray(by_dim_stdA[k], float) for k in dims_sorted]
    axes1[0, 0].boxplot(dataA_std, whis=(5, 95), showfliers=False)
    axes1[0, 0].set_title("A — Standard basis")
    axes1[0, 0].set_ylabel("Relative error")
    axes1[0, 0].grid(True, axis="y", linestyle="--", alpha=0.6)

    # A — V(x0)-basis (visible block) (bottom-left)
    dataA_vis = [np.asarray(by_dim_visA[k], float) for k in dims_sorted]
    axes1[1, 0].boxplot(dataA_vis, whis=(5, 95), showfliers=False)
    axes1[1, 0].set_title("A — V(x0)-basis (visible)")
    axes1[1, 0].set_xlabel("dim $V(x_0)$")
    axes1[1, 0].set_ylabel("Relative error")
    axes1[1, 0].grid(True, axis="y", linestyle="--", alpha=0.6)
    axes1[1, 0].set_xticks(list(range(1, len(dims_sorted) + 1)), [str(k) for k in dims_sorted])

    # B — standard basis (top-right)
    dataB_std = [np.asarray(by_dim_stdB[k], float) for k in dims_sorted]
    axes1[0, 1].boxplot(dataB_std, whis=(5, 95), showfliers=False)
    axes1[0, 1].set_title("B — Standard basis")
    axes1[0, 1].grid(True, axis="y", linestyle="--", alpha=0.6)

    # B — V(x0)-basis (visible block) (bottom-right)
    dataB_vis = [np.asarray(by_dim_visB[k], float) for k in dims_sorted]
    axes1[1, 1].boxplot(dataB_vis, whis=(5, 95), showfliers=False)
    axes1[1, 1].set_title("B — V(x0)-basis (visible)")
    axes1[1, 1].set_xlabel("dim $V(x_0)$")
    axes1[1, 1].grid(True, axis="y", linestyle="--", alpha=0.6)
    axes1[1, 1].set_xticks(list(range(1, len(dims_sorted) + 1)), [str(k) for k in dims_sorted])

    # subtle vertical separator between A (left col) and B (right col)
    #fig1.lines.append(plt.Line2D([0.5, 0.5], [0.08, 0.95], transform=fig1.transFigure, linewidth=1))

    fig1.suptitle(f"{algo_name}: Standard vs V(x0)-basis errors", y=0.98)
    fig1.tight_layout()
    fig1.savefig(out_dir / f"vis_sweep_{algo_name}_std_vs_V.png", dpi=150)
    plt.close(fig1)

    # ---------- NEW (Figure 2): Axis-aligned A/B comparison (Standard vs V(x0)) ----------
    from matplotlib.lines import Line2D

    fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharex=True)

    positions = np.arange(1, len(dims_sorted) + 1)
    offset = 0.18
    width = 0.32

    # A: Standard vs V(x0) grouped
    bpA_std = axes2[0].boxplot(
        dataA_std,
        positions=positions - offset,
        widths=width,
        whis=(5, 95),
        showfliers=False,
    )
    bpA_vis = axes2[0].boxplot(
        dataA_vis,
        positions=positions + offset,
        widths=width,
        whis=(5, 95),
        showfliers=False,
    )
    axes2[0].set_title("A — Standard vs V(x0) (axis-aligned)")
    axes2[0].set_ylabel("Relative error")
    axes2[0].grid(True, axis="y", linestyle="--", alpha=0.6)

    # B: Standard vs V(x0) grouped
    bpB_std = axes2[1].boxplot(
        dataB_std,
        positions=positions - offset,
        widths=width,
        whis=(5, 95),
        showfliers=False,
    )
    bpB_vis = axes2[1].boxplot(
        dataB_vis,
        positions=positions + offset,
        widths=width,
        whis=(5, 95),
        showfliers=False,
    )
    axes2[1].set_title("B — Standard vs V(x0) (axis-aligned)")
    axes2[1].grid(True, axis="y", linestyle="--", alpha=0.6)

    # Shared x-ticks
    for ax in axes2:
        ax.set_xlabel("dim $V(x_0)$")
        ax.set_xticks(list(range(1, len(dims_sorted) + 1)), [str(k) for k in dims_sorted])

    # Compute global y-limits across all four series so A and B are strictly axis-aligned
    _all_vals = []
    for seq in (dataA_std, dataA_vis, dataB_std, dataB_vis):
        for arr in seq:
            if isinstance(arr, np.ndarray) and arr.size:
                _all_vals.append(np.asarray(arr, float))
    if _all_vals:
        ymin = 0.0
        ymax = float(np.max([np.nanmax(a) for a in _all_vals if a.size]))
        pad = 0.05 * (ymax if ymax > 0 else 1.0)
        for ax in axes2:
            ax.set_ylim(ymin, ymax + pad)

    # Legend (no custom colors/styles)
    legend_handles = [
        Line2D([0], [0], label="Standard basis"),
        Line2D([0], [0], label="V(x0)-basis (visible)"),
    ]
    axes2[1].legend(handles=legend_handles, loc="upper right", frameon=False)

    fig2.suptitle(f"{algo_name}: Axis-aligned Standard vs V(x0) errors", y=0.98)
    fig2.tight_layout()
    fig2.savefig(out_dir / f"vis_sweep_{algo_name}_std_vs_V_axis_aligned.png", dpi=150)
    plt.close(fig2)



        # ---------- NEW: unified (A-B mean) errors: standard vs V(x0)-basis ----------
    fig3, axes3 = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharex=True)

    # Left: standard basis unified
    uni_std_data = [np.asarray(by_dim_unified_std[k], float) for k in dims_sorted]
    axes3[0].boxplot(uni_std_data, whis=(5, 95), showfliers=False)
    axes3[0].set_title(f"{algo_name}: Unified error (A-B mean) — Standard basis")
    axes3[0].set_ylabel("Relative error (A-B mean)")
    axes3[0].grid(True, axis="y", linestyle="--", alpha=0.6)

    # Right: V(x0)-basis unified (visible block)
    uni_vis_data = [np.asarray(by_dim_unified_vis[k], float) for k in dims_sorted]
    axes3[1].boxplot(uni_vis_data, whis=(5, 95), showfliers=False)
    axes3[1].set_title(f"{algo_name}: Unified error (A-B mean) — V(x0)-basis")
    axes3[1].grid(True, axis="y", linestyle="--", alpha=0.6)

    # Shared x-ticks
    for ax in axes3:
        ax.set_xlabel("dim $V(x_0)$")
        ax.set_xticks(list(range(1, len(dims_sorted) + 1)), [str(k) for k in dims_sorted])

    fig3.tight_layout()
    fig3.savefig(out_dir / f"vis_sweep_{algo_name}_unified.png", dpi=150)
    plt.close(fig3)



# ---------- NEW: wrapper to run the full sweep for several algorithms ----------
def run_visibility_sweep_plots(
    cfg: Optional[EstimatorConsistencyConfig] = None,
    algos: Optional[Sequence[str]] = None,
    ensemble_size: int = 200,
    out_dir: Optional[pathlib.Path] = None,
) -> None:
    if cfg is None:
        cfg = EstimatorConsistencyConfig()
    rng = np.random.default_rng(cfg.seed)

    all_algos = list(_sweep_estimators().keys())
    use_algos = list(algos) if algos is not None else all_algos

    out_dir = out_dir or (cfg.save_dir / "vis_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in use_algos:
        _visibility_sweep_for_algo(cfg, name, rng, ensemble_size=ensemble_size, dims=None, out_dir=out_dir)



# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


@dataclass
class EstimatorConsistencyConfig(ExperimentConfig):
    """Configuration for the partially identifiable consistency study."""

    dims_invisible: Sequence[int] = (1,)
    """How many invisible directions ``d`` to test (``dim V = n - d``)."""

    n_systems: int = 8
    """Number of distinct systems drawn for each ``d``."""

    n_x0_per_system: int = 6
    """Number of initial conditions sampled per system."""

    n_signal_realizations: int = 4
    """Independent PRBS inputs simulated per ``(system, x0)`` pair."""

    pe_order_max: int = 16
    """Maximum order used when estimating PE of the generated inputs."""

    prbs_dwell_scale: float = 8.0
    """Scale used to adapt the PRBS dwell time to the visible dimension."""

    max_system_draws: int = 128
    """Maximum attempts allowed when searching for a system with target dim ``V``."""

    max_x0_draws: int = 256
    """Maximum attempts allowed when sampling an initial state with target dim ``V``."""

    det: bool = False   # NEW: toggle deterministic Krylov-based x0 construction
    """If True, use Krylov-based construction to hit target dim V(x0) instead of rejection sampling."""

    save_dir: pathlib.Path = field(default_factory=lambda: pathlib.Path("out_estimator_consistency"))

    def __post_init__(self) -> None:
        super().__post_init__()
        self.save_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _sample_unit_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(n)
    nrm = float(np.linalg.norm(v))
    if nrm == 0.0:
        return _sample_unit_sphere(n, rng)
    return v / nrm


def _visible_basis(Ad: np.ndarray, Bd: np.ndarray, x0: np.ndarray) -> np.ndarray:
    from ..metrics import build_visible_basis_dt
    return build_visible_basis_dt(Ad, Bd, x0, tol=1e-12)


def _orthonormal_basis(M: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Return an orthonormal basis for the column space of ``M``."""
    if M.size == 0:
        return np.zeros((M.shape[0], 0))
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    if s.size == 0:
        return np.zeros((M.shape[0], 0))
    cutoff = tol * max(M.shape)
    rank = int(np.sum(s > cutoff))
    return U[:, :rank]


def _reachable_basis(A: np.ndarray, B: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Compute an orthonormal basis for the reachable subspace of ``(A, B)``."""
    n = A.shape[0]
    zero = np.zeros(n)
    # ``unrestricted`` spans iterated images of both ``A`` and ``B``
    K = unified_generator(A, B, zero, mode="unrestricted")
    return _orthonormal_basis(K, tol=tol)


def _identifiability_summary(Ad: np.ndarray, Bd: np.ndarray, x0: np.ndarray) -> Dict[str, float]:
    """Return identifiability diagnostics derived directly from (Ad, Bd, x0)."""

    K = unified_generator(Ad, Bd, x0, mode="unrestricted")
    if K.size:
        svals = np.linalg.svd(K, compute_uv=False)
        sigma_min = float(svals[-1])
        cond = float(svals[0] / (svals[-1] + 1e-15))
    else:
        sigma_min = 0.0
        cond = np.inf

    Vbasis = _visible_basis(Ad, Bd, x0)
    dim_V = int(Vbasis.shape[1])

    eta = float(eta0(Ad, Bd, x0, rtol=1e-12))

    return {
        "sigma_min": sigma_min,
        "sigma_cond": cond,
        "dim_V": dim_V,
        "eta0": eta,
    }


def _adapted_basis(Vbasis: np.ndarray, n: int, tol: float = 1e-12) -> Tuple[np.ndarray, int]:
    """Return an orthonormal basis whose first ``k`` columns span ``Vbasis``."""
    if Vbasis.size == 0:
        return np.eye(n), 0
    U, s, _ = np.linalg.svd(Vbasis, full_matrices=True)
    k = int(np.sum(s > tol))
    return U, k


def _estimation_errors(
    Ahat: np.ndarray,
    Bhat: np.ndarray,
    Ad: np.ndarray,
    Bd: np.ndarray,
    Vbasis: np.ndarray,
) -> Dict[str, float]:
    nrmA = float(np.linalg.norm(Ad, ord="fro") + 1e-15)
    nrmB = float(np.linalg.norm(Bd, ord="fro") + 1e-15)
    errA = float(np.linalg.norm(Ahat - Ad, ord="fro") / nrmA)
    errB = float(np.linalg.norm(Bhat - Bd, ord="fro") / nrmB)

    dA_V, dB_V = projected_errors(Ahat, Bhat, Ad, Bd, Vbasis)
    scale_VA = float(np.linalg.norm(Vbasis.T @ Ad @ Vbasis, ord="fro") + 1e-15)
    scale_VB = float(np.linalg.norm(Vbasis.T @ Bd, ord="fro") + 1e-15)

    n = Ad.shape[0]
    Q, k = _adapted_basis(Vbasis, n)
    A_tilde = Q.T @ Ad @ Q
    Ahat_tilde = Q.T @ Ahat @ Q
    B_tilde = Q.T @ Bd
    Bhat_tilde = Q.T @ Bhat

    if k > 0:
        scale_vis_A = float(np.linalg.norm(A_tilde[:k, :k], ord="fro") + 1e-15)
        scale_vis_B = float(np.linalg.norm(B_tilde[:k, :], ord="fro") + 1e-15)
        errA_vis_block = float(np.linalg.norm(Ahat_tilde[:k, :k] - A_tilde[:k, :k], ord="fro") / scale_vis_A)
        errB_vis_block = float(np.linalg.norm(Bhat_tilde[:k, :] - B_tilde[:k, :], ord="fro") / scale_vis_B)
    else:
        errA_vis_block = float(np.linalg.norm(Ahat_tilde - A_tilde, ord="fro"))
        errB_vis_block = float(np.linalg.norm(Bhat_tilde - B_tilde, ord="fro"))

    if k < n:
        scale_dark_A = float(np.linalg.norm(A_tilde[k:, k:], ord="fro") + 1e-15)
        scale_dark_B = float(np.linalg.norm(B_tilde[k:, :], ord="fro") + 1e-15)
        errA_dark_block = float(np.linalg.norm(Ahat_tilde[k:, k:] - A_tilde[k:, k:], ord="fro") / scale_dark_A)
        errB_dark_block = float(np.linalg.norm(Bhat_tilde[k:, :] - B_tilde[k:, :], ord="fro") / scale_dark_B)
    else:
        errA_dark_block = 0.0
        errB_dark_block = 0.0

    return {
        "errA_rel": errA,
        "errB_rel": errB,
        "errA_V_rel": float(dA_V / scale_VA if scale_VA > 0 else dA_V),
        "errB_V_rel": float(dB_V / scale_VB if scale_VB > 0 else dB_V),
        "errA_vis_block_rel": errA_vis_block,
        "errB_vis_block_rel": errB_vis_block,
        "errA_dark_block_rel": errA_dark_block,
        "errB_dark_block_rel": errB_dark_block,
    }


def _estimate_algorithms() -> Mapping[str, Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    return {
        "dmdc_pinv": dmdc_pinv,
        "dmdc_tls": dmdc_tls,
        "moesp": moesp_fit,
    }


def _prepare_partially_identifiable_system(
    cfg: EstimatorConsistencyConfig,
    rng: np.random.Generator,
    dim_visible: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``(Ad, Bd)`` with ``dim V(x0)`` matching ``dim_visible`` for some ``x0``."""

    if not (0 < dim_visible <= cfg.n):
        raise ValueError(f"Visible dimension must lie in (0, n]; got {dim_visible} with n={cfg.n}.")

    base = getattr(cfg, "partial_base_ensemble", None) or cfg.ensemble
    if base == "A_stbl_B_ctrb":
        base = "stable"

    attempts = 0
    while attempts < cfg.max_system_draws:
        attempts += 1
        A, B, _ = draw_with_ctrb_rank(
            cfg.n,
            cfg.m,
            dim_visible,
            rng,
            ensemble_type=base,
            embed_random_basis=True,
        )
        Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)
        Rbasis = _reachable_basis(Ad, Bd, tol=1e-12)
        if Rbasis.shape[1] != dim_visible:
            continue
        return Ad, Bd, Rbasis
    
    raise RuntimeError(
        f"Unable to synthesise a system with reachable dimension {dim_visible} after {cfg.max_system_draws} attempts."
    )


# ---------- NEW: deterministic Krylov-based x0 constructor ----------
def _construct_x0_with_dimV(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Rbasis: np.ndarray,
    dim_visible: int,
    rng: np.random.Generator,
    tol: float = 1e-12,
    max_tries: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic/near-deterministic construction of x0 with target k=dim_visible.
    Works inside the reachable subspace basis Rbasis (n×r). Builds a Krylov ladder
    for Ar = R^T Ad R and picks the k-th new direction.
    """
    r = int(Rbasis.shape[1])
    k = int(dim_visible)
    if not (1 <= k <= r):
        raise ValueError(f"Requested dim V(x0) k={k} must lie in [1, r={r}].")

    Ar = Rbasis.T @ Ad @ Rbasis  # r×r
    for attempt in range(max_tries):
        # seed
        y0 = rng.standard_normal(r)
        nrm = float(np.linalg.norm(y0))
        if nrm <= tol:
            continue
        y0 /= nrm

        # build Krylov columns: [y0, Ar y0, ..., Ar^{k-1} y0]
        Kcols = []
        v = y0
        for _ in range(k):
            Kcols.append(v)
            v = Ar @ v
        K = np.column_stack(Kcols)  # r×k

        # orthonormalize K to get an ONB of K_k
        Q, _ = np.linalg.qr(K, mode="reduced")  # r×q, q<=k
        if Q.shape[1] < k:
            # unlucky seed produced deficient Krylov growth; retry
            continue

        # pick the k-th new direction in K_k \ K_{k-1}
        y = Q[:, k - 1]
        x0 = Rbasis @ y
        x0 /= float(np.linalg.norm(x0) + 1e-15)

        Vbasis = _visible_basis(Ad, Bd, x0)
        if Vbasis.shape[1] == k:
            return x0, Vbasis

        # micro-nudge inside K_k away from K_{k-1}, then retry verify
        if k >= 2:
            y = Q[:, k - 1] + 1e-3 * Q[:, k - 2]
        else:
            y = Q[:, 0] + 1e-3 * rng.standard_normal(r)
        y /= float(np.linalg.norm(y) + 1e-15)
        x0 = Rbasis @ y
        x0 /= float(np.linalg.norm(x0) + 1e-15)
        Vbasis = _visible_basis(Ad, Bd, x0)
        if Vbasis.shape[1] == k:
            return x0, Vbasis

    # Fallback: if repeated seeds fail (rare), use the original rejection sampler
    return _sample_visible_initial_state(Ad, Bd, Rbasis, k, rng, max_attempts=256)


def _sample_visible_initial_state_det(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Rbasis: np.ndarray,
    dim_visible: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Thin wrapper to call the deterministic constructor."""
    return _construct_x0_with_dimV(Ad, Bd, Rbasis, dim_visible, rng)


# ---------- Original rejection-sampling path (unchanged) ----------
def _sample_visible_initial_state(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Rbasis: np.ndarray,
    dim_visible: int,
    rng: np.random.Generator,
    max_attempts: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample ``x0`` such that ``dim V(x0)`` equals ``dim_visible``."""
    n = Ad.shape[0]
    for _ in range(max_attempts):
        if dim_visible >= n:
            x0 = _sample_unit_sphere(n, rng)
        else:
            coeff = rng.standard_normal(dim_visible)
            x0 = Rbasis @ coeff
            nrm = float(np.linalg.norm(x0))
            if nrm <= 1e-12:
                continue
            x0 = x0 / nrm
        Vbasis = _visible_basis(Ad, Bd, x0)
        if Vbasis.shape[1] == dim_visible:
            return x0, Vbasis
    raise RuntimeError(
        f"Failed to draw x0 with dim V(x0)={dim_visible} after {max_attempts} attempts."
    )


def _draw_prbs(cfg: EstimatorConsistencyConfig, rng: np.random.Generator, dim_visible: int) -> Tuple[np.ndarray, int]:
    dwell = max(1, int(round(cfg.prbs_dwell_scale / max(1, dim_visible))))
    U = prbs_dt(cfg.T, cfg.m, scale=cfg.u_scale, dwell=dwell, rng=rng)
    return U, dwell


def _partially_identifiable_trials(cfg: EstimatorConsistencyConfig, rng: np.random.Generator) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    algs = _estimate_algorithms()

    for hidden_dim in cfg.dims_invisible:
        dim_visible = cfg.n - int(hidden_dim)
        if dim_visible <= 0:
            raise ValueError(
                f"Invisible dimension {hidden_dim} is incompatible with n={cfg.n}."
            )
        for sys_idx in range(cfg.n_systems):
            Ad, Bd, Rbasis = _prepare_partially_identifiable_system(cfg, rng, dim_visible)

            for x_idx in range(cfg.n_x0_per_system):
                # --- choose track: deterministic vs. rejection ---
                if getattr(cfg, "det", False):  # NEW
                    x0, Vbasis = _sample_visible_initial_state_det(
                        Ad, Bd, Rbasis, dim_visible, rng
                    )
                else:
                    x0, Vbasis = _sample_visible_initial_state(
                        Ad,
                        Bd,
                        Rbasis,
                        dim_visible,
                        rng,
                        cfg.max_x0_draws,
                    )

                ident = _identifiability_summary(Ad, Bd, x0)

                for signal_idx in range(cfg.n_signal_realizations):
                    U, dwell = _draw_prbs(cfg, rng, dim_visible)
                    pe_est = estimate_pe_order(
                        U,
                        s_max=min(cfg.pe_order_max, cfg.T // 2),
                    )

                    X = simulate_dt(x0, Ad, Bd, U, noise_std=cfg.noise_std, rng=rng)
                    X0, X1 = X[:, :-1], X[:, 1:]
                    U_cm = U.T
                    zstats = regressor_stats(X0, U_cm, rtol_rank=1e-12)

                    for name, estimator in algs.items():
                        try:
                            Ahat, Bhat = estimator(X0, X1, U_cm)
                        except Exception as exc:  # pragma: no cover - defensive guard
                            rec = {
                                "hidden_dim": int(hidden_dim),
                                "dim_visible": float(dim_visible),
                                "system": sys_idx,
                                "x0_id": x_idx,
                                "signal_id": signal_idx,
                                "algo": name,
                                "failed": 1,
                                "failure_msg": str(exc),
                                "dwell": dwell,
                                "pe_estimated": pe_est,
                                "target_pe": dim_visible,
                                **ident,
                            }
                            rec.update({
                                k: np.nan
                                for k in (
                                    "errA_rel",
                                    "errB_rel",
                                    "errA_V_rel",
                                    "errB_V_rel",
                                    "errA_vis_block_rel",
                                    "errB_vis_block_rel",
                                    "errA_dark_block_rel",
                                    "errB_dark_block_rel",
                                )
                            })
                            rec.update({k: np.nan for k in ("eqv_ok", "eqv_dA_V", "eqv_dB", "eqv_leak")})
                            rec.update({k: np.nan for k in zstats})
                            records.append(rec)
                            continue

                        errs = _estimation_errors(Ahat, Bhat, Ad, Bd, Vbasis)
                        ok_eqv, info_eqv = same_equiv_class_dt_rel(
                            Ad,
                            Bd,
                            Ahat,
                            Bhat,
                            x0,
                            rtol_eq=getattr(cfg, "rtol_eq_rel", 1e-2),
                            rtol_rank=1e-12,
                            use_leak=True,
                        )

                        rec = {
                            "hidden_dim": int(hidden_dim),
                            "dim_visible": float(dim_visible),
                            "system": sys_idx,
                            "x0_id": x_idx,
                            "signal_id": signal_idx,
                            "algo": name,
                            "failed": 0,
                            "eqv_ok": int(ok_eqv),
                            "dwell": dwell,
                            "pe_estimated": pe_est,
                            "target_pe": dim_visible,
                            **ident,
                            **errs,
                            **zstats,
                            "eqv_dim_V": info_eqv.get("dim_V", np.nan),
                            "eqv_dA_V": info_eqv.get("dA_V", np.nan),
                            "eqv_dB": info_eqv.get("dB", np.nan),
                            "eqv_leak": info_eqv.get("leak", np.nan),
                        }
                        records.append(rec)

    return pd.DataFrame.from_records(records)


def _summaries_from_trials(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    metrics = [
        "errA_rel",
        "errB_rel",
        "errA_V_rel",
        "errB_V_rel",
        "errA_vis_block_rel",
        "errB_vis_block_rel",
        "errA_dark_block_rel",
        "errB_dark_block_rel",
    ]
    grouped = (
        df.loc[df["failed"] == 0]
        .groupby(["hidden_dim", "dim_visible", "algo"])[metrics]
        .agg(["mean", "median"])
        .reset_index()
    )

    flat_columns: List[str] = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            name = "_".join(str(part) for part in col if str(part))
            flat_columns.append(name)
        else:
            flat_columns.append(str(col))
    grouped.columns = flat_columns
    return grouped


# ---------------------------------------------------------------------------
# Public experiment entry point
# ---------------------------------------------------------------------------


def run_experiment(cfg: Optional[EstimatorConsistencyConfig] = None) -> Dict[str, pd.DataFrame]:
    """Run the estimator consistency experiment on partially identifiable systems."""

    if cfg is None:
        cfg = EstimatorConsistencyConfig()

    rng = np.random.default_rng(cfg.seed)

    trials = _partially_identifiable_trials(cfg, rng)
    summary = _summaries_from_trials(trials)

    outputs = {
        "trials": trials,
        "summary": summary,
    }

    for name, df in outputs.items():
        df.to_csv(cfg.save_dir / f"{name}.csv", index=False)

    return outputs


def main() -> None:
    # NEW: simple CLI to toggle deterministic x0 construction
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", action="store_true",
                    help="Use deterministic Krylov-based x0 construction (target dim V(x0)).")
    ap.add_argument("--vis", action="store_true",
                    help="Run visibility-dimension sweep and save figures per algorithm.")
    ap.add_argument("--vis-ntrials", type=int, default=200,
                    help="Ensemble size per visible dimension (default: 200).")
    ap.add_argument("--vis-outdir", type=str, default=None,
                    help="Output directory for sweep figures (defaults to escons_alg_sweep).")
    ap.add_argument("--algos", type=str, default="SINDy,MOESP,DMDc,NODE",
                    help="Comma-separated list of algorithms to include.")

    args, _ = ap.parse_known_args()
    cfg = EstimatorConsistencyConfig()
    cfg.det = bool(args.det)

    if args.vis:
        out_dir = pathlib.Path(args.vis_outdir) if args.vis_outdir else None
        algos = [s.strip() for s in args.algos.split(",") if s.strip()]
        run_visibility_sweep_plots(cfg, algos=algos, ensemble_size=int(args.vis_ntrials), out_dir=out_dir)
    else:
        # default behavior: run the original experiment
        run_experiment(cfg)



if __name__ == "__main__":
    main()
