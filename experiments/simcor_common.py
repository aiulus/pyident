"""Common utilities for the simplified correlation experiments.

This module centralises the helper functionality used by
``sim_simpcore`` and ``sim_stratcore`` so that we can re-use the same
validation and metric computations across both entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from ..config import ExperimentConfig
from ..metrics import (
    build_visible_basis_dt,
    eta0,
    krylov_smin_norm,
    left_eigvec_overlap,
    pbh_margin_structured,
    unified_generator,
)
from ..signals import estimate_pe_order
from ..simulation import prbs


@dataclass
class CoreExperimentConfig(ExperimentConfig):
    """Base configuration shared by the simplified correlation scripts."""

    ens_size: int = 24
    x0_samples: int = 8
    x0_amp: float = 1.0
    outdir: Path = Path("out_simcore")

    def __post_init__(self) -> None:  # type: ignore[override]
        super().__post_init__()
        if self.ens_size <= 0:
            raise ValueError("ens_size must be positive.")
        if self.x0_samples <= 0:
            raise ValueError("x0_samples must be positive.")
        if self.x0_amp <= 0.0:
            raise ValueError("x0_amp must be positive.")
        if isinstance(self.outdir, str):
            self.outdir = Path(self.outdir)


def sample_unit_sphere(
    n: int, rng: np.random.Generator, radius: float = 1.0
) -> np.ndarray:
    """Draw a random point from a sphere of ``R^n`` with the given ``radius``."""

    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    v = rng.standard_normal(n)
    nrm = float(np.linalg.norm(v))
    if nrm == 0.0:
        return sample_unit_sphere(n, rng, radius=radius)
    return radius * (v / nrm)


def compute_identifiability_metrics(
    Ad: np.ndarray, Bd: np.ndarray, x0: np.ndarray
) -> Dict[str, float]:
    """Discrete-time identifiability proxies used throughout the scripts."""

    generator = unified_generator(Ad, Bd, x0, mode="unrestricted")
    if generator.size:
        svals = np.linalg.svd(generator, compute_uv=False)
        sigma_min = float(svals[-1])
    else:
        sigma_min = 0.0
    eta = float(eta0(Ad, Bd, x0, rtol=1e-12))
    Vbasis = build_visible_basis_dt(Ad, Bd, x0, tol=1e-10)
    dim_visible = int(Vbasis.shape[1])
    return {
        "sigma_min": sigma_min,
        "eta0": eta,
        "dim_visible": dim_visible,
    }


def compute_core_metrics(A: np.ndarray, B: np.ndarray, x0: np.ndarray) -> Dict[str, float]:
    """Continuous-time metrics carried over from ``sim1``."""

    pbh_struct = float(pbh_margin_structured(A, B, x0))
    krylov_smin = float(krylov_smin_norm(A, B, x0))
    Xaug = np.concatenate([x0.reshape(-1, 1), B], axis=1)
    mu = left_eigvec_overlap(A, Xaug)
    mu_min = float(np.min(mu)) if mu.size else 0.0
    return {
        "pbh_struct": pbh_struct,
        "krylov_smin": krylov_smin,
        "mu_min": mu_min,
    }


def relative_errors(
    Ahat: np.ndarray, Bhat: np.ndarray, Ad: np.ndarray, Bd: np.ndarray
) -> Dict[str, float]:
    """Relative Frobenius errors against the discrete-time ground truth."""

    errA = float(np.linalg.norm(Ahat - Ad, ord="fro"))
    errB = float(np.linalg.norm(Bhat - Bd, ord="fro"))
    nrmA = float(np.linalg.norm(Ad, ord="fro") + 1e-15)
    nrmB = float(np.linalg.norm(Bd, ord="fro") + 1e-15)
    relA = errA / nrmA
    relB = errB / nrmB
    rel_mean = 0.5 * (relA + relB)
    return {
        "errA_rel": relA,
        "errB_rel": relB,
        "err_mean_rel": rel_mean,
    }


def target_pe_order(cfg: ExperimentConfig) -> int:
    """Desired persistent-excitation order for the PRBS input."""

    return 2 * (cfg.n + cfg.m)


def minimum_horizon(cfg: ExperimentConfig, order: int) -> int:
    """Minimum trajectory horizon for an order-``r`` Hankel matrix."""

    T_needed = cfg.m * order + order - 1
    return max(cfg.T, T_needed)


def prbs_with_order(
    cfg: ExperimentConfig, rng: np.random.Generator
) -> Tuple[np.ndarray, int]:
    """Generate a PRBS input and report the achieved PE order."""

    target = target_pe_order(cfg)
    T = minimum_horizon(cfg, target)
    best_U: np.ndarray | None = None
    best_order = -1
    for _ in range(25):
        U = prbs(T, cfg.m, scale=cfg.u_scale, dwell=cfg.dwell, rng=rng)
        order_est = int(estimate_pe_order(U, s_max=min(target, T // 2)))
        if order_est >= target:
            return U, order_est
        if order_est > best_order:
            best_order = order_est
            best_U = U
    if best_U is None:
        best_U = prbs(T, cfg.m, scale=cfg.u_scale, dwell=cfg.dwell, rng=rng)
        best_order = int(estimate_pe_order(best_U, s_max=min(target, T // 2)))
    return best_U, best_order


__all__ = [
    "CoreExperimentConfig",
    "compute_core_metrics",
    "compute_identifiability_metrics",
    "minimum_horizon",
    "prbs_with_order",
    "relative_errors",
    "sample_unit_sphere",
    "target_pe_order",
]