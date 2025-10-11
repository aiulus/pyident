"""Utilities for drawing systems and initial states with prescribed visible dimension."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..ensembles import draw_with_ctrb_rank
from ..metrics import build_visible_basis_dt, cont2discrete_zoh, unified_generator


def _sample_unit_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random unit vector in :math:`\mathbb{R}^n`."""

    v = rng.standard_normal(n)
    nrm = float(np.linalg.norm(v))
    if nrm == 0.0:
        return _sample_unit_sphere(n, rng)
    return v / nrm


def visible_basis(Ad: np.ndarray, Bd: np.ndarray, x0: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Return an orthonormal basis for the visible subspace of ``(Ad, Bd, x0)``."""

    return build_visible_basis_dt(Ad, Bd, x0, tol=tol)


def _orthonormal_basis(M: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    if M.size == 0:
        return np.zeros((M.shape[0], 0))
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    if s.size == 0:
        return np.zeros((M.shape[0], 0))
    cutoff = tol * max(M.shape)
    rank = int(np.sum(s > cutoff))
    return U[:, :rank]


def reachable_basis(Ad: np.ndarray, Bd: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Compute an orthonormal basis for the reachable subspace of ``(Ad, Bd)``."""

    n = Ad.shape[0]
    zero = np.zeros(n)
    K = unified_generator(Ad, Bd, zero, mode="unrestricted")
    return _orthonormal_basis(K, tol=tol)


@dataclass
class VisibleDrawConfig:
    n: int
    m: int
    dt: float
    dim_visible: int
    ensemble: str = "stable"
    max_system_attempts: int = 128
    max_x0_attempts: int = 256
    tol: float = 1e-12
    deterministic_x0: bool = False
    force_hurwitz: bool = False
    stability_margin: float = 0.05


def _ensure_hurwitz(A: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """Return a Hurwitz-stable copy of ``A`` with margin ``margin``."""

    if margin < 0:
        raise ValueError("stability margin must be non-negative")

    if A.size == 0:
        return A

    lam = np.linalg.eigvals(A)
    max_real = float(np.max(np.real(lam))) if lam.size else float("-inf")
    shift = max(0.0, max_real + margin)
    if shift == 0.0:
        return A
    return A - shift * np.eye(A.shape[0], dtype=A.dtype)


def prepare_system_with_visible_dim(
    cfg: VisibleDrawConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``(A, B, Ad, Bd, Rbasis)`` with reachable dimension ``cfg.dim_visible``."""

    attempts = 0
    while attempts < cfg.max_system_attempts:
        attempts += 1
        A, B, _ = draw_with_ctrb_rank(
            n=cfg.n,
            m=cfg.m,
            r=cfg.dim_visible,
            rng=rng,
            ensemble_type=cfg.ensemble,
            embed_random_basis=True,
        )
        
        if cfg.force_hurwitz:
            A = _ensure_hurwitz(A, margin=cfg.stability_margin)

        Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)
        Rbasis = reachable_basis(Ad, Bd, tol=cfg.tol)
        if Rbasis.shape[1] != cfg.dim_visible:
            continue
        return A, B, Ad, Bd, Rbasis

    raise RuntimeError(
        f"Unable to synthesise a system with reachable dimension {cfg.dim_visible} "
        f"after {cfg.max_system_attempts} attempts."
    )


def construct_x0_with_dim_visible(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Rbasis: np.ndarray,
    dim_visible: int,
    rng: np.random.Generator,
    *,
    tol: float = 1e-12,
    max_tries: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministically construct ``x0`` with ``dim V(x0) = dim_visible``."""

    r = int(Rbasis.shape[1])
    k = int(dim_visible)
    if not (1 <= k <= r):
        raise ValueError(f"Requested visible dimension {k} must lie in [1, {r}].")

    Ar = Rbasis.T @ Ad @ Rbasis
    for attempt in range(max_tries):
        y0 = rng.standard_normal(r)
        nrm = float(np.linalg.norm(y0))
        if nrm <= tol:
            continue
        y0 /= nrm

        Kcols = []
        v = y0
        for _ in range(k):
            Kcols.append(v)
            v = Ar @ v
        K = np.column_stack(Kcols)

        Q, _ = np.linalg.qr(K, mode="reduced")
        if Q.shape[1] < k:
            continue

        y = Q[:, k - 1]
        x0 = Rbasis @ y
        x0 /= float(np.linalg.norm(x0) + 1e-15)

        Vbasis = visible_basis(Ad, Bd, x0, tol=tol)
        if Vbasis.shape[1] == k:
            return x0, Vbasis

        if k >= 2:
            y = Q[:, k - 1] + 1e-3 * Q[:, k - 2]
        else:
            y = Q[:, 0] + 1e-3 * rng.standard_normal(r)
        y /= float(np.linalg.norm(y) + 1e-15)
        x0 = Rbasis @ y
        x0 /= float(np.linalg.norm(x0) + 1e-15)
        Vbasis = visible_basis(Ad, Bd, x0, tol=tol)
        if Vbasis.shape[1] == k:
            return x0, Vbasis

    return sample_visible_initial_state(
        Ad,
        Bd,
        Rbasis,
        dim_visible,
        rng,
        max_attempts=256,
        tol=tol,
    )


def sample_visible_initial_state(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Rbasis: np.ndarray,
    dim_visible: int,
    rng: np.random.Generator,
    *,
    max_attempts: int = 256,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample ``x0`` such that ``dim V(x0)`` equals ``dim_visible``."""

    n = Ad.shape[0]
    r = int(Rbasis.shape[1])
    for _ in range(max_attempts):
        if dim_visible >= n:
            x0 = _sample_unit_sphere(n, rng)
        else:
            if r == 0:
                raise RuntimeError("Reachable basis has zero columns; cannot sample x0.")

            if dim_visible >= r:
                y = _sample_unit_sphere(r, rng)
            else:
                subspace = rng.standard_normal((r, dim_visible))
                Q, _ = np.linalg.qr(subspace, mode="reduced")
                coeff = rng.standard_normal(dim_visible)
                y = Q @ coeff

            x0 = Rbasis @ y
            nrm = float(np.linalg.norm(x0))
            if nrm <= tol:
                continue
            x0 = x0 / nrm
        Vbasis = visible_basis(Ad, Bd, x0, tol=tol)
        if Vbasis.shape[1] == dim_visible:
            return x0, Vbasis

    raise RuntimeError(
        f"Failed to draw x0 with dim V(x0)={dim_visible} after {max_attempts} attempts."
    )


def sample_visible_initial_state_det(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Rbasis: np.ndarray,
    dim_visible: int,
    rng: np.random.Generator,
    *,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper that uses the deterministic Krylov-based constructor."""

    return construct_x0_with_dim_visible(
        Ad,
        Bd,
        Rbasis,
        dim_visible,
        rng,
        tol=tol,
    )


def draw_system_state_with_visible_dim(
    cfg: VisibleDrawConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``(A, B, Ad, Bd, x0, Vbasis)`` hitting the requested visible dimension."""

    A, B, Ad, Bd, Rbasis = prepare_system_with_visible_dim(cfg, rng)
    if cfg.deterministic_x0:
        x0, Vbasis = construct_x0_with_dim_visible(
            Ad,
            Bd,
            Rbasis,
            cfg.dim_visible,
            rng,
            tol=cfg.tol,
        )
    else:
        x0, Vbasis = sample_visible_initial_state(
            Ad,
            Bd,
            Rbasis,
            cfg.dim_visible,
            rng,
            max_attempts=cfg.max_x0_attempts,
            tol=cfg.tol,
        )
    return A, B, Ad, Bd, x0, Vbasis