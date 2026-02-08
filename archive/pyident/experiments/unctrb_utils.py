from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..pe_sig import PRBSSpec, MultisineSpec, generate_pe_signal


def spectral_radius(A: np.ndarray) -> float:
    if A.size == 0:
        return 0.0
    vals = np.linalg.eigvals(A)
    return float(np.max(np.abs(vals))) if vals.size else 0.0


def b_norms(B: np.ndarray) -> Tuple[float, float, float]:
    if B.size == 0:
        return 0.0, 0.0, 0.0
    fro = float(np.linalg.norm(B, ord="fro"))
    row_norms = np.linalg.norm(B, axis=1) if B.ndim == 2 else np.array([])
    col_norms = np.linalg.norm(B, axis=0) if B.ndim == 2 else np.array([])
    min_row = float(row_norms.min()) if row_norms.size else 0.0
    min_col = float(col_norms.min()) if col_norms.size else 0.0
    return fro, min_row, min_col


def condition_number(
    Z: np.ndarray,
    *,
    tol: float = 1e-12,
    normalize_columns: bool = True,
) -> float:
    Z = np.asarray(Z, dtype=float)
    if Z.size == 0:
        return float("inf")
    if normalize_columns:
        coln = np.linalg.norm(Z, axis=0)
        coln[coln == 0.0] = 1.0
        Z = Z / coln
    s = np.linalg.svd(Z, compute_uv=False)
    if s.size == 0:
        return float("inf")
    smin = float(s[-1])
    smax = float(s[0])
    if smin <= tol:
        return float("inf")
    return smax / smin


def pe_order_from_nmax(n_max: int) -> int:
    return 2 * int(n_max) + 1


def min_T_for_block_pe(pe_order: int, m: int) -> int:
    return int(pe_order) * (int(m) + 1) - 1


def generate_pe_input(
    *,
    T: int,
    m: int,
    dt: float,
    dwell: int,
    rng: np.random.Generator,
    pe_order: int,
    family: str = "prbs",
    pe_method: str = "block",
    pe_tol: float = 1e-8,
    max_tries: int = 128,
    scale: float = 1.0,
    prbs: Optional[PRBSSpec] = None,
    multisine: Optional[MultisineSpec] = None,
) -> tuple[np.ndarray, int]:
    if pe_method == "block":
        min_T = min_T_for_block_pe(pe_order, m)
        if T < min_T:
            raise ValueError(
                f"T={T} is too short for block-PE order {pe_order} with m={m}; "
                f"need at least T >= {min_T}."
            )

    if prbs is None:
        prbs = PRBSSpec(clock=float(dt) * int(dwell))
    if multisine is None:
        multisine = MultisineSpec()

    bundle = generate_pe_signal(
        family=family,
        T=T,
        m=m,
        dt=dt,
        pe_order=pe_order,
        rng=rng,
        prbs=prbs,
        multisine=multisine,
        ensure_pe=True,
        pe_method=pe_method,
        pe_tol=pe_tol,
        max_tries=max_tries,
    )

    U = np.asarray(bundle.u, dtype=float) * float(scale)
    pe_est = int(bundle.pe_order_est) if bundle.pe_order_est is not None else 0
    return U, pe_est

