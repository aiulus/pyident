from __future__ import annotations
import numpy as np
import math
from typing import Tuple, Optional
from .loggers.tolerances import TolerancePolicy

# ---------------------------------------------------------------------
# Input generators + PE tests
# ---------------------------------------------------------------------


def prbs(
    T: int,
    m: int,
    rng: np.random.Generator,
    levels: Tuple[float, float] = (-1.0, 1.0),
    period: int = 31,
) -> np.ndarray:
    """PRBS: repeat a random +/- pattern of length `period` to fill horizon T."""
    if period <= 0:
        raise ValueError("period must be positive.")
    base = rng.choice(levels, size=(period, m))
    reps = int(np.ceil(T / period))
    u = np.tile(base, (reps, 1))[:T]
    return u


def multisine(
    T: int,
    m: int,
    rng: np.random.Generator,
    k_lines: int = 8,
) -> np.ndarray:
    """Sum of sines with random phases on k_lines distinct bins; per-channel normalized."""
    t = np.arange(T)
    u = np.zeros((T, m))
    # guard small T
    max_bin = max(2, T // 8)
    for j in range(m):
        k = min(k_lines, max_bin - 1)
        freqs = rng.choice(np.arange(1, max_bin), size=k, replace=False)
        phases = rng.uniform(0, 2 * np.pi, size=k)
        w = np.zeros(T)
        for f, ph in zip(freqs, phases):
            w += np.sin(2 * np.pi * f * t / T + ph)
        denom = np.max(np.abs(w)) if np.max(np.abs(w)) > 0 else 1.0
        u[:, j] = w / denom
    return u


# ------------------ PE (block-Hankel) --------------------------------

def hankel_blocks(u: np.ndarray, s: int) -> np.ndarray:
    """Build block Hankel of depth s for multi-input u (T×m) -> (s*m)×cols."""
    T, m = u.shape
    cols = T - 2 * s + 1
    if s <= 0:
        raise ValueError("s must be positive.")
    if cols <= 0:
        raise ValueError("Not enough samples for requested Hankel depth.")
    H = [u[i:i + cols].T for i in range(s)]  # each (m×cols)
    return np.vstack(H)  # (s*m × cols)


def estimate_pe_order_block(u: np.ndarray, s_max: int, tol: float = 1e-8) -> int:
    # Preserve API; route through TolerancePolicy for consistency
    pol = TolerancePolicy(svd_atol=tol)  # respect old param while centralizing
    m = u.shape[1]
    s_max_eff = min(s_max, u.shape[0] // 2)
    for s in range(s_max_eff, 0, -1):
        H = hankel_blocks(u, s)
        # SVD-based rank
        svals = np.linalg.svd(H, compute_uv=False)
        r = pol.rank_from_singulars(svals)
        if r == H.shape[0]:  # full row rank
            return s
    return 0


def estimate_pe_order_block_old(u: np.ndarray, s_max: int, tol: float = 1e-8) -> int:
    """Largest s such that block-Hankel H_s has full row rank (up to tol)."""
    T, m = u.shape
    s_max_eff = min(s_max, max(1, T // 2))
    for s in range(s_max_eff, 0, -1):
        H = hankel_blocks(u, s)
        r = np.linalg.matrix_rank(H, tol=tol)
        if r == H.shape[0]:
            return s
    return 0

def estimate_pe_order(u: np.ndarray, s_max: int, tol: float = 1e-8) -> int:
    # kept for backward compatibility
    return estimate_pe_order_block(u, s_max=s_max, tol=tol)

# --- add these helpers near the top (after imports) ---
def _estimate_block_pe(U, s_max, tol=None):
    # Prefer tolerant signature if available; otherwise fall back.
    try:
        if tol is not None:
            return int(estimate_pe_order(U, s_max=s_max, tol=tol))
    except TypeError:
        pass
    return int(estimate_pe_order(U, s_max=s_max))

def _estimate_moment_pe(U, r_max, dt, tol=None):
    try:
        if tol is not None:
            return int(estimate_moment_pe_order(U, r_max=r_max, dt=dt, tol=tol))
    except TypeError:
        pass
    return int(estimate_moment_pe_order(U, r_max=r_max, dt=dt))



# ------------------ Moment-PE ------------------------

def _moment_map_old(u: np.ndarray, r: int, t0: Optional[int], dt: float = 1.0) -> np.ndarray:
    """Discrete approximation to psi_k(u) = int_0^{t0} ((t0-s)^k/k!) u(s) ds.

    Disclaimer: this is a *Riemann-sum* discretization with step `dt`.
    It approximates the continuous-time functional used in the manuscript.
    """
    T, m = u.shape
    if r <= 0:
        raise ValueError("r must be positive.")
    if t0 is None:
        t0 = T - 1
    if not (0 <= t0 < T):
        raise ValueError(f"t0 must be in [0, T-1], got {t0} for T={T}.")

    # times 0..t0 (inclusive)
    s = np.arange(t0 + 1)
    out = np.zeros((m * r,), dtype=float)
    idx = 0
    for k in range(r):
        # weights: ((t0 - s)^k / k!)
        w = ((t0 - s) ** k) / (np.math.factorial(k))
        # channel-wise integration
        psi_k = (u[: t0 + 1, :] * w[:, None]).sum(axis=0) * dt
        out[idx: idx + m] = psi_k
        idx += m
    return out


def _moment_map(u: np.ndarray, r: int, t0: Optional[int] = None, dt: float = 1.0) -> np.ndarray:
    T, m = u.shape                    # we use time-major: (T, m)
    if r <= 0:
        raise ValueError("r must be positive.")
    if t0 is None:
        t0 = T - 1
    if not (0 <= t0 < T):
        raise ValueError(f"t0 must be in [0, T-1], got {t0} for T={T}.")
    s = np.arange(t0 + 1, dtype=float)
    w = np.empty((r, t0 + 1), dtype=float)
    for k in range(r):
        denom = math.factorial(k) if k <= 20 else math.exp(math.lgamma(k+1))
        w[k, :] = ((t0 - s) ** k) / denom
    M = (w * dt) @ u[:t0+1, :]        # (r × m)
    return M.reshape(-1, order="C")   # stack ψ_k blocks of size m

def estimate_moment_pe_order(u: np.ndarray, r_max: int, t0: Optional[int] = None,
                             dt: float = 1.0, tol: float = 1e-10) -> int:
    T, _ = u.shape
    r_max_eff = max(1, min(int(r_max), T))
    best, prev = 0, 0.0
    for r in range(1, r_max_eff + 1):
        nv = float(np.linalg.norm(_moment_map(u, r, t0=t0, dt=dt)))
        if nv > tol and nv >= prev * (1 - 1e-12):
            best, prev = r, nv
        else:
            break
    return best

# ------------------ Pointwise restriction ----------------------------

def restrict_pointwise(u: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Project inputs onto span(W) (columns of W form admissible directions).

    u: (T×m), W: (m×q). Returns u_proj ∈ span(W).

    Disclaimer: orthogonal projection uses P = W W^+.
    If W is ill-conditioned, results depend on pseudoinverse regularization.
    """
    if W.ndim != 2:
        raise ValueError("W must be a 2D array (m×q).")
    m = u.shape[1]
    if W.shape[0] != m:
        raise ValueError(f"W first dimension must equal m={m}, got {W.shape}.")
    Winv = np.linalg.pinv(W)
    P = W @ Winv  # (m×m) projector onto span(W)
    return u @ P  # row-wise projection of each time sample


