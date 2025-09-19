from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
from .tolerances import TolerancePolicy

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

def _moment_map(u: np.ndarray, r: int, t_scale: float = 1.0) -> np.ndarray:
    """
    Discrete-time 'moments' of u with polynomial basis phi_k(t) = (t/t_scale)^k.
    Returns M in R^{r x m} with M[k,:] = sum_t phi_k(t) * u[t,:]
    """
    T, m = u.shape
    t = np.arange(T, dtype=float) / float(t_scale if t_scale != 0 else 1.0)
    Phi = np.stack([t**k for k in range(r)], axis=1)  # (T, r)
    # least-squares moment map (can also do direct sum with weights)
    M = Phi.T @ u   # (r, m)
    return M


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


def estimate_moment_pe_order(
    u: np.ndarray,
    r_max: int,
    t0: Optional[int] = None,
    dt: float = 1.0,
    tol: float = 1e-10,
) -> int:
    """Return the largest r such that the moment map has full range on the span
    of provided inputs (single input sequence -> rank test over stacked psi_k).

    Implementation detail: with a *single* input realization, we test if the stacked
    vector [psi_0; ...; psi_{r-1}] is non-degenerate up to `tol`. For multiple inputs,
    stack across realizations before testing rank.
    """
    T, m = u.shape
    r_max_eff = max(1, min(r_max, T))
    # With a single realization, “full range” reduces to growing, non-degenerate stack.
    # We check the norm grows in a numerically stable way; for more realizations,
    # users should pass a (R×T×m) array and compute a proper rank.
    best = 0
    for r in range(1, r_max_eff + 1):
        v = _moment_map(u, r, t0=t0, dt=dt)
        if np.linalg.norm(v) > tol:
            best = r
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


