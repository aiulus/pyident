from __future__ import annotations
import numpy as np
import numpy.linalg as npl
from typing import Tuple

def prbs(T: int, m: int, rng: np.random.Generator, levels: Tuple[float, float]=(-1.0, 1.0), period: int = 31) -> np.ndarray:
    """PRBS signal: repeat a random +/- pattern of length 'period' to fill horizon T."""
    base = rng.choice(levels, size=(period, m))
    reps = int(np.ceil(T / period))
    u = np.tile(base, (reps, 1))[:T]
    return u

def multisine(T: int, m: int, rng: np.random.Generator, k_lines: int = 8) -> np.ndarray:
    """Sum of sines with random phases on k_lines distinct bins."""
    t = np.arange(T)
    u = np.zeros((T, m))
    for j in range(m):
        freqs = rng.choice(np.arange(1, max(2, T//8)), size=k_lines, replace=False)
        phases = rng.uniform(0, 2*np.pi, size=k_lines)
        w = np.zeros(T)
        for f, ph in zip(freqs, phases):
            w += np.sin(2*np.pi*f*t/T + ph)
        u[:, j] = w / np.max(np.abs(w))
    return u

def hankel_blocks(u: np.ndarray, s: int) -> np.ndarray:
    """Build block Hankel of depth s for multi-input u (T x m)."""
    T, m = u.shape
    cols = T - 2*s + 1
    if cols <= 0:
        raise ValueError("Not enough samples for requested Hankel depth.")
    H = []
    for i in range(s):
        H.append(u[i:i+cols].T)  # (m x cols)
    return np.vstack(H)  # (s*m x cols)

def estimate_pe_order(u: np.ndarray, s_max: int, tol: float = 1e-8) -> int:
    """Largest s such that block-Hankel H_s has full row rank (up to tol)."""
    T, m = u.shape
    for s in range(min(s_max, T//2), 0, -1):
        H = hankel_blocks(u, s)  
        r = np.linalg.matrix_rank(H, tol=tol)
        if r == H.shape[0]:
            return s
    return 0

def restrict_pointwise(u: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Project inputs onto span(W) (columns of W form a basis of allowed directions)."""
    Winv = np.linalg.pinv(W)
    P = W @ Winv  
    return u @ P.T
