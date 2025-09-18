import numpy as np
from typing import Tuple, Optional


def _block_hankel(z: np.ndarray, s: int, start: int, cols: int) -> np.ndarray:
    """z: (T, q). Return H_s starting at 'start' with 'cols' columns, shape (s*q, cols)."""
    T, q = z.shape
    H = np.empty((s * q, cols), dtype=z.dtype)
    for i in range(s):
        H[i * q:(i + 1) * q, :] = z[start + i:start + i + cols].T
    return H


def moesp_pi(
    u: np.ndarray,            # (T, m)
    y: np.ndarray,            # (T, p)
    s: int,                   # block rows (>= n)
    n: int,                   # model order
    rcond: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Past-Input MOESP (PI-MOESP) with D=0.

    Returns
    -------
    Ahat (n×n), Bhat (n×m), Chat (p×n)

    Disclaimer
    -------------------------
    - Requires sufficient excitation so that Null(U_f^T) has dimension >= n.
    - Rank decisions use SVD with tolerance tied to the largest singular value (1e-12 factor).
    - No output noise model; y is assumed to be the measured output (here y=x for full-state).
    """
    T, m = u.shape
    Ty, p = y.shape
    if Ty != T:
        raise ValueError("u and y must have the same time length.")
    cols = T - 2 * s + 1
    if cols <= 1:
        raise ValueError("Not enough samples for MOESP: need T >= 2*s + 1.")

    # Past/Future block Hankels
    Up = _block_hankel(u, s, start=0, cols=cols)      # (s*m, cols)   
    Uf = _block_hankel(u, s, start=s, cols=cols)      # (s*m, cols)
    Yf = _block_hankel(y, s, start=s, cols=cols)      # (s*p, cols)

    # Project Yf onto orthogonal complement of row(Uf)
    Uu, Su, Vt = np.linalg.svd(Uf, full_matrices=False)
    tol_u = Su.max() * 1e-12 if Su.size else 0.0
    rank_u = int(np.sum(Su > tol_u))
    V2 = Vt[rank_u:, :].T                            
    if V2.shape[1] < n:
        raise ValueError("Insufficient excitation: dim Null(Uf^T) < n.")

    Yf_perp = Yf @ V2                                 

    # SVD to extract extended observability Gamma_s
    Uy, Sy, Vy = np.linalg.svd(Yf_perp, full_matrices=False)
    U1 = Uy[:, :n]
    S1 = Sy[:n]
    Gamma = U1 @ np.diag(np.sqrt(S1)) # (s*p, n)

    # C and A from shift structure of Gamma_s
    Chat = Gamma[:p, :]  # first block row = C
    Gamma_up   = Gamma[:-p, :]  # rows 0..(s-2)*p
    Gamma_down = Gamma[p:,  :]  # rows p..(s-1)*p
    Ahat = np.linalg.lstsq(Gamma_up, Gamma_down, rcond=rcond)[0]  # (n, n)

    # Reconstruct state sequence X_f from Gamma_s X_f ≈ Y_f,⊥
    Xf = np.linalg.lstsq(Gamma, Yf_perp, rcond=rcond)[0] # (n, cols - rank_u)

    # Align to estimate B: use the first block-row of Uf (u_k) aligned with Xk
    Xk  = Xf[:, :-1]
    Xk1 = Xf[:,  1:]
    Uf_proj = (Uf @ V2) # (s*m, cols - rank_u)
    Uk = Uf_proj[:m, :-1] # u_k aligned with Xk
    resid = Xk1 - Ahat @ Xk
    Bhat = resid @ np.linalg.pinv(Uk, rcond=rcond)

    return Ahat, Bhat, Chat


def moesp_fit(
    X: np.ndarray,
    Xp: np.ndarray,
    U: np.ndarray,
    s: int = 10,
    n: Optional[int] = None,
    rcond: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper for full-state measurements (y = x).

    Inputs
    ------
    X   : (n, T-1)   state snapshots (we treat this as y over a horizon of length T' = T-1)
    Xp  : (n, T-1)   next snapshots (unused by MOESP but kept for API symmetry)
    U   : (m, T-1)   inputs aligned with columns of X
    s   : block rows (>= n ideally)
    n   : model order (defaults to X.shape[0] = full order for full-state)

    Returns
    -------
    Ahat : (n, n)
    Bhat : (n, m)
    """
    n_state = X.shape[0]
    if n is None:
        n = n_state

    # Rebuild time series (T' samples)
    u_ts = U.T # (T', m)
    y_ts = X.T # (T', n_state) with y = x (full-state)
    Ahat, Bhat, _C = moesp_pi(u_ts, y_ts, s=s, n=n, rcond=rcond)
    return Ahat, Bhat

__all__ = ["moesp"]