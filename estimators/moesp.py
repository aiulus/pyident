import numpy as np
from typing import Tuple

def _block_hankel(z: np.ndarray, s: int, start: int, cols: int) -> np.ndarray:
    # z: (T, q). Returns H_s starting at index 'start' with 'cols' columns.
    T, q = z.shape
    H = np.empty((s*q, cols))
    for i in range(s):
        H[i*q:(i+1)*q, :] = z[start+i:start+i+cols].T
    return H

def moesp(u: np.ndarray, y: np.ndarray, s: int, n: int, rcond: float = 1e-10
          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PI-MOESP (past-input MOESP), D=0.
    Inputs:
      u: (T, m), y: (T, p). s: block-rows (>= n). n: model order.
    Returns:
      Ahat (n,n), Bhat (n,m), Chat (p,n).
    """
    T, m = u.shape
    Ty, p = y.shape
    assert Ty == T, "u and y must have same length"
    cols = T - 2*s + 1
    if cols <= 1:
        raise ValueError("Not enough data: need T >= 2*s + 1.")

    # Past/Future block Hankels
    Up = _block_hankel(u, s, start=0,   cols=cols)    # (s*m, cols)
    Uf = _block_hankel(u, s, start=s,   cols=cols)    # (s*m, cols)
    Yf = _block_hankel(y, s, start=s,   cols=cols)    # (s*p, cols)

    # Project Yf onto the orthogonal complement of row(Uf)
    # Using SVD of Uf to get a basis for Null(Uf^T) in the columns of V2
    Uu, Su, Vt = np.linalg.svd(Uf, full_matrices=False)
    rank_u = np.sum(Su > (Su.max() * 1e-12))
    V2 = Vt[rank_u:, :].T  # (cols, cols - rank_u); columns span Null(Uf^T)
    if V2.shape[1] < n:
        raise ValueError("Insufficient excitation: Null(Uf^T) too small for order n.")

    Yf_perp = Yf @ V2  # (s*p, cols - rank_u)

    # SVD to get extended observability Γ_s
    Uy, Sy, Vy = np.linalg.svd(Yf_perp, full_matrices=False)
    U1 = Uy[:, :n]
    S1 = Sy[:n]
    Gamma = U1 @ np.diag(np.sqrt(S1))  # extended observability (up to similarity)

    # Extract C and A from Γ_s shift structure
    Chat = Gamma[:p, :]                             # first p rows
    Gamma_up   = Gamma[:-p, :]                      # rows 0..(s-2)*p
    Gamma_down = Gamma[p:,  :]                      # rows p..(s-1)*p
    Ahat = np.linalg.lstsq(Gamma_up, Gamma_down, rcond=rcond)[0]  # (n,n)

    # Reconstruct "future" state sequence Xf from Γ_s Xf ≈ Yf_perp
    Xf = np.linalg.lstsq(Gamma, Yf_perp, rcond=rcond)[0]          # (n, cols - rank_u)

    # Align regressors to estimate B with A fixed:
    # Use time pairs (x_k, x_{k+1}) from consecutive columns of Xf
    Xk  = Xf[:, :-1]
    Xk1 = Xf[:,  1:]
    Uf_proj = (Uf @ V2)               # (s*m, cols - rank_u)
    Uk = Uf_proj[:m, :-1]             # first block row gives u_k aligned with Xk

    # Solve Xk1 = Ahat Xk + Bhat Uk  ->  Bhat = (Xk1 - Ahat Xk) Uk^+
    resid = Xk1 - Ahat @ Xk
    Bhat = resid @ np.linalg.pinv(Uk, rcond=rcond)

    return Ahat, Bhat, Chat
