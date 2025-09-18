import numpy as np
from typing import Tuple, Optional

def _block_hankel(z: np.ndarray, s: int, start: int, cols: int) -> np.ndarray:
    """z: (T, q). Return H_s starting at 'start' with 'cols' columns, shape (s*q, cols)."""
    T, q = z.shape
    H = np.empty((s * q, cols), dtype=z.dtype)
    for i in range(s):
        H[i*q:(i+1)*q, :] = z[start + i:start + i + cols].T
    return H

def moesp_fullstate(
    u: np.ndarray,            # (T, m)
    x: np.ndarray,            # (T, n) -- y = x (full state available)
    n: int,
    i: Optional[int] = None,
    f: Optional[int] = None,
    rcond: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """MOESP (full-state). Returns (Ahat, Bhat)."""
    T, m = u.shape[0], u.shape[1]
    p = x.shape[1]   # = n (full state)

    # sensible defaults (work for small T, n=6, m=2)
    if i is None: i = max(n, int(np.ceil(1.25 * n)))
    if f is None: f = max(int(np.ceil(n / m)) + 1, 2)
    N_eff = T - (i + f - 1)
    if N_eff <= max(n, 10):
        raise ValueError(f"MOESP: N_eff={N_eff} too small (T={T}, i={i}, f={f}).")

    # block Hankels (time-major in z; we built _block_hankel accordingly)
    Up = _block_hankel(u, i, 0, N_eff)    # (m*i, N_eff)
    Uf = _block_hankel(u, f, i, N_eff)    # (m*f, N_eff)
    Xp = _block_hankel(x, i, 0, N_eff)    # (n*i, N_eff)
    Xf = _block_hankel(x, f, i, N_eff)    # (n*f, N_eff)

    # QR on past data, then project future outputs onto orth-complement of row([Up;Xp])
    Wp = np.vstack([Up, Xp])              # ((m+n)*i, N_eff)
    Q, _ = np.linalg.qr(Wp.T, mode="reduced")  # N_eff x r
    P_orth = np.eye(N_eff) - Q @ Q.T

    Xfo = Xf @ P_orth
    Ufo = Uf @ P_orth

    # eliminate Ufo via least squares (weighted MOESP)
    if np.linalg.matrix_rank(Ufo) > 0:
        G = Xfo @ Ufo.T @ np.linalg.pinv(Ufo @ Ufo.T, rcond=rcond)
        Xtil = Xfo - G @ Ufo
    else:
        Xtil = Xfo

    # SVD → dominant subspace (extended observability)
    U1, S1, V1t = np.linalg.svd(Xtil, full_matrices=False)
    rel = S1 / (S1[0] if S1.size else 1.0)
    r = int((rel > rcond).sum())
    if r < n:
        raise ValueError(f"MOESP: subspace rank {r} < n={n}. Increase T or adjust i/f.")

    Un = U1[:, :n]                        # (n*f, n)
    # Recover A via shift-invariance across f blocks
    # Un partition: [Gamma; Gamma A; ...; Gamma A^{f-1}] (stacked by n-rows)
    rows = n
    Un_top = Un[: (f-1)*rows, :]
    Un_bot = Un[rows : f*rows, :]
    Ahat = np.linalg.lstsq(Un_top, Un_bot, rcond=rcond)[0]

    # With full state, B is best recovered by a robust LS on the original relation:
    # X_plus = Ahat X_minus + B U_minus
    X_minus = x[:-1].T    # (n, T-1)
    X_plus  = x[1:].T     # (n, T-1)
    U_minus = u[:-1].T    # (m, T-1)
    Z = np.vstack([X_minus, U_minus])      # (n+m, T-1)
    Theta = X_plus @ Z.T @ np.linalg.pinv(Z @ Z.T, rcond=rcond)  # (n, n+m)
    Ahat_ls = Theta[:, :n]
    Bhat_ls = Theta[:, n:]
    # Blend Ahat (subspace) with Ahat_ls if you want; by default trust subspace A
    Bhat = Bhat_ls
    return Ahat, Bhat

# Backward-compatible wrapper if your module exports `moesp(...)`
def moesp(U: np.ndarray, X: np.ndarray, n: Optional[int] = None, *, s: Optional[int] = None, rcond: float = 1e-10):
    """
    Keep your external signature: U: (m,T), X: (n,T) in your code.
    This wrapper just transposes to time-major and calls moesp_fullstate.
    """
    m, T = U.shape
    n_state, T2 = X.shape
    assert T == T2, "U and X must have same T"
    if n is None: n = n_state
    Ahat, Bhat = moesp_fullstate(U.T, X.T, n=n, i=s, f=None, rcond=rcond)
    return Ahat, Bhat



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

     # Project onto orthogonal complement of row(Up) via QR (numerically safer)
    # This is the PI-MOESP variant; if you later add Yp, use [Up;Yp] here.
    Qp, _ = np.linalg.qr(Up.T, mode="reduced")     # cols × r
    P_orth = np.eye(Qp.shape[0]) - Qp @ Qp.T       # (cols × cols)
    Yf_perp = Yf @ P_orth
    Uf_perp = Uf @ P_orth

  
   # Eliminate remaining future inputs via LS (weighted MOESP)
    if np.linalg.matrix_rank(Uf_perp) > 0:
        G = Yf_perp @ Uf_perp.T @ np.linalg.pinv(Uf_perp @ Uf_perp.T, rcond=rcond)
        Ytil = Yf_perp - G @ Uf_perp
    else:
        Ytil = Yf_perp

    # SVD to extract extended observability Gamma_s
    Uy, Sy, Vy = np.linalg.svd(Ytil, full_matrices=False)
    # rank gate (actionable failure instead of bogus Null(Uf^T))
    if Sy.size == 0 or Sy[0] <= 0:
        raise ValueError("MOESP: degenerate projected data; increase T or adjust s.")
    rel = Sy / Sy[0]
    r = int((rel > rcond).sum())
    if r < n:
        raise ValueError(f"MOESP: subspace rank {r} < n={n}. Increase T or adjust s.")
    U1 = Uy[:, :n]
    S1 = Sy[:n]
    Gamma = U1 @ np.diag(np.sqrt(S1)) # (s*p, n)


    # C and A from shift structure of Gamma_s
    Chat = Gamma[:p, :]  # first block row = C
    Gamma_up   = Gamma[:-p, :]  # rows 0..(s-2)*p
    Gamma_down = Gamma[p:,  :]  # rows p..(s-1)*p
    Ahat = np.linalg.lstsq(Gamma_up, Gamma_down, rcond=rcond)[0]  # (n, n)

    # Reconstruct state sequence X_f from Gamma_s
    Xf = np.linalg.lstsq(Gamma, Ytil, rcond=rcond)[0] # (n, cols_projected) 

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