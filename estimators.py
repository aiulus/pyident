# pyident/estimators.py
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

# -----------------------------
# 1) DMDc (minimum-norm pinv)
# -----------------------------
def dmdc_pinv(
    X: np.ndarray,   # (n, T)
    Xp: np.ndarray,  # (n, T)
    U: np.ndarray,   # (m, T)
    rcond: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimum-norm DMDc via pseudoinverse.
    Solves Xp ≈ [A B] [X; U] with Θ = Xp Z^+ , Z=[X;U].
    Exact in noiseless data; returns a solution even if Z is rank-deficient.
    """
    n = X.shape[0]
    Z = np.vstack([X, U])              # (n+m, T)
    Z_pinv = np.linalg.pinv(Z, rcond=rcond)
    AB = Xp @ Z_pinv                   # (n, n+m)
    Ahat = AB[:, :n]
    Bhat = AB[:, n:]
    return Ahat, Bhat

# Backward-compat shim used across the repo/tests
def dmdc_fit(X: np.ndarray, Xp: np.ndarray, U: np.ndarray, rcond: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    return dmdc_pinv(X, Xp, U, rcond=rcond)


# -----------------------------
# 2) DMDc (ridge / Tikhonov)
# -----------------------------
def dmdc_ridge(
    X: np.ndarray, Xp: np.ndarray, U: np.ndarray, lam: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ridge-regularized DMDc:
      Θ = (Xp Z^T) (Z Z^T + λ I)^{-1},  Z=[X;U].
    More stable when Z is ill-conditioned or short.
    """
    n = X.shape[0]
    Z = np.vstack([X, U])                   # (n+m, T)
    ZZt = Z @ Z.T
    G = ZZt + lam * np.eye(ZZt.shape[0], dtype=ZZt.dtype)
    AB = (Xp @ Z.T) @ np.linalg.solve(G, np.eye(G.shape[0], dtype=G.dtype))
    Ahat = AB[:, :n]
    Bhat = AB[:, n:]
    return Ahat, Bhat


# -----------------------------
# 3) DMDc (truncated SVD)
# -----------------------------
def dmdc_tsvd(
    X: np.ndarray, Xp: np.ndarray, U: np.ndarray,
    rank: Optional[int] = None, svd_tol: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Truncated-SVD DMDc:
      Z = U Σ V^T ; use top-r or tol-based truncation to form Z^+_r.
    Useful to restrict to the excited/identifiable subspace explicitly.
    """
    n = X.shape[0]
    Z = np.vstack([X, U])                    # (n+m, T)
    Uz, Sz, Vtz = np.linalg.svd(Z, full_matrices=False)
    if rank is None:
        if svd_tol is not None:
            r = int(np.sum(Sz > svd_tol))
        else:
            r = int(np.sum(Sz > max(1e-12, Sz.max() * 1e-12)))
    else:
        r = int(min(rank, Sz.size))
    if r <= 0:
        # fall back to pinv if everything is tiny
        return dmdc_pinv(X, Xp, U, rcond=1e-10)
    Z_pinv_r = (Vtz[:r, :].T) @ np.diag(1.0 / Sz[:r]) @ (Uz[:, :r].T)
    AB = Xp @ Z_pinv_r
    Ahat = AB[:, :n]
    Bhat = AB[:, n:]
    return Ahat, Bhat


# ----------------------------------------------------
# 4) MOESP (full-state) + one-step B refit (simplified)
# ----------------------------------------------------
def moesp_fullstate(
    u_ts: np.ndarray,   # (T, m)
    x_ts: np.ndarray,   # (T, n) full state (C=I), same T as u
    n: int,
    i: Optional[int] = None,   # kept for signature parity; not used in simplified flow
    f: Optional[int] = None,   # kept for signature parity; not used in simplified flow
    rcond: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full-state identification in the noiseless case can be done via one-step LS,
    which equals the MOESP A-estimate under ideal conditions. We compute A,B from:
        Xp = A X + B U
    where X = x_ts[:-1].T, U = u_ts[:-1].T, Xp = x_ts[1:].T.
    """
    X  = x_ts[:-1].T   # (n, T-1)
    Xp = x_ts[1:].T    # (n, T-1)
    U  = u_ts[:-1].T   # (m, T-1)
    # use minimum-norm solution; for conditioning, user can switch to ridge/tsvd
    return dmdc_pinv(X, Xp, U, rcond=rcond)


# Backward-compat wrapper: same call sites as old code/tests
def moesp_fit(
    X: np.ndarray,      # (n, T)
    Xp: np.ndarray,     # (n, T)
    U: np.ndarray,      # (m, T)
    s: Optional[int] = None,   # unused in simplified flow
    n: Optional[int] = None,
    rcond: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper to accept (X,Xp,U) blocks (as used in run_single/tests) and call
    moesp_fullstate with time-major sequences.
    """
    assert X.shape[1] == U.shape[1] == Xp.shape[1], "time lengths must match"
    u_ts = U.T                       # (T, m)
    x_ts = X.T                       # (T, n)
    n_use = int(n if n is not None else X.shape[0])
    return moesp_fullstate(u_ts, x_ts, n=n_use, i=s, f=None, rcond=rcond)


# ----------------------------------------------------
# 5) Identifiable-component projector (utility)
# ----------------------------------------------------
def project_identifiable(theta: np.ndarray, Z: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """
    Project parameter matrix Θ onto the identifiable component given regressors Z=[X;U].
    This right-multiplies by the projector onto col(Z):
        P = Z Z^+   (shape (n+m)×(n+m)),
    so Θ_ident = Θ P only alters directions that were not excited by the data.
    """
    P = Z @ np.linalg.pinv(Z, rcond=rcond)   # projector onto col(Z)
    return theta @ P
