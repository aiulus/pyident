from __future__ import annotations
import numpy as np
import numpy.linalg as npl
from typing import Tuple, Optional
from scipy.linalg import expm, eigvals, solve_continuous_lyapunov, solve_discrete_lyapunov, null_space

# ---------------------------------------------------------------------
# Metrics & core objects 
# - Krylov generators (full / pointwise / PE-truncated)
# - Visible subspace basis
# - Gramian-based tests
# - PBH margins (structured/unstructured)    [kept here; can be moved to pbh.py]
# - Mode overlaps / projection errors
#
# Disclaimer:
# - Any “infinite-horizon” CT Gramian is only returned when A is Hurwitz.
# - PBH margins are evaluated exactly at eigenvalues; near-defective cases
#   may need small complex-radius sweeps (optional hook provided).
# ---------------------------------------------------------------------


# ====================== Discretization ===============================

def cont2discrete_zoh(A: np.ndarray, B: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-order hold (ZOH) discretization of (A,B) with step dt."""
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A * dt
    M[:n, n:] = B * dt
    Md = expm(M)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd



# ====================== Krylov / Visible subspace ====================

def _krylov(A: np.ndarray, X: np.ndarray, depth: int) -> np.ndarray:
    """Return [X, AX, ..., A^{depth-1} X]. If depth<=0, return n×0."""
    n = A.shape[0]
    if depth <= 0:
        return np.zeros((n, 0), dtype=A.dtype)
    blocks = [X]
    P = X
    for _ in range(1, depth):
        P = A @ P
        blocks.append(P)
    return np.concatenate(blocks, axis=1)



def krylov_generator(A: np.ndarray, X: np.ndarray, depth: Optional[int] = None) -> np.ndarray:
    """Public wrapper. If depth is None, use n (Cayley–Hamilton upper bound)."""
    n = A.shape[0]
    d = n if depth is None else int(depth)
    return _krylov(A, X, d)


def unified_generator(
    A: np.ndarray,
    B: np.ndarray,
    x0: np.ndarray,
    mode: str = "unrestricted",
    W: Optional[np.ndarray] = None,
    r: Optional[int] = None,
) -> np.ndarray:
    """
    - mode="unrestricted": K = [x0, B, A[x0 B], ..., A^{n-1}[x0 B]]
    - mode="pointwise":    K = [x0, B@W, A[x0 B@W], ..., A^{n-1}[x0 B@W]]
    - mode="moment-pe":    K = [Krylov(A,[x0 B], r-1), A^r x0, ..., A^{n-1} x0]
    """
    n = A.shape[0]
    if mode == "unrestricted":
        X = np.concatenate([x0.reshape(-1, 1), B], axis=1)
        return _krylov(A, X, n)
    elif mode == "pointwise":
        if W is None:
            raise ValueError("pointwise mode requires W (m×q).")
        X = np.concatenate([x0.reshape(-1, 1), B @ W], axis=1)
        return _krylov(A, X, n)
    elif mode == "moment-pe":
        if r is None or r < 1:
            raise ValueError("moment-pe mode requires r>=1.")
        X = np.concatenate([x0.reshape(-1, 1), B], axis=1)
        Khead = _krylov(A, X, max(1, min(r, n)))           # depth r
        # add [A^r x0, ..., A^{n-1} x0]
        v = x0.copy()
        for _ in range(r):
            v = A @ v
        tail = [v.reshape(-1, 1)]
        for _ in range(r + 1, n):
            v = A @ v
            tail.append(v.reshape(-1, 1))
        Ktail = np.concatenate(tail, axis=1) if tail else np.zeros((n, 0))
        return np.concatenate([Khead, Ktail], axis=1)
    else:
        raise ValueError(f"unknown mode: {mode}")

def visible_subspace_basis(
    A: np.ndarray,
    B: np.ndarray,
    x0: np.ndarray,
    *,
    mode: str = "unrestricted",
    W: Optional[np.ndarray] = None,
    r: Optional[int] = None,
    tol: float = 1e-10,
):
    """Return (P_V, k) where columns of P_V span the visible subspace."""
    K = unified_generator(A, B, x0, mode=mode, W=W, r=r)
    if K.size == 0:
        return np.zeros((A.shape[0], 0)), 0
    U, S, _ = npl.svd(K, full_matrices=False)
    k = int((S > tol).sum())
    return U[:, :k], k



def visible_subspace(
    A: np.ndarray,
    B: np.ndarray,
    x0: np.ndarray,
    mode: str = "unrestricted",
    W: Optional[np.ndarray] = None,
    r: Optional[int] = None,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, int]:
    """Return an orthonormal basis P_V for the visible subspace V_• and its dimension.

    We compute P_V via rank-revealing SVD of the unified generator K(𝒰; x0).
    """
    K = unified_generator(A, B, x0, mode=mode, W=W, r=r)
    if K.size == 0:
        return np.zeros((A.shape[0], 0)), 0
    U, S, _ = npl.svd(K, full_matrices=False)
    k = int((S > tol).sum())
    return U[:, :k], k


def projector_from_basis(Vbasis: np.ndarray) -> np.ndarray:
    """Orthogonal projector onto span(Vbasis)."""
    if Vbasis.size == 0:
        return np.zeros((0, 0), dtype=float)
    Q, _ = npl.qr(Vbasis, mode="reduced")
    return Q @ Q.T


# ====================== Gramians =====================================

def is_hurwitz(A: np.ndarray, tol: float = 0.0) -> bool:
    lam = npl.eigvals(A)
    return bool(np.all(np.real(lam) < -tol))

def gramian_ct_infinite(A: np.ndarray, K: np.ndarray):
    """CT infinite-horizon Gramian for ẋ = A x + K w, if A is Hurwitz."""
    if not is_hurwitz(A, 0.0):
        return None
    try:
        return solve_continuous_lyapunov(A, -K @ K.T)
    except Exception:
        return None

def gramian_dt_infinite(Ad: np.ndarray, K: np.ndarray):
    """DT infinite-horizon Gramian for x_{k+1} = Ad x_k + K w_k, if ρ(Ad)<1."""
    try:
        rho = float(np.max(np.abs(npl.eigvals(Ad))))
    except Exception:
        rho = np.inf
    if rho < 1.0 - 1e-8:
        try:
            return solve_discrete_lyapunov(Ad, K @ K.T)
        except Exception:
            return None
    return None

def gramian_dt_finite(Ad: np.ndarray, K: np.ndarray, T: int) -> np.ndarray:
    """DT finite-horizon Gramian for x_{k+1}=Ad x_k + K w_k over T steps."""
    n = Ad.shape[0]
    W = np.zeros((n, n), dtype=float)
    P = np.eye(n)
    for _ in range(T):
        W = W + P @ (K @ K.T) @ P.T
        P = Ad @ P
    return W



# ====================== PBH margins (to move to pbh.py if desired) ===

def pbh_margin_structured(
    A: np.ndarray,
    B: np.ndarray,
    x0: np.ndarray,
    eigvals: Optional[np.ndarray] = None,
) -> float:
    """Structured Frobenius-distance proxy (TODO: link to the manuscript).

    Q has columns forming an orthonormal basis of x0^⊥, computed via null_space(x0^T).
    """
    if eigvals is None:
        eigvals = npl.eigvals(A)
    x = x0.reshape(-1, 1)
    Q = null_space(x.T)
    n = A.shape[0]
    aug = np.concatenate([x, B], axis=1)
    margin = np.inf
    for lam in eigvals:
        M = np.concatenate([lam * np.eye(n) - A, aug], axis=1).astype(np.complex128)
        smin = npl.svd(Q.T @ M, compute_uv=False).min().real if Q.size else npl.svd(M, compute_uv=False).min().real
        margin = min(margin, float(smin))
    return float(margin)


def pbh_margin_unstructured(
    A: np.ndarray,
    K: np.ndarray,
    eigvals: Optional[np.ndarray] = None,
) -> float:
    """Unstructured Frobenius-distance proxy: min_λ σ_min([λI−A, K])."""
    lam = npl.eigvals(A) if eigvals is None else eigvals
    n = A.shape[0]
    best = np.inf
    for l in lam:
        M = np.concatenate([l * np.eye(n) - A, K], axis=1)
        s = npl.svd(M, compute_uv=False)
        best = min(best, float(s[-1]))
    return best

def pbh_margin_with_ring(A: np.ndarray, K: np.ndarray,
                         ring_eps: float = 1e-6, ring_pts: int = 8) -> float:
    lam = np.linalg.eigvals(A)
    n = A.shape[0]
    margin = np.inf
    for l in lam:
        for t in range(ring_pts):
            theta = 2*np.pi*(t/ring_pts)
            lam_s = l + ring_eps*(np.cos(theta) + 1j*np.sin(theta))
            M = np.concatenate([lam_s*np.eye(n) - A, K], axis=1).astype(np.complex128)
            smin = np.linalg.svd(M, compute_uv=False).min().real
            margin = min(margin, float(smin))
    return float(margin)



# ====================== Overlaps / errors ============================

def left_eigvec_overlap(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return s with s_i = ||w_i^T X||_2 / ||w_i||_2 (left eigenvectors of A)."""
    lam, W = npl.eig(A.T)  # columns are eigenvectors of A^T
    num = npl.norm(W.T @ X, axis=1)
    den = npl.norm(W, axis=0)
    den = np.where(den == 0, 1.0, den)
    return (num / den).real


def x0R0_principle_angle(x0: np.ndarray, R0: np.ndarray) -> float:
    """Cosine of principal angle between x0 and span(R0): ||P_R x0|| / ||x0||."""
    x0 = x0.reshape(-1, 1)
    if x0.shape[0] != R0.shape[0]:
        raise ValueError("x0 and R0 row dimensions must match.")
    if npl.norm(x0) == 0:
        return 0.0
    # projector onto span(R0)
    Rplus = npl.pinv(R0)
    PR = R0 @ Rplus
    return float(npl.norm(PR @ x0) / npl.norm(x0))


def projected_errors(
    Ahat: np.ndarray,
    Bhat: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    Vbasis: np.ndarray,
) -> Tuple[float, float]:
    """||P_V (Ahat - A) P_V||_F, ||P_V (Bhat - B)||_F with P_V projector from Vbasis."""
    PV = projector_from_basis(Vbasis)
    dA = PV @ (Ahat - A) @ PV
    dB = PV @ (Bhat - B)
    return float(npl.norm(dA, "fro")), float(npl.norm(dB, "fro"))
