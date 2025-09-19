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
    """Zero-order hold discretization of (A, B) with step dt."""
    n, m = B.shape
    M = np.zeros((n + m, n + m), dtype=float)
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd


# ====================== Krylov / Visible subspace ====================

def _krylov(A: np.ndarray, X: np.ndarray, depth: int) -> np.ndarray:
    """Return [X, AX, ..., A^{depth-1} X]. If depth<=0, return empty (n×0)."""
    n = A.shape[0]
    if depth <= 0:
        return np.zeros((n, 0), dtype=A.dtype)
    blocks = []
    M = X.copy()
    for _ in range(depth):
        blocks.append(M)
        M = A @ M
    return np.hstack(blocks)


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
    TODO: link to the manuscript
    mode:
      - "unrestricted": K = [K_core, A K_core, …, A^{n-1} K_core], K_core = [x0, B].
      - "pointwise":    replace B by B·P, P the projector onto span(W).
      - "moment-pe":    K = [K_core, …, A^{r-2} K_core, A^r x0, …, A^{n-1} x0].
    """
    n = A.shape[0]
    x = x0.reshape(-1, 1)

    if mode == "unrestricted":
        Kcore = np.concatenate([x, B], axis=1)
        K = _krylov(A, Kcore, n)

    elif mode == "pointwise":
        if W is None:
            raise ValueError("W is required for pointwise mode.")
        P = W @ np.linalg.pinv(W)  # projector onto span(W)
        BW = B @ P
        Kcore = np.concatenate([x, BW], axis=1)
        K = _krylov(A, Kcore, n)

    elif mode == "moment-pe":
        if r is None or r < 1:
            raise ValueError("moment-pe mode requires r>=1.")
        Kcore = np.concatenate([x, B], axis=1)
        K1 = _krylov(A, Kcore, max(0, r - 1))  # [K_core ... A^{r-2} K_core]
        # Tail: [A^r x0 ... A^{n-1} x0]
        tail = []
        vec = x.copy()
        for k in range(1, n):  # build A^k x0 up to n-1
            vec = A @ vec
            if k >= r:
                tail.append(vec)
        K2 = np.hstack(tail) if len(tail) else np.zeros((n, 0), dtype=A.dtype)
        K = np.hstack([K1, K2])

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return K


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
    """Orthogonal projector onto span(Vbasis) (handles empty basis)."""
    n = Vbasis.shape[0]
    if Vbasis.size == 0:
        return np.zeros((n, n))
    # Orthonormalize if not already
    Q, _ = npl.qr(Vbasis, mode="reduced")
    return Q @ Q.T


# ====================== Gramians =====================================

def is_hurwitz(A: np.ndarray, tol: float = 0.0) -> bool:
    """Return True iff Re(lambda_i) < -tol for all eigenvalues of A."""
    lam = npl.eigvals(A)
    return bool(np.all(np.real(lam) < -tol))


def gramian_ct_infinite(A: np.ndarray, K: np.ndarray) -> Optional[np.ndarray]:
    """Continuous-time infinite-horizon reachability Gramian:
         A W + W A^T = - K K^T
       Only returned if A is Hurwitz; else None.

    Disclaimer: this *assumes stability*. If A is not Hurwitz,
    prefer finite-horizon gramians or DT equivalents.
    """
    if not is_hurwitz(A, tol=0.0):
        return None
    try:
        return solve_continuous_lyapunov(A, -K @ K.T)
    except Exception:
        return None


def gramian_dt_finite(A: np.ndarray, K: np.ndarray, T: int) -> np.ndarray:
    n = A.shape[0]
    W = np.zeros((n, n), dtype=float)
    Ak = np.eye(n, dtype=float)
    KKt = K @ K.T
    for _ in range(int(T)):
        W += Ak @ KKt @ Ak.T
        Ak = A @ Ak
    return W

def gramian_infinite(A: np.ndarray, B: np.ndarray):
    """
    Return infinite-horizon controllability Gramian W (CT if Hurwitz, else DT if rho(A)<1),
    or None if neither converges.
    """
    vals = eigvals(A)
    lam_max_real = float(np.max(np.real(vals))) if vals.size else 0.0
    # CT case: Hurwitz
    if lam_max_real < -1e-8:
        try:
            return solve_continuous_lyapunov(A, -(B @ B.T))
        except Exception:
            return None
    # DT case: spectral radius < 1
    try:
        rho = float(np.max(np.abs(vals)))
    except Exception:
        rho = np.inf
    if rho < 1 - 1e-8:
        try:
            return solve_discrete_lyapunov(A, B @ B.T)
        except Exception:
            return None
    return None


def gramian_dt_infinite(A_d: np.ndarray, K: np.ndarray):
    """
    Infinite-horizon DT controllability Gramian for x_{k+1} = A_d x_k + K w_k.
    Returns None if spectral radius >= 1 or solver fails.
    """
    try:
        rho = float(np.max(np.abs(npl.eigvals(A_d))))
    except Exception:
        rho = np.inf
    if rho < 1.0 - 1e-8:
        try:
            return solve_discrete_lyapunov(A_d, K @ K.T)
        except Exception:
            return None
    return None


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
    """Unstructured Frobenius-distance proxy: min_λ σ_min([λI-A, K])."""
    if eigvals is None:
        eigvals = npl.eigvals(A)
    n = A.shape[0]
    margin = np.inf
    for lam in eigvals:
        M = np.concatenate([lam * np.eye(n) - A, K], axis=1).astype(np.complex128)
        smin = npl.svd(M, compute_uv=False).min().real
        margin = min(margin, float(smin))
    return float(margin)


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
    """Per-mode overlaps with columns of X using left eigenvectors of A.

    Returns s with s_i = ||w_i^T X||_2 / ||w_i||_2, where A^T w_i = λ_i w_i.
    """
    _, V_AT = npl.eig(A.T)     # columns are left eigenvectors of A
    W = V_AT.astype(np.complex128)
    k = W.shape[1]
    Xc = X.astype(np.complex128)
    scores = np.empty(k, dtype=float)
    for i in range(k):
        wi = W[:, i].reshape(1, -1)
        denom = npl.norm(wi)
        scores[i] = 0.0 if denom == 0 else float(npl.norm(wi @ Xc) / denom)
    return scores


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
