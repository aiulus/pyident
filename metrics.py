import numpy as np
import numpy.linalg as npl
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
from config import SolverOpts
from scipy.linalg import expm, solve_continuous_lyapunov, subspace_angles, null_space


def cont2discrete_zoh(A: np.ndarray, B: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-order hold discretization of (A, B) with step dt."""
    n, m = B.shape
    M = np.zeros((n+m, n+m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M*dt) 
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd

def _krylov(A: np.ndarray, X: np.ndarray, depth: int) -> np.ndarray:
    """
    Build [X, AX, ..., A^{depth-1} X] (n x depth*cols).
    depth == 0 returns an empty (n x 0) matrix by design.
    """
    n = A.shape[0]
    if depth <= 0:
        return np.zeros((n, 0))
    blocks = []
    M = X.copy()
    for _ in range(depth):
        blocks.append(M)
        M = A @ M
    return np.hstack(blocks)

def krylov_generator(A: np.ndarray, X: np.ndarray, depth: Optional[int] = None) -> np.ndarray:
    """
    Public wrapper for _krylov.
    """
    n = A.shape[0]
    k = n if depth is None else int(depth)
    return _krylov(A, X, k)

def is_hurwitz(A: np.ndarray, tol: float = 0.0) -> bool:
    """Return True iff Re(lambda_i) < -tol for all eigenvalues of A."""
    lam = np.linalg.eigvals(A)
    return bool(np.all(np.real(lam) < -tol))

def gramian_ct(A: np.ndarray, K: np.ndarray) -> Optional[np.ndarray]:
    """
    Continuous-time infinite-horizon reachability Gramian solving:
        A W + W A^T = - K K^T

    This is exact only if A is Hurwitz (stable). Otherwise returns None.

    K should be the augmented input matrix [x0  B].
    """
    if not is_hurwitz(A, tol=0.0):
        # Non-Hurwitz -> solution may not exist / be unbounded.
        return None
    try:
        return solve_continuous_lyapunov(A, - K @ K.T)
    except Exception:
        return None
    
def subspace_dimension(K: np.ndarray, tol: float = 1e-10) -> int:
    """Rank of K with tolerance tol."""
    return int(np.linalg.matrix_rank(K, tol=tol))

def projected_errors(Ahat: np.ndarray, Bhat: np.ndarray, A: np.ndarray, B: np.ndarray,
                     Vbasis: np.ndarray) -> Tuple[float, float]:
    """
    Return ||P_V (Ahat - A) P_V||_F, ||P_V (Bhat - B)||_F where P_V projects onto span(Vbasis).
    """
    n = A.shape[0]
    if Vbasis.size == 0:
        PV = np.zeros((n, n))
    else:
        PV = Vbasis @ np.linalg.pinv(Vbasis)  
    dA = PV @ (Ahat - A) @ PV
    dB = PV @ (Bhat - B)
    return float(np.linalg.norm(dA, 'fro')), float(np.linalg.norm(dB, 'fro'))


def pbh_margin_structured(A: np.ndarray, B: np.ndarray, x0: np.ndarray,
                          eigvals: np.ndarray | None = None) -> float:
    if eigvals is None:
        eigvals = np.linalg.eigvals(A)
    x = x0.reshape(-1, 1)
    Q = null_space(x.T)
    n = A.shape[0]
    margin = np.inf
    aug = np.concatenate([x, B], axis=1)
    for lam in eigvals:
        M = np.concatenate([lam * np.eye(n) - A, aug], axis=1).astype(np.complex128)
        smin = np.linalg.svd(Q.T @ M, compute_uv=False).min().real
        margin = min(margin, float(smin))
    return margin

def pbh_margin_unstructured(A: np.ndarray, K: np.ndarray,
                            eigvals: Optional[np.ndarray] = None) -> float:
    if eigvals is None:
        eigvals = np.linalg.eigvals(A)
    n = A.shape[0]
    margin = np.inf
    for lam in eigvals:
        M = np.concatenate([lam * np.eye(n) - A, K], axis=1).astype(np.complex128)
        smin = np.linalg.svd(M, compute_uv=False).min().real
        margin = min(margin, float(smin))
    return float(margin)
    
def left_eigvec_overlap(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Per-mode overlaps with columns of X, using left eigenvectors of A.

    Returns vector s with s_i = ||w_i^T X||_2 / ||w_i||_2,
    where A^T w_i = lambda_i w_i (columns of eig(A^T) are left eigenvectors of A).
    """
    evals, V_AT = np.linalg.eig(A.T)     
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
    x0 = x0.reshape(-1, 1)
    PR = R0 @ npl.pinv(R0.T @ R0) @ R0.T
    Px0 = np.linalg.norm(PR @ x0)
    nfactor = np.linalg.norm(x0) 
    return 0.0 if nfactor == 0.0 else float(Px0 / nfactor)

def unified_generator(A: np.ndarray, B: np.ndarray, x0: np.ndarray,
                    mode: str = "unrestricted",
                    W: Optional[np.ndarray] = None,
                    r: Optional[int] = None) -> np.ndarray:
    n = A.shape[0]
    if mode == "unrestricted":
        Kcore = np.concatenate([x0.reshape(-1, 1), B], axis=1)
        depth = n
        K = _krylov(A, Kcore, depth)
    elif mode == "pointwise":
        assert W is not None, "W is required for pointwise mode"
        BW = B @ (W @ np.linalg.pinv(W)) # projection onto span(W)
        Kcore = np.concatenate([x0.reshape(-1, 1), BW], axis=1)
        K = _krylov(A, Kcore, n)
    elif mode == "moment-pe":
        assert r is not None and r >= 1
        Kcore = np.concatenate([x0.reshape(-1, 1), B], axis=1)
        K1 = _krylov(A, Kcore, max(0, r-1))
        tail = []
        vec = x0.reshape(-1, 1)
        for k in range(r, n):
            vec = A @ vec 
            tail.append(vec)
        K2 = np.concatenate(tail, axis=1) if len(tail) else np.zeros((n, 0))
        K = np.hstack([K1, K2])
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return K

def _krylov(A: np.ndarray, X: np.ndarray, depth: int) -> np.ndarray:
    blocks = []
    M = X.copy()
    for _ in range(depth):
        blocks.append(M)
        M = A @ M
    return np.hstack(blocks) if blocks else X

def pbh_margin(A: np.ndarray, B: np.ndarray, x0: np.ndarray,
               K: np.ndarray, eigvals: Optional[np.ndarray] = None) -> float:
    if eigvals is None:
        eigvals = np.linalg.eigvals(A)
    n = A.shape[0]
    margin = np.inf
    for lam in eigvals:
        M = np.concatenate([lam * np.eye(n) - A, K], axis=1).astype(np.complex128)
        s = np.linalg.svd(M, compute_uv=False)
        margin = min(margin, float(s.min().real))
    return float(margin)


