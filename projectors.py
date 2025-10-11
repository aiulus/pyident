from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from numpy.linalg import svd, norm
from scipy.linalg import qr

EPS = 1e-12


def build_visible_basis_dt(Ad, Bd, x0, tol=1e-10, max_pow=None):
    """
    Returns P (n×k) with orthonormal columns spanning
    span{ [x0 Bd], Ad[x0 Bd], ..., Ad^{n-1}[x0 Bd] }.

    Prefers column-pivoted QR from SciPy; falls back to SVD if SciPy unavailable.
    """
    import numpy as np
    n = Ad.shape[0]
    if max_pow is None:
        max_pow = n - 1

    K0 = np.column_stack([x0.reshape(-1, 1), Bd])  # n×(1+m)
    blocks = [K0]
    Ak = Ad.copy()
    for _ in range(max_pow):
        blocks.append(Ak @ K0)
        Ak = Ak @ Ad
    M = np.concatenate(blocks, axis=1)  # n×((n)*(1+m))

    thr = tol * np.linalg.norm(M, 'fro')

    # Try SciPy pivoted QR first
    try:
        result = qr(M, mode='economic', pivoting=True)
        if len(result) == 3:
            Q, R, piv = result
        elif len(result) == 2:
            Q, R = result
        else:
            Q = result[0]
            R = np.zeros((Q.shape[1], Q.shape[1]), dtype=Q.dtype)  # fallback: dummy R
        diag = np.abs(np.diag(R))
        r = int((diag > thr).sum())
        Q = np.asarray(Q)  # ensure Q is a NumPy array for slicing
        return Q[:, :r]
    except Exception:
        # Fallback: SVD-based column space
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        r = int((s > thr).sum())
        return U[:, :r]

# ---------- basic LA helpers ----------
def _svd_nullspace(M: np.ndarray, tol: Optional[float] = None) -> np.ndarray:
    """
    Orthonormal basis for ker(M) using SVD rank cut.
    Works for tall/wide matrices. Returns n x k matrix (possibly k=0).
    """
    if M.size == 0:
        return np.zeros((M.shape[1], 0))
    U, S, Vt = np.linalg.svd(M, full_matrices=True)
    if tol is None:
        tol = max(M.shape) * np.finfo(float).eps * (S[0] if S.size else 1.0)
    r = int(np.sum(S > tol))                # numerical rank
    return Vt.T[:, r:]                      # the last n-r columns span ker(M)


def _orth(A: np.ndarray) -> np.ndarray:
    if A.size == 0:
        return np.zeros((A.shape[0], 0))
    U, S, Vt = svd(A, full_matrices=False)
    tol = max(A.shape) * np.finfo(float).eps * (S[0] if S.size else 1.0)
    r = int(np.sum(S > tol))
    return U[:, :r]

def projector_from_basis(Vbasis):
    n = Vbasis.shape[0] if Vbasis.size else 0
    if Vbasis.size == 0:
        return np.zeros((n, n))
    Q, _ = np.linalg.qr(Vbasis, mode="reduced")
    return Q @ Q.T


def projector_onto_complement(Vbasis: np.ndarray) -> np.ndarray:
    """Orthogonal projector onto ``span(Vbasis)``\ :sup:`⊥`.

    For an empty basis this reduces to the identity on the ambient space.
    """

    n = Vbasis.shape[0] if Vbasis.ndim >= 1 else 0
    if Vbasis.size == 0 or Vbasis.shape[1] == 0:
        return np.eye(n)

    return np.eye(n) - projector_from_basis(Vbasis)


def normalize(x: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    nrm = float(norm(x))
    if nrm <= tol:
        raise ValueError("Vector too small to normalize.")
    return x / nrm

# ---------- core: left-uncontrollable space ----------
def left_uncontrollable_subspace(A: np.ndarray, B: np.ndarray,
                                 tol: Optional[float] = None,
                                 max_iter: int = 50) -> np.ndarray:
    """
    Largest A^T-invariant subspace contained in ker(B^T).
    """
    n = A.shape[0]
    Wk = _svd_nullspace(B.T, tol=tol)      # start in ker(B^T)
    if Wk.shape[1] == 0:
        return Wk
    Wk = _orth(Wk)

    for _ in range(max_iter):
        Pk = projector_onto_complement(Wk)      # Pk = I - Wk Wk^T  (projects onto (span Wk)^\perp)
        N  = np.vstack([Pk @ A.T, B.T])    # enforce: Pk A^T w = 0 and B^T w = 0
        Wn = _svd_nullspace(N, tol=tol)
        Wn = _orth(Wn)

        # convergence: projector distance and/or dim stability
        if (Wn.shape[1] == Wk.shape[1] and
            norm(projector_onto_complement(Wn) - projector_onto_complement(Wk), 2) <= (tol or 1e-10)):
            return Wn
        Wk = Wn
    return Wk


# ---------- selecting a subset to keep dark ----------
def choose_dark_subset(W_all: np.ndarray, k: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Pick k columns from an orthonormal basis W_all to suppress (k ≥ 1)."""
    k = int(max(1, min(k, W_all.shape[1])))
    if W_all.shape[1] == 0:
        return W_all
    rng = np.random.default_rng() if rng is None else rng
    idx = rng.choice(W_all.shape[1], size=k, replace=False)
    return W_all[:, idx]

# ---------- main builders ----------
def make_dark_projector(A: np.ndarray, B: np.ndarray, k_off: int = 1,
                        rng: Optional[np.random.Generator] = None,
                        tol: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (W_all, W_off, P_dark) where W_all spans all uncontrollable left modes,
    W_off (⊆ W_all) has k_off columns chosen to stay dark, and P_dark projects onto ker(W_off^T).
    """
    W_all = left_uncontrollable_subspace(A, B, tol=tol)
    if W_all.shape[1] == 0:
        n = A.shape[0]
        return W_all, np.zeros((n, 0)), np.eye(n)
    W_off = choose_dark_subset(W_all, k=k_off, rng=rng)
    # columns of W_all are orthonormal; subset preserves orthonormality
    P_dark = projector_onto_complement(W_off)
    return W_all, W_off, P_dark

def build_projected_x0(A: np.ndarray, B: np.ndarray, x0_seed: np.ndarray,
                       k_off: int = 1, rng: Optional[np.random.Generator] = None,
                       tol: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project x0_seed onto ker(W_off^T) to make at least k_off uncontrollable directions dark.
    Returns (x0_proj, W_off, P_dark).
    """
    W_all, W_off, P_dark = make_dark_projector(A, B, k_off=k_off, rng=rng, tol=tol)
    x = P_dark @ x0_seed
    if norm(x) <= (tol or 1e-10):
        # try a few random seeds
        rng = np.random.default_rng() if rng is None else rng
        for _ in range(32):
            x_try = P_dark @ rng.standard_normal(A.shape[0])
            if norm(x_try) > (tol or 1e-10):
                x = x_try
                break
    if norm(x) <= (tol or 1e-10):
        # if still tiny (e.g., k_off == n), fall back to any unit vector in ker(W_off^T)
        # find an orth basis for ker(W_off^T) by orthonormal complement of W_off
        n = A.shape[0]
        if W_off.shape[1] >= n:
            raise RuntimeError("ker(W_off^T) is trivial; cannot project.")
        Qc = _orth(np.eye(n) - W_off @ W_off.T)
        x = Qc[:, 0]
    return normalize(x), W_off, P_dark



