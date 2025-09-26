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


def same_equiv_class(A1: np.ndarray, B1: np.ndarray,
                     A2: np.ndarray, B2: np.ndarray,
                     x0: np.ndarray, tol: float = 1e-10) -> tuple[bool, dict]:
    """
    Check (A2,B2) ∈ [A1,B1]_{x0} under full input richness.
    Criterion (per thesis):
      (i) B2 == B1
      (ii) A2|_V == A1|_V on V = K(A1,[x0 B1]).

    Returns (ok, info) where ok is True/False and info has diagnostics.
    """
    # 1) exact-equality of B (within tol)
    dB = np.linalg.norm(B2 - B1, ord='fro')

    # 2) build visible subspace for (A1,B1,x0)
    P, k = visible_subspace(A1, B1, x0, tol=tol)  # P is n×k with orthonormal cols
    n = A1.shape[0]
    I = np.eye(n)

    # 3) compare restrictions on V and check leakage
    A1_V = P.T @ A1 @ P
    A2_V = P.T @ A2 @ P
    dA_V = np.linalg.norm(A2_V - A1_V, ord='fro')

    # Optional: ensure A2 maps V into V numerically (no leakage)
    leak_A2 = np.linalg.norm((I - P @ P.T) @ A2 @ P, ord='fro')

    ok = (dB <= tol) and (dA_V <= tol) and (leak_A2 <= tol)

    info = {
        "dim_V": int(k),
        "||B2-B1||_F": float(dB),
        "||A2|V - A1|V||_F": float(dA_V),
        "leak_A2": float(leak_A2),
        "tol": float(tol),
    }
    return ok, info


def visible_basis_dt(Ad,Bd,x0,tol_rank=1e-12):
    K = unified_generator(Ad, Bd, x0, mode="unrestricted")
    U, s, _ = np.linalg.svd(K, full_matrices=False)
    r = int((s > tol_rank * s[0]).sum())  # scale by s[0]
    return U[:, :r]


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

def pair_distance(
        Ahat: np.ndarray, 
        Bhat: np.ndarray, 
        A: np.ndarray, 
        B: np.ndarray) -> float:
    errA = float(npl.norm(Ahat - A, "fro"))
    errB = float(npl.norm(Bhat - B, "fro"))
    return float(np.mean([errA, errB]))


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

def controllability_subspace_basis(A: np.ndarray, B: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """Basis of span([B, AB, ..., A^{n-1}B]) via SVD (thin)."""
    n = A.shape[0]
    blocks = []
    Ak = np.eye(n)
    for k in range(n):
        blocks.append(Ak @ B)
        Ak = Ak @ A
    C = np.concatenate(blocks, axis=1)
    U, S, _ = np.linalg.svd(C, full_matrices=False)
    r = int(np.sum(S > rtol))
    return U[:, :r]

def eta0(A: np.ndarray, B: np.ndarray, x0: np.ndarray, rtol: float = 1e-10) -> float:
    """eta = ||Proj_{span(ctrl)} x0|| / ||x0||, ctrl = span([B, AB, ...])."""
    if np.linalg.norm(x0) == 0.0:
        return 0.0
    V = controllability_subspace_basis(A, B, rtol=rtol)
    px = V @ (V.T @ x0)
    return float(np.linalg.norm(px) / (np.linalg.norm(x0) + 1e-12))

def left_eig_overlaps(A: np.ndarray, x0: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    alpha_i = |v_i^T x0| / ||x0|| ; beta_i = ||v_i^T B||_2  where v_i are (unit) left eigenvectors of A.
    Returns (eigs, alpha, beta). Handles complex values; returns real magnitudes.
    """
    w, Vt = np.linalg.eig(A.T)           # columns of Vt are left eigenvectors of A
    V = np.asarray(Vt)
    norms = np.linalg.norm(V, axis=0)
    norms[norms == 0.0] = 1.0
    V = V / norms
    alpha = np.abs(V.T.conj() @ x0) / (np.linalg.norm(x0) + 1e-12)
    beta = np.linalg.norm(V.T.conj() @ B, axis=1)
    return w, alpha.real, beta.real

def ctrl_growth_metrics(A: np.ndarray, B: np.ndarray, rtol: float = 1e-10) -> tuple[int, int]:
    """
    Simple Brunovsky-like growth: track ranks of C_k = [B, AB, ..., A^{k-1}B].
    nu_max = smallest k with rank(C_k) = n; nu_gap = last_increase - first_increase (in steps).
    If never full rank, nu_max is current k*, nu_gap computed on observed increases.
    """
    n = A.shape[0]
    blocks = []
    growth_steps = []
    rank_prev = 0
    for k in range(1, n + 1):
        blocks.append(np.linalg.matrix_power(A, k - 1) @ B)
        Ck = np.concatenate(blocks, axis=1)
        r = int(np.linalg.matrix_rank(Ck, tol=rtol))
        if r > rank_prev:
            growth_steps.append(k)
            rank_prev = r
        if r >= n:
            break
    if not growth_steps:
        return 0, 0
    nu_max = growth_steps[-1]
    nu_gap = growth_steps[-1] - growth_steps[0]
    return int(nu_max), int(nu_gap)


# --- Legacy shims replacing pyident/sys_utils.py ---

def c2d(A: np.ndarray, B: np.ndarray, dt: float):
    """
    Backward-compatible alias for ZOH discretization.
    Old name from sys_utils.c2d -> now calls cont2discrete_zoh.
    Returns (Ad, Bd).
    """
    return cont2discrete_zoh(A, B, dt)


def simulate(T: int, x0: np.ndarray, Ad: np.ndarray, Bd: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Deterministic DT simulation (legacy signature from sys_utils.simulate):
        X[:,0] = x0
        X[:,t+1] = Ad @ X[:,t] + Bd @ u_t
    Accepts u as either shape (m, T) or (T, m). Returns X of shape (n, T+1).
    """
    if u.ndim != 2:
        raise ValueError(f"u must be 2D, got shape {u.shape}")
    m = Bd.shape[1]
    if u.shape == (m, T):      # (m, T)
        U = u.T
    elif u.shape == (T, m):    # (T, m)
        U = u
    else:
        raise ValueError(f"Incompatible u shape {u.shape} for B shape {Bd.shape} and T={T}")

    n = Ad.shape[0]
    X = np.zeros((n, T + 1), dtype=Ad.dtype)
    X[:, 0] = x0
    for t in range(T):
        X[:, t + 1] = Ad @ X[:, t] + Bd @ U[t, :]
    return X


# --- NEW: regressor diagnostics -------------------------------------
def regressor_stats(X0: np.ndarray, U: np.ndarray, rtol_rank: float = 1e-12) -> dict:
    """Rank/cond of Z=[X0;U]."""
    Z = np.vstack([X0, U])
    s = npl.svd(Z, compute_uv=False)
    if s.size == 0:
        return {"rank": 0, "cond": np.inf, "smin": 0.0}
    rank = int(np.sum(s > rtol_rank * s[0]))
    cond = float(s[0] / (s[-1] + 1e-18))
    return {"rank": rank, "cond": cond, "smin": float(s[-1])}

# --- NEW: DT theoretical-class checker (relative thresholds) ----------
def same_equiv_class_dt_rel(Ad, Bd, Ahat, Bhat, x0,
                            rtol_eq: float = 1e-2,
                            rtol_rank: float = 1e-12,
                            use_leak: bool = True):
    # robust V basis
    K = unified_generator(Ad, Bd, x0, mode="unrestricted")
    if K.size == 0:
        return False, {"dim_V": 0}
    U, s, _ = npl.svd(K, full_matrices=False)
    k = int(np.sum(s > rtol_rank * s[0]))
    P = U[:, :k]; I = np.eye(Ad.shape[0])

    dA_V = float(npl.norm(P.T @ (Ahat - Ad) @ P, "fro"))
    leak = float(npl.norm((I - P @ P.T) @ Ahat @ P, "fro"))
    dB   = float(npl.norm(Bhat - Bd, "fro"))

    thrA = rtol_eq * max(1.0, npl.norm(P.T @ Ad @ P, "fro"))
    thrB = rtol_eq * max(1.0, npl.norm(Bd, "fro"))
    thrL = rtol_eq * max(1.0, npl.norm(Ad, "fro"))

    ok = (dA_V <= thrA) and (dB <= thrB) and ( (not use_leak) or (leak <= thrL) )
    info = {"dim_V": int(k), "dA_V": dA_V, "leak": leak, "dB": dB,
            "thrA": thrA, "thrB": thrB, "thrLeak": thrL}
    return ok, info

# --- NEW: data-equivalence residual ----------------------------------
def data_equivalence_residual(X0, X1, U, Ahat, Bhat, rtol: float = 1e-10):
    R = X1 - Ahat @ X0 - Bhat @ U
    rel = float(npl.norm(R, "fro") / (npl.norm(X1, "fro") + 1e-18))
    return (rel <= rtol), {"resid_rel": rel}

# --- NEW: DT Markov-parameter match ----------------------------------
def markov_params_dt(Ad, Bd, kmax: int):
    M = []
    Ak = np.eye(Ad.shape[0])
    for _ in range(kmax):
        M.append(Ak @ Bd)
        Ak = Ad @ Ak
    return M

def markov_match_dt(Ad, Bd, Ahat, Bhat, kmax: int, rtol: float = 1e-2):
    M  = markov_params_dt(Ad, Bd, kmax)
    Mh = markov_params_dt(Ahat, Bhat, kmax)
    errs = [float(npl.norm(Mh[k]-M[k],"fro") / (npl.norm(M[k],"fro")+1e-12)) for k in range(kmax)]
    ok = (max(errs) <= rtol)
    return ok, {"markov_errs": errs, "markov_err_max": float(max(errs))}

