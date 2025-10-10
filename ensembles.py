import warnings
from typing import Tuple, Literal, Optional, Dict, Any, Callable

import numpy as np
import numpy.linalg as npl


# ---------------------------------------------------------------------
# Random-matrix factories for (A, B).
# All generators return numpy arrays with shapes:
#   A: (n, n),  B: (n, m)
# ---------------------------------------------------------------------


def ginibre(n: int, m: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    A = rng.normal(size=(n, n)) / np.sqrt(n)
    B = rng.normal(size=(n, m)) / np.sqrt(n)
    return A, B


def binary(n: int, m: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Binary ensemble.This may induce strong degeneracies (e.g., repeated singular values).
      Prefer rademacher() if you need symmetric +/- 1."""
    A = rng.choice([0.0, 1.0], size=(n, n))
    B = rng.choice([0.0, 1.0], size=(n, m))
    return A, B


def rademacher(n: int, m: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    A = rng.choice([-1.0, 1.0], size=(n, n))
    B = rng.choice([-1.0, 1.0], size=(n, m))
    return A, B


def sparse_continuous_column(
    n: int,
    rng: np.random.Generator,
    p_density: float = 0.5,
) -> np.ndarray:
    
    v = rng.standard_normal((n, 1))
    m = rng.binomial(1, p_density, size=(n, 1)).astype(float)
    return v * m


def sparse_continuous(
    n: int,
    m: int,
    rng: np.random.Generator,
    which: Literal["A", "B", "both"] = "both",
    p_density: float = 0.3,
    p_density_A: Optional[float] = None,
    p_density_B: Optional[float] = None,
    check_zero_rows: bool = False,
    max_attempts: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    p_density_A = p_density if p_density_A is None else p_density_A
    p_density_B = p_density if p_density_B is None else p_density_B

    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, m))

    A_mask = rng.binomial(1, p_density_A, size=(n, n)).astype(float)
    B_mask = rng.binomial(1, p_density_B, size=(n, m)).astype(float)

    if which in ("A", "both"):
        A *= A_mask
    if which in ("B", "both"):
        B *= B_mask

    if check_zero_rows and which in ("B", "both"):
        z = (np.linalg.norm(B, axis=1) == 0)
        tries = 0
        while np.any(z) and tries < max_attempts:
            B[z, :] = rng.standard_normal((z.sum(), m)) \
                      * rng.binomial(1, p_density_B, size=(z.sum(), m))
            z = (np.linalg.norm(B, axis=1) == 0)
            tries += 1

    return A, B



def stable(n: int, m: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw (A,B) with A Hurwitz (CT-stable).
    Strategy: draw M ~ N(0,1), then shift by (alpha + max Re lambda(M))I
    """
    alpha = 0.10  # margin
    M = rng.standard_normal((n, n))
    lam = np.linalg.eigvals(M)
    shift = float(max(0.0, np.max(np.real(lam))) + alpha)
    A = M - shift * np.eye(n)
    B = rng.standard_normal((n, m))
    return A, B


def stable_continuous(n, m, rng, lam_min=0.2, lam_max=1.5):
    """
    Strictly Hurwitz A with a spectral margin (all eigenvalues <= -lam_min), and random B.
    """
    # Orthonormal basis
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    # Draw negative eigenvalues with margin
    lam = lam_min + rng.uniform(0.0, lam_max - lam_min, size=n)
    A = - (Q @ np.diag(lam) @ Q.T)   # symmetric negative-definite ⇒ Hurwitz with margin >= lam_min
    # B: standard normal
    B = rng.standard_normal((n, m))
    return A, B

# ---------------------------------------------------------------------
# Convenience adapter used by run_single / CLI
# ---------------------------------------------------------------------
def sample_system_instance(cfg, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch by cfg.ensemble with cfg’s density fields.

    Supported:
    - 'ginibre', 'binary', 'stable', 'sparse'
    """
    if cfg.ensemble == "ginibre":
        return ginibre(cfg.n, cfg.m, rng)
    if cfg.ensemble == "binary":
        return binary(cfg.n, cfg.m, rng)
    if cfg.ensemble == "sparse":
        return sparse_continuous(
            n=cfg.n, m=cfg.m, rng=rng,
            which=getattr(cfg, "sparse_which", "both"),
            p_density=getattr(cfg, "p_density", 0.5),
            p_density_A=getattr(cfg, "_density_A", None),
            p_density_B=getattr(cfg, "_density_B", None),
            check_zero_rows=getattr(cfg, "check_zero_rows", False),
        )
    if cfg.ensemble == "stable":
        return stable(cfg.n, cfg.m, rng)
    raise ValueError(f"unknown ensemble: {cfg.ensemble}")


def draw_initial_state(n: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Initial condition sampler."""
    if mode == "gaussian":
        x0 = rng.standard_normal(n)
    elif mode == "rademacher":
        x0 = rng.choice([-1.0, 1.0], size=n)
    elif mode == "ones":
        x0 = np.ones(n)
    elif mode == "zero":
        x0 = np.zeros(n)
    else:
        raise ValueError(f"Unknown x0_mode={mode}")
    return x0

# ---------------------------------------------------------------------
# New: controllability tools + (A,B) construction with target rank
# ---------------------------------------------------------------------

def _ctrb_matrix(A: np.ndarray, B: np.ndarray, order: Optional[int] = None) -> np.ndarray:
    """
    Controllability matrix C = [B, AB, A^2 B, ..., A^{order-1} B].
    If order is None, uses n = A.shape[0].
    """
    n = int(A.shape[0])
    m = int(B.shape[1])
    ordr: int = n if order is None else int(order)

    C = np.empty((n, m * ordr), dtype=A.dtype)
    Ak = np.eye(n, dtype=A.dtype)
    for k in range(ordr):
        C[:, k*m:(k+1)*m] = Ak @ B
        Ak = A @ Ak
    return C


def _svd_rank(M, rtol=None, atol=None):
    s = np.linalg.svd(M, compute_uv=False)
    smax = float(s[0]) if s.size else 0.0
    thr = 0.0
    if atol is None:
        atol = max(M.shape)*np.finfo(float).eps*smax
    thr = atol
    if rtol is not None:
        thr = max(thr, float(rtol)*smax)
    return int(np.sum(s > thr))



def controllability_rank(A: np.ndarray, B: np.ndarray, order: Optional[int] = None, rtol: Optional[float] = None) -> Tuple[int, np.ndarray]:
    """
    Return (rank, C) where C is the controllability matrix and rank = rank(C).
    """
    C = _ctrb_matrix(A, B, order=order)
    return _svd_rank(C, rtol=rtol), C


def _call_with_filtered(fn: Callable, /, **kwargs):
    """Call fn with only kwargs it accepts (lets us forward base generator kwargs safely)."""
    import inspect
    sig = inspect.signature(fn)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


def _draw_base_pair(base: str, n: int, m: int, rng: np.random.Generator, **base_kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw (A,B) using one of the existing base generators in this module.
    """
    base = base.lower()
    if base == "ginibre":
        return ginibre(n, m, rng)
    if base == "stable":
        return stable(n, m, rng)
    if base == "binary":
        return binary(n, m, rng)
    if base in {"sparse", "sparse_continuous"}:
        kw = _filter_kwargs(sparse_continuous, base_kwargs)
        return sparse_continuous(n=n, m=m, rng=rng, **kw)
    raise ValueError(f"unknown base generator '{base}'")


def draw_with_ctrb_rank(
    n: int,
    m: int,
    r: int,
    rng: np.random.Generator,
    *,
    ensemble_type: str = "ginibre",
    max_tries: int = 50,
    embed_random_basis: bool = True,
    base_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Construct (A,B) with *exact* controllability rank r in dimension n with m inputs.

    Theory (Kalman decomposition):
      We build a block pair (A_blk, B_blk) in a basis [R, U] where
      - dim(R) = r, (A_c, B_c) is controllable on R,
      - dim(U) = n-r, B has no support on U (=> unreachable),
      then embed with a random orthonormal Q to avoid axis artifacts.

    Parameters
    ----------
    n, m : state and input dimensions
    r    : desired controllability rank (0 <= r <= n)
    rng  : NumPy Generator for reproducibility
    base_c : base generator for the controllable block (ginibre/stable/binary/sparse)
    max_tries : how many attempts to draw a controllable (A_c,B_c) of size rxm
    embed_random_basis : if True, embed blocks with a random orthonormal matrix Q
    base_kwargs_c / base_kwargs_u : forwarded to the respective base generators

    Returns
    -------
    A, B, meta
      meta includes:
        - "Q": the embedding basis (np.eye(n) if embed_random_basis=False)
        - "Ar": A_c, "Au": A_u, "Br": B_c
        - "R_basis": the embedded reachable basis Q[:, :r]
        - "rank": verified controllability rank (== r)
    """
    if not (0 <= r <= n):
        raise ValueError(f"r must be in [0, n]. Got r={r}, n={n}")
    if m < 1:
        raise ValueError("m must be ≥ 1")

    base_kwargs = base_kwargs or {}

    # (1) build controllable block (A_c,B_c) of size r (or empty if r=0)
    if r == 0:
        A_c = np.zeros((0, 0))
        B_c = np.zeros((0, m))
    else:
        tries = 0
        while True:
            tries += 1
            A_c, B_c = _draw_base_pair(ensemble_type, r, m, rng, **base_kwargs)
            rk, _ = controllability_rank(A_c, B_c, order=r)
            if rk == r:
                break
            if tries >= max_tries:
                raise RuntimeError(f"Could not draw controllable block of size r={r} from base '{ensemble_type}' after {max_tries} tries.")

    # (2) build unreachable block A_u of size n-r (or empty if r=n)
    d = n - r
    if d == 0:
        A_u = np.zeros((0, 0))
    else:
        A_u, _Bu_dummy = _draw_base_pair(ensemble_type, d, m, rng, **base_kwargs)

    # (3) assemble block pair in canonical coordinates
    if r == 0 and d == 0:
        raise ValueError("n=0 is not supported.")
    A_blk = np.block([
        [A_c,                   np.zeros((r, d))],
        [np.zeros((d, r)),      A_u           ],
    ])
    B_blk = np.vstack([B_c, np.zeros((d, m))])  # no actuation on U

    # (4) embed with random orthonormal basis (optional)
    Q = np.eye(n)
    if embed_random_basis:
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))  # Haar orthonormal

    A = Q @ A_blk @ Q.T
    B = Q @ B_blk

    # (5) verify controllability rank exactly r (use full order=n and robust rtol)
    rtols = (1e-10, 1e-8, 1e-6, 1e-12, 1e-14)
    rk_final = None
    for rtol in rtols:
        rk_final, R_basis_num = controllability_rank(A, B, order=n, rtol=rtol)
        if rk_final == r:
            break
    if rk_final != r:
        raise RuntimeError(
            f"Target controllability rank r={r} not achieved after embedding "
            f"(got {rk_final}). Try a looser rtol or re-draw."
        )

    meta = {
        "Q": Q, "Ar": A_c, "Au": A_u, "Br": B_c,
        "R_basis": R_basis_num,  # numeric reachable basis (orthonormal)
        "rank": rk_final,
    }

    return A, B, meta


# ---------------------------------------------------------------------
# New: "good vs. bad" initial states for uncontrollable pairs
# ---------------------------------------------------------------------
def initial_state_classifier(
    A: np.ndarray,
    B: np.ndarray,
    *,
    rtol: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Characterize initial states w.r.t. V(x0) (visible subspace via unified generator).
    For an uncontrollable pair (A,B) with rank r < n, write X = R ⊕ U where
    R = reachable subspace (dim r). Then V(x0) = R ⊕ K_U(x0_U),
    where x0_U is x0 projected onto U and K_U is the Krylov span under A|_U.

    'Good' initial states are those with rank K_U(x0_U) = dim(U) (i.e., x0 is
    a cyclic vector for A|_U), which makes V(x0) = ℝⁿ. Others are 'bad'.

    Returns a dictionary with:
      - "rank": controllability rank r
      - "R_basis": orthonormal basis of R (nxr)
      - "U_basis": orthonormal complement basis (nx(n-r))
      - "is_good": callable x0 -> bool
      - "is_bad":  callable x0 -> bool
      - "sample_good":  callable rng -> x0 (tries until good)
      - "sample_bad":   callable rng -> x0 (constructs via eigenvector in U if possible)
    """
    n = A.shape[0]
    r, C = controllability_rank(A, B, order=n, rtol=rtol)

    # SVD gives orthonormal bases: left singular vectors of C
    U_left, svals, _ = np.linalg.svd(C, full_matrices=True)
    R_basis = U_left[:, :r]                      # reachable subspace basis
    U_basis = U_left[:, r:] if r < n else np.zeros((n, 0))   # complement

    # Projected dynamics on U (note: R is A-invariant; complement used here is fine)
    def _restrict_to_U(M: np.ndarray) -> np.ndarray:
        if U_basis.shape[1] == 0:
            return np.zeros((0, 0), dtype=M.dtype)
        return U_basis.T @ M @ U_basis

    A_U = _restrict_to_U(A)
    dimU = A_U.shape[0]

    def _krylov_rank_on_U(x0: np.ndarray) -> int:
        if dimU == 0:
            return 0
        xU = U_basis.T @ x0
        # Build Krylov [xU, A_U xU, ..., A_U^{dimU-1} xU]
        K = np.empty((dimU, dimU), dtype=A.dtype)
        vk = xU
        for k in range(dimU):
            K[:, k] = vk
            vk = A_U @ vk
        return _svd_rank(K, rtol=rtol)

    def is_good(x0: np.ndarray) -> bool:
        """True iff V(x0) = ℝⁿ (i.e., Krylov on U has full rank dimU)."""
        return _krylov_rank_on_U(x0) == dimU

    def is_bad(x0: np.ndarray) -> bool:
        return not is_good(x0)

    def sample_good(rng: np.random.Generator) -> np.ndarray:
        """Draw a random x0 until it is 'good' (almost sure in continuous draws)."""
        if dimU == 0:
            # Everything is trivially good if the pair is controllable already.
            return rng.standard_normal(n)
        for _ in range(100):
            # Mix reachable + unreachable components to avoid degeneracies
            x = R_basis @ rng.standard_normal(r) + U_basis @ rng.standard_normal(dimU)
            if is_good(x):
                return x
        # Extremely unlikely fallback
        return R_basis @ rng.standard_normal(r) + U_basis @ rng.standard_normal(dimU)

    def sample_bad(rng: np.random.Generator) -> np.ndarray:
        """
        Construct a 'bad' x0 by taking an eigenvector of A|_U (if available),
        which yields Krylov rank 1 on U; else degenerately choose a basis vector.
        """
        if dimU == 0:
            return np.zeros(n)  # no 'bad' states if already controllable
        try:
            w, V = np.linalg.eig(A_U)
            # pick an eigenvector with largest magnitude to improve conditioning
            idx = int(np.argmax(np.abs(w)))
            v = np.real_if_close(V[:, idx])
            v = v / (np.linalg.norm(v) + 1e-12)
            xU = v
        except Exception:
            # fallback: a coordinate axis in U
            e = np.zeros(dimU); e[0] = 1.0
            xU = e
        xR = np.zeros(r)
        return R_basis @ xR + U_basis @ xU

    return {
        "rank": r,
        "R_basis": R_basis,
        "U_basis": U_basis,
        "is_good": is_good,
        "is_bad": is_bad,
        "sample_good": sample_good,
        "sample_bad": sample_bad,
    }

def _filter_kwargs(fn, kwargs):
    import inspect
    sig = inspect.signature(fn)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}