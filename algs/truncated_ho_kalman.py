"""
Truncated Ho–Kalman (THK) — Subspace variant for single-trajectory data with C = I, D = 0 (noise-free ideally).

Given a single input/state trajectory (u_t, x_t), this routine estimates the reduced-order
realization (Φ_k, Γ_k) on the data-visible subspace V with dim(V) = k ≤ n, together with the
embedding T_V : R^k → R^n (so that x ≈ T_V z). It also returns a minimality certificate
based on SVD ranks and residuals.

Assumptions
-----------
- Discrete-time LTI: x_{t+1} = Φ x_t + Γ u_t, y_t = x_t (C = I_n, D = 0).
- One sufficiently long, persistently exciting trajectory (ideally noise-free).
- Window sizes p, f ≥ k. If k is unknown, choose moderately large p,f and obtain k from a rank test.

Outputs
-------
- Φ_k (k×k), Γ_k (k×m), T_V (n×k): reduced dynamics and embedding.
- info: dict containing SVD singular values, estimated k, numerical tolerances, and residuals.

Notes
-----
- If you want continuous time, map back via A_k = (1/Δ) * logm(Φ_k) with appropriate branch handling.
- For noisy data, the same code works but you will need regularization (e.g., Tikhonov in pinv) and
  careful thresholding for rank decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np


# ---------- utilities ----------

def block_hankel(seq: np.ndarray, block_rows: int, cols: int) -> np.ndarray:
    """Build a block Hankel matrix from a sequence of row-vectors.

    Parameters
    ----------
    seq : array, shape (T, d)
        Sequence s_0, ..., s_{T-1} with each s_t ∈ R^d stored as rows.
    block_rows : int
        Number of block rows (e.g., p for past, f for future).
    cols : int
        Number of columns N; requires T >= block_rows + cols - 1.

    Returns
    -------
    H : array, shape (d*block_rows, cols)
        H[:, j] = [ s_j; s_{j+1}; ...; s_{j+block_rows-1} ].
    """
    T, d = seq.shape
    assert T >= block_rows + cols - 1, "Not enough samples for requested Hankel shape."
    H = np.empty((d * block_rows, cols), dtype=seq.dtype)
    for j in range(cols):
        blk = seq[j : j + block_rows, :]  # (block_rows, d)
        H[:, j] = blk.reshape(-1, order="C")
    return H


def hankel_past_future(x: np.ndarray, u: np.ndarray, p: int, f: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct past/future Hankels for state and input.

    x : (T+1, n) states at t=0..T
    u : (T,   m) inputs at t=0..T-1
    Returns X_p (np x N), X_f (nf x N), U_p (mp x N), U_f (mf x N)
    with N = T - (p+f) + 1.
    """
    T = u.shape[0]
    n = x.shape[1]
    m = u.shape[1]
    assert x.shape[0] == T + 1, "x must have one more sample than u."
    N = T - (p + f) + 1
    assert N >= 1, "Trajectory too short for chosen (p,f)."
    Xp = block_hankel(x[:-1, :], block_rows=p, cols=N)    # uses x_0..x_{T-1}
    Xf = block_hankel(x[p:  p + f + N - 1, :], block_rows=f, cols=N)  # starts at x_p
    Up = block_hankel(u,             block_rows=p, cols=N)
    Uf = block_hankel(u[p:, :],      block_rows=f, cols=N)
    # Sanity check on alignment: the first block of X_f should be x_{p..p+N-1}
    first_future = x[p : p + N, :]
    assert np.allclose(Xf[:n, :].T, first_future), "Future-state alignment failed."
    return Xp, Xf, Up, Uf


def numerical_rank(svals: np.ndarray, tol: Optional[float] = None) -> Tuple[int, float]:
    """Return numerical rank and the tolerance used given singular values s (descending)."""
    if tol is None:
        # default per Golub–Van Loan style heuristic
        eps = np.finfo(float).eps
        tol = max(svals.shape) * eps * (svals[0] if svals.size else 1.0)
    r = int(np.sum(svals > tol))
    return r, tol


@dataclass
class THKResult:
    Phi_k: np.ndarray  # (k, k)
    Gamma_k: np.ndarray  # (k, m)
    T_V: np.ndarray   # (n, k)
    info: Dict[str, object]


# ---------- main algorithm ----------

def truncated_ho_kalman_subspace(
    x: np.ndarray,
    u: np.ndarray,
    p: Optional[int] = None,
    f: Optional[int] = None,
    k: Optional[int] = None,
    rank_tol: Optional[float] = None,
) -> THKResult:
    """
    Truncated Ho–Kalman (subspace variant) for single-trajectory data with C = I.

    Parameters
    ----------
    x : array, shape (T+1, n)
        State sequence x_0..x_T. (Assumed measured, i.e., C = I.)
    u : array, shape (T, m)
        Input sequence u_0..u_{T-1}.
    p, f : int, optional
        Past/future window sizes. If None, they are chosen automatically (balanced, moderately large).
    k : int, optional
        Target model order (= dim V). If None, estimated from the numerical rank of X_f.
    rank_tol : float, optional
        Tolerance for numerical rank; if None, a heuristic is used.

    Returns
    -------
    THKResult with (Φ_k, Γ_k, T_V) and diagnostics in .info
    """
    T = u.shape[0]
    n = x.shape[1]
    m = u.shape[1]

    # 0) Choose windows if not provided (aim for p ≈ f ≈ T/3, but at least 2 and small multiples of n)
    if p is None or f is None:
        base = max(2, min(10, T // 4))
        p = p or base
        f = f or base

    # 1) Build Hankels
    Xp, Xf, Up, Uf = hankel_past_future(x, u, p=p, f=f)
    nf, N = Xf.shape
    assert nf % n == 0, "Future Hankel must be multiple of state dimension."
    f_blocks = nf // n

    # 2) SVD of future state Hankel; estimate k if needed
    U_svd, svals, Vt_svd = np.linalg.svd(Xf, full_matrices=False)
    k_est, tol_used = numerical_rank(svals, rank_tol)
    if k is None:
        k = k_est
    if k < 1:
        raise ValueError("Estimated rank k < 1; data may be degenerate. Increase p,f or ensure PE input.")

    Uk = U_svd[:, :k]
    Sk = svals[:k]
    Vk = Vt_svd[:k, :]

    # 3) Construct an extended observability-like factor O_f and a state-sequence-like factor S_k
    #    X_f ≈ O_f S_k with O_f ∈ R^{(n f)×k}. Use symmetric square-root split.
    sqrtSk = np.sqrt(Sk)
    Of = Uk @ np.diag(sqrtSk)                # (n f) x k
    Sk_state = (np.diag(sqrtSk) @ Vk).astype(x.dtype)  # k x N

    # 4) Shift-invariance on Of to get Φ_k
    Of_up = Of[:-n, :]        # drop last state block
    Of_dn = Of[n:,  :]        # drop first state block
    Phi_k = np.linalg.pinv(Of_up) @ Of_dn   # exact in noiseless data

    # 5) Extract T_V as the top state block of Of (n x k)
    T_V = Of[:n, :]

    # 6) Compute reduced-state trajectory z_j at the start of each future window (time t0 = p + j)
    #    Solve T_V z_j ≈ x_{t0} (least squares). This ties the internal coordinate to physical states.
    X0 = Xf[:n, :]                 # first state block: x_{p..p+N-1} ∈ R^{n×N}
    Z = np.linalg.pinv(T_V) @ X0   # k×N

    # 7) Estimate Γ_k using the reduced-state dynamics z_{j+1} = Φ_k z_j + Γ_k u_{t0}
    #    Assemble LS: E = Z_next - Φ_k Z; solve E ≈ Γ_k U0
    Z_next = Z[:, 1:]
    Z_curr = Z[:, :-1]
    U0 = (u[p : p + N - 1, :]).T     # m×(N-1), aligned with starts t0 = p..p+N-2
    E = Z_next - Phi_k @ Z_curr
    # Solve E = Γ_k U0 in LS sense
    Gamma_k = (E @ U0.T) @ np.linalg.pinv(U0 @ U0.T)

    # 8) Diagnostics / certificates
    # Residual for shift-invariance
    shift_res = np.linalg.norm(Of_dn - Of_up @ Phi_k, ord='fro') / max(1e-12, np.linalg.norm(Of_dn, ord='fro'))
    # State reconstruction residual
    rec_res = np.linalg.norm(X0 - T_V @ Z, ord='fro') / max(1e-12, np.linalg.norm(X0, ord='fro'))
    # Dynamic residual (reduced coordinates)
    dyn_res = np.linalg.norm(E - Gamma_k @ U0, ord='fro') / max(1e-12, np.linalg.norm(Z_next, ord='fro'))

    # SVD gap diagnostic (how clean is the truncation boundary?)
    sv_gap = float('inf')
    if len(svals) > k:
        sv_gap = (Sk[-1] / max(1e-16, svals[k]))

    info = dict(
        p=p, f=f, N=N, n=n, m=m,
        singular_values=svals,
        k_est=k_est, k_used=k, rank_tol=tol_used,
        sv_gap=sv_gap,
        shift_residual=shift_res,
        state_reconstruction_residual=rec_res,
        dynamic_residual=dyn_res,
    )

    return THKResult(Phi_k=Phi_k, Gamma_k=Gamma_k, T_V=T_V, info=info)


# ---------- simple test harness (synthetic noiseless example) ----------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # True system (n=6, m=2); build Φ with a visible 3D subspace V and arbitrary V^⊥
    n, m, k_true = 6, 2, 3
    # Construct a random stable Φ
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    eigs = -0.1 - 0.05 * np.arange(n)  # continuous-time eigenvalues
    A = Q @ np.diag(eigs) @ np.linalg.inv(Q)
    dt = 0.1
    # Discretize
    Phi = np.eye(n) + dt * A  # crude but OK for small dt (to avoid SciPy dependency)
    Gamma = rng.standard_normal((n, m)) * 0.5

    # Choose x0 and inputs to make dim(V)=k_true by constraining excitation
    x0 = np.zeros((n,))
    # Build B so that rank of [x0, B] visible Krylov is k_true
    # Force visibility only in first k_true coordinates
    S = np.zeros((n, k_true))
    S[:k_true, :] = np.eye(k_true)
    # Project dynamics to make only these coordinates excited
    Phi = S @ rng.standard_normal((k_true, k_true)) @ S.T + (np.eye(n) - S @ S.T) @ Phi @ (np.eye(n) - S @ S.T)
    Gamma = S @ rng.standard_normal((k_true, m))

    # Simulate single trajectory (u_t, x_t)
    T = 400
    u = rng.standard_normal((T, m))
    x = np.zeros((T + 1, n))
    for t in range(T):
        x[t + 1] = Phi @ x[t] + Gamma @ u[t]

    # Run THK with automatic k, p, f
    res = truncated_ho_kalman_subspace(x, u, p=20, f=20, k=None)

    print("Estimated k:", res.info["k_used"], "(k_est:", res.info["k_est"], ")")
    print("Singular values (first 8):", np.round(res.info["singular_values"][:8], 4))
    print("Shift residual:", res.info["shift_residual"])
    print("State recon residual:", res.info["state_reconstruction_residual"])
    print("Dynamic residual:", res.info["dynamic_residual"])
    print("Φ_k shape:", res.Phi_k.shape, "Γ_k shape:", res.Gamma_k.shape, "T_V shape:", res.T_V.shape)
