import numpy as np
from typing import Tuple, Optional


def dmdc_fit(
    X: np.ndarray,
    Xp: np.ndarray,
    U: np.ndarray,
    rcond: float = 1e-10,
    rank: Optional[int] = None,
    ridge: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discrete-time DMDc: solve Xp ≈ [A B] [X; U] in least squares.

    Shapes
    -------
    X   : (n, T-1)   state snapshots
    Xp  : (n, T-1)   next state snapshots
    U   : (m, T-1)   inputs aligned with columns of X

    Options (numerical relaxations)
    -------
    rank : if set, use truncated-SVD pseudoinverse of Z with this rank.
    ridge: if set, Tikhonov λ (solves (Z Z.T + lambda I)^{-1}).

    Returns
    -------
    Ahat : (n, n)
    Bhat : (n, m)
    """
    n = X.shape[0]
    Z = np.vstack([X, U])  # (n+m, T-1)

    if ridge is not None:
        # Note: ridge biases towards smaller coefficients but improves stability.
        ZZt = Z @ Z.T
        G = ZZt + (ridge * np.eye(ZZt.shape[0], dtype=ZZt.dtype))
        AB = (Xp @ Z.T) @ np.linalg.solve(G, np.eye(G.shape[0], dtype=G.dtype))
    elif rank is not None:
        # Note: truncated pseudoinverse discards small singular directions (noise suppression).
        Uz, Sz, Vtz = np.linalg.svd(Z, full_matrices=False)
        r = int(min(rank, np.sum(Sz > 0)))
        Sz_inv = np.diag(1.0 / Sz[:r])
        Z_pinv = (Vtz[:r, :].T) @ Sz_inv @ (Uz[:, :r].T)
        AB = Xp @ Z_pinv
    else:
        # Stable LS via lstsq on transposed system.
        AB_T, *_ = np.linalg.lstsq(Z.T, Xp.T, rcond=rcond)
        AB = AB_T.T

    Ahat = AB[:, :n]
    Bhat = AB[:, n:]
    return Ahat, Bhat


def dmdc_tls_fit(
    X: np.ndarray,
    Xp: np.ndarray,
    U: np.ndarray,
    *,
    rcond: float = 1e-12,
    energy: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TLS-DMDc (total least squares): solves Xp ≈ [A B][X;U] allowing perturbations
    in BOTH regressors and responses. Noise-robust vs LS when noise corrupts X, U, Xp.

    Method
    ------
    Form data matrix D = [Z.T  Y.T] with Z=[X;U], Y=Xp (all N columns are time samples).
    SVD: D = U Σ V.T; partition V = [[V11 V12],[V21 V22]] wrt (Z | Y) columns.
    TLS solution for Θ in Y ≈ Z Θ is Θ_TLS = - V12 V22^{-1}. Here Θ_TLS = [A B].T,
    so [A B] = Θ_TLS.T.

    Numerical notes (responsibility)
    --------------------------------
    - If V22 is (near) singular, TLS is ill-posed. We fall back to a damped inverse
      using np.linalg.lstsq with rcond, which regularizes the inversion of V22.
    - Optional 'energy' (0<energy<1) keeps only enough singular values of D to
      explain that fraction of spectral energy before forming the TLS solution.

    Returns
    -------
    Ahat : (n, n)
    Bhat : (n, m)
    """
    n = X.shape[0]
    Z = np.vstack([X, U])         
    Y = Xp                         
    N = Z.shape[1]
    D = np.hstack([Z.T, Y.T])     

    Ud, Sd, Vtd = np.linalg.svd(D, full_matrices=False)

    if energy is not None:
        # retain leading components covering requested energy
        cumsum = np.cumsum(Sd**2)
        total = cumsum[-1] if cumsum.size else 0.0
        k = np.searchsorted(cumsum, energy * total) + 1
        k = max(1, min(k, Sd.size))
        Vtd = Vtd[:k, :]
        # re-orthonormalize the retained right singular vectors
        # Note: this truncation changes TLS solution. Report if used.
        V, _ = np.linalg.qr(Vtd.T)
        Vt_used = V.T
    else:
        Vt_used = Vtd

    # Partition V (right singular vectors) along columns: first p=(n+m), last q=n
    p = Z.shape[0]
    q = Y.shape[0]
    V = Vt_used.T
    V11 = V[:p, :p]      # not used explicitly
    V12 = V[:p, p:p+q]
    V21 = V[p:p+q, :p]   # not used explicitly
    V22 = V[p:p+q, p:p+q]

    # Solve Θ = -V12 V22^{-1} in a numerically safe way
    # (if V22 ill-conditioned, use lstsq regularized by rcond).
    try:
        V22_inv = np.linalg.inv(V22)
    except np.linalg.LinAlgError:
        V22_inv, *_ = np.linalg.lstsq(V22, np.eye(V22.shape[0], dtype=V22.dtype), rcond=rcond)

    Theta = - V12 @ V22_inv         
    AB = Theta.T                    

    Ahat = AB[:, :n]
    Bhat = AB[:, n:]
    return Ahat, Bhat


def dmdc_iv_fit(
    X: np.ndarray,
    Xp: np.ndarray,
    U: np.ndarray,
    *,
    lag: int = 1,
    rcond: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    IV-DMDc (instrumental variables): Y ≈ Θ X with instruments W uncorrelated with measurement noise.
    Here Y=Xp, X=[X;U]. We choose W from *past* data (lag >= 1) to suppress bias.

    Construction
    -----------
    Let N = X.shape[1]. For lag L:
      Xc = X[:, L:],  Yc = Xp[:, L:],  W = [X[:, :-L]; U[:, :-L]]  (same column count N-L)
    IV estimate: Θ_IV = (Yc W.T) (Xc W.T)^+.

    Disclaimer
    -------------------------
    - Assumes additive measurement noise at time k is uncorrelated with instruments built from k−L.
    - Larger lag improves instrument validity but reduces effective sample size.
    - If (Xc W.T) is ill-conditioned, we use np.linalg.lstsq with rcond.

    Returns
    -------
    Ahat : (n, n)
    Bhat : (n, m)
    """
    n = X.shape[0]
    Z = np.vstack([X, U])    
    N = Z.shape[1]
    if lag < 1 or lag >= N:
        raise ValueError(f"lag must be in [1, N-1]. Got lag={lag}, N={N}.")

    Xc = Z[:, lag:]           
    Yc = Xp[:, lag:]          
    W  = np.vstack([X[:, :-lag], U[:, :-lag]])  

    # Instruments should have shape (k × (N-L)). Above, X has shape (n×N), U (m×N),
    # so W results in (n+m) × (N-L) as desired.
    assert W.shape[1] == Xc.shape[1]

    YW = Yc @ W.T            
    XW = Xc @ W.T           

    # Θ = YW @ (XW)^+
    # Solve via lstsq to avoid explicit pseudoinverse.
    Theta_T, *_ = np.linalg.lstsq(XW.T, YW.T, rcond=rcond)  # solves XW.T Θ.T ~~ YW.T
    AB = Theta_T.T

    Ahat = AB[:, :n]
    Bhat = AB[:, n:]
    return Ahat, Bhat
