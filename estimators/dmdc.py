import numpy as np
from typing import Tuple, Optional

def dmdc_fit(
    X: np.ndarray, Xp: np.ndarray, U: np.ndarray,
    rcond: float = 1e-10, rank: Optional[int] = None, ridge: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discrete-time DMDc: solve Xp ≈ [A B] [X; U] in least-squares sense.

    Shapes
    ------
    X   : (n, T-1)   state snapshots
    Xp  : (n, T-1)   next state snapshots
    U   : (m, T-1)   inputs aligned with columns of X

    Options
    -------
    rank : if given, use SVD-truncated pseudoinverse with this rank.
    ridge: if given, Tikhonov regularization λ (on rows of Z).

    Returns
    -------
    Ahat : (n, n)
    Bhat : (n, m)
    """
    n = X.shape[0]
    Z = np.vstack([X, U])  # (n+m, T-1)

    if ridge is not None:
        # AB = (Xp Zᵀ) (Z Zᵀ + λ I)^{-1}
        ZZt = Z @ Z.T
        G = ZZt + (ridge * np.eye(ZZt.shape[0], dtype=ZZt.dtype))
        AB = (Xp @ Z.T) @ np.linalg.solve(G, np.eye(G.shape[0], dtype=G.dtype))
    elif rank is not None:
        # Truncated SVD pseudoinverse
        Uz, Sz, Vtz = np.linalg.svd(Z, full_matrices=False)
        r = min(rank, np.sum(Sz > 0))
        Sz_inv = np.diag(1.0 / Sz[:r])
        Z_pinv = (Vtz[:r, :].T) @ Sz_inv @ (Uz[:, :r].T)
        AB = Xp @ Z_pinv
    else:
        # Stable LS without forming the normal equations
        # (matrix-lstsq on transposes for speed & stability)
        AB_T, *_ = np.linalg.lstsq(Z.T, Xp.T, rcond=rcond)
        AB = AB_T.T

    Ahat = AB[:, :n]
    Bhat = AB[:, n:]
    return Ahat, Bhat
