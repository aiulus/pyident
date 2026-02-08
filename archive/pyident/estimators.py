"""Minimal estimator set for the ICLR pipeline (DMDc only)."""
from __future__ import annotations

from typing import Tuple
import numpy as np


def _ensure_channel_major(U: np.ndarray, expected_T: int) -> np.ndarray:
    """Return U with shape (m, T); accept time-major (T, m)."""
    if U.ndim != 2:
        raise ValueError(f"U must be 2-D, got shape {U.shape}.")
    if U.shape[1] == expected_T:
        return U
    if U.shape[0] == expected_T:
        return U.T
    raise ValueError(
        f"Unable to align U of shape {U.shape} with expected T={expected_T}."
    )


def dmdc_tls(
    X: np.ndarray,
    Xp: np.ndarray,
    U: np.ndarray,
    rcond: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Row-wise scaled TLS for Xp ≈ Θ [X;U].
    Column-scale the augmented matrix to improve conditioning; fallback to ridge-OLS if needed.
    """
    n, T = X.shape
    U_cm = _ensure_channel_major(U, T)
    Z = np.vstack([X, U_cm]).T
    Y = Xp.T

    Theta = np.zeros((n, Z.shape[1]), dtype=X.dtype)
    for j in range(n):
        M = np.hstack([Z, Y[:, [j]]])
        scales = np.linalg.norm(M, axis=0)
        scales[scales == 0.0] = 1.0
        Ms = M / scales

        _, _, Vt = np.linalg.svd(Ms, full_matrices=False)
        v = Vt[-1, :] / scales

        denom = v[-1]
        if abs(denom) < 1e-12:
            lam = 1e-6
            G = (Z.T @ Z) + lam * np.eye(Z.shape[1])
            theta_j = np.linalg.solve(G, Z.T @ Y[:, j])
        else:
            theta_j = -v[:-1] / denom
        Theta[j, :] = theta_j

    Ahat = Theta[:, :n]
    Bhat = Theta[:, n:]
    return Ahat, Bhat
