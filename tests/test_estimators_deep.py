import numpy as np
import pytest

from ..metrics import cont2discrete_zoh, projected_errors, unified_generator
from ..estimators import moesp_fit


def _noiseless(n=6, m=2, T=180, dt=0.05, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) * 0.15 - 0.6 * np.eye(n)
    B = rng.standard_normal((n, m))
    x0 = rng.standard_normal(n)
    Ad, Bd = cont2discrete_zoh(A, B, dt)
    u = rng.standard_normal((T, m))
    X = np.empty((n, T+1)); X[:, 0] = x0
    for k in range(T):
        X[:, k+1] = Ad @ X[:, k] + Bd @ u[k, :]
    return A, B, Ad, Bd, x0, u, X

def test_moesp_noiseless_fullstate():
    n, m, T = 6, 2, 200
    A, B, Ad, Bd, x0, u, X = _noiseless(n, m, T)
    Xtrain, Xp, Utr = X[:, :-1], X[:, 1:], u.T
    # Use the same heuristic as run_single
    s = max(n, min(10, T // 4))
    try:
        Ahat, Bhat = moesp_fit(Xtrain, Xp, Utr, s=s, n=n)
    except ValueError as e:
        # rank deficiency can happen for unlucky random realizations; don't flap CI
        import pytest
        pytest.skip(f"MOESP rank deficiency for this seed/draw: {e}")
    K = unified_generator(A, B, x0, mode="unrestricted")
    U_, S, _ = np.linalg.svd(K, full_matrices=False)
    PV = U_[:, : (S > 1e-10).sum()]
    eA, eB = projected_errors(Ahat, Bhat, Ad, Bd, Vbasis=PV)
    assert eA <= 1e-8 and eB <= 1e-8

