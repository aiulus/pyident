import numpy as np
from ..metrics import visible_subspace_basis

def _angles(P, Q):
    s = np.linalg.svd(P.T @ Q, full_matrices=False)[1]
    s = np.clip(s, 0.0, 1.0)
    return float(np.arccos(s).max()) if s.size else 0.0

def test_subspace_invariant_under_scalar_scaling():
    rng = np.random.default_rng(0)
    n, m = 7, 3
    A = rng.standard_normal((n, n)) * 0.1 - 0.6 * np.eye(n)
    B = rng.standard_normal((n, m))
    x0 = rng.standard_normal(n)
    V1, k1 = visible_subspace_basis(A, B, x0)
    for s in (1e-2, 10.0, 1e2):
        V2, k2 = visible_subspace_basis(s * A, s * B, x0)
        ang = _angles(V1, V2)
        assert k2 == k1 and ang <= 1e-7
