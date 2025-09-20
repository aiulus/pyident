import numpy as np
from ..metrics import visible_subspace_basis, projector_from_basis

def _angles(P, Q):
    s = np.linalg.svd(P.T @ Q, full_matrices=False)[1]
    s = np.clip(s, 0.0, 1.0)
    return float(np.arccos(s).max()) if s.size else 0.0

def test_projector_properties():
    rng = np.random.default_rng(0)
    n, m = 6, 2
    A = rng.standard_normal((n, n)) * 0.1 - 0.6 * np.eye(n)
    B = rng.standard_normal((n, m))
    x0 = rng.standard_normal(n)
    V, k = visible_subspace_basis(A, B, x0)
    P = projector_from_basis(V)
    assert np.linalg.norm(P @ P - P) <= 1e-8
    assert np.linalg.norm(P - P.T) <= 1e-8

def test_similarity_invariance():
    rng = np.random.default_rng(1)
    n, m = 6, 2
    A = rng.standard_normal((n, n)) * 0.1 - 0.6 * np.eye(n)
    B = rng.standard_normal((n, m))
    x0 = rng.standard_normal(n)
    V, k = visible_subspace_basis(A, B, x0)

    # random invertible T via QR
    T, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A2 = T @ A @ np.linalg.inv(T)
    B2 = T @ B
    x2 = T @ x0
    V2, k2 = visible_subspace_basis(A2, B2, x2)

    # map back
    V2_back = np.linalg.inv(T) @ V2
    ang = _angles(V, V2_back)
    assert k2 == k and ang <= 1e-7
