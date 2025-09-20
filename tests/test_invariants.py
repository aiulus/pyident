import numpy as np
from ..metrics import cont2discrete_zoh, unified_generator, visible_subspace_basis, projector_from_basis

def _angles_between(P, Q):
    # principal angles via svd of P^T Q
    U, s, Vt = np.linalg.svd(P.T @ Q, full_matrices=False)
    # return maximum angle (in radians)
    s = np.clip(s, 0.0, 1.0)
    return float(np.arccos(s).max()) if s.size else 0.0

def test_visible_subspace_ct_dt_invariance():
    rng = np.random.default_rng(0)
    n,m = 6,3
    A = rng.standard_normal((n,n))*0.1 - 0.7*np.eye(n)
    B = rng.standard_normal((n,m))
    x0 = rng.standard_normal(n)
    Ad,Bd = cont2discrete_zoh(A,B,0.05)
    # CT V from K(A; x0,B)
    PV_ct, k_ct = visible_subspace_basis(A,B,x0,mode="unrestricted")
    # DT V from K(Ad; x0,Bd) — same theoretical subspace
    from ..metrics import unified_generator as ug
    K_dt = ug(Ad, Bd, x0, mode="unrestricted")  # (abuse API with DT matrices is fine here)
    U,S,_ = np.linalg.svd(K_dt, full_matrices=False)
    k_dt = (S > 1e-10).sum()
    PV_dt = U[:, :k_dt]
    ang = _angles_between(PV_ct, PV_dt)
    assert k_ct == k_dt and ang <= 1e-8

def test_equivalence_class_invariance_outside_V():
    rng = np.random.default_rng(1)
    n,m = 6,2
    A = rng.standard_normal((n,n))*0.1 - 0.6*np.eye(n)
    B = rng.standard_normal((n,m))
    x0 = rng.standard_normal(n)
    PV, k = visible_subspace_basis(A,B,x0)
    P = PV
    # Complete basis with Q for orthogonal complement
    Q,_ = np.linalg.qr(np.eye(n) - P@P.T)
    # Build A' = P A P^T + Q M Q^T with arbitrary M (change outside V)
    M = rng.standard_normal((n,n))
    M = (M + M.T)/2.0
    A_prime = P @ (P.T @ A @ P) @ P.T + Q @ M @ Q.T
    # V must be unchanged
    PV2, k2 = visible_subspace_basis(A_prime, B, x0)
    ang = _angles_between(PV, PV2)
    assert k2 == k and ang <= 1e-8
