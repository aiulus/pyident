import numpy as np
from .. import metrics

def test_markov_match_noiseless():
    # Simple DT system; Markov parameters match exactly
    rng = np.random.default_rng(7)
    n,m = 4,2
    A = rng.standard_normal((n,n))*0.8
    B = rng.standard_normal((n,m))
    kmax = 8
    ok, info = metrics.markov_match_dt(A,B,A,B,kmax=kmax, rtol=1e-12)
    assert ok and info["markov_err_max"] <= 1e-12

def test_data_equivalence_residual_noiseless():
    rng = np.random.default_rng(11)
    n,m,T = 5,2,200
    A = rng.standard_normal((n,n))*0.85
    B = rng.standard_normal((n,m))
    x0 = rng.standard_normal(n)
    U = rng.standard_normal((m,T))
    # simulate
    X = np.zeros((n,T+1)); X[:,0]=x0
    for t in range(T): X[:,t+1]=A@X[:,t]+B@U[:,t]
    ok, info = metrics.data_equivalence_residual(X[:,:-1], X[:,1:], U, A, B, rtol=1e-12)
    assert ok and info["resid_rel"] <= 1e-12

def test_pbh_structured_margin_zero_constructed_case():
    # Construct a PBH-violating case at λ=λ2: second row of [x0 B] is zero
    A = np.diag([1.0, 2.0, 3.0])
    B = np.array([[1.0, 0.0],
                  [0.0, 0.0],
                  [0.0, 1.0]])
    x0 = np.array([1.0, 0.0, 0.0])
    margin = metrics.pbh_margin_structured(A, B, x0)
    assert margin <= 1e-12

def test_equiv_class_dt_rel_on_visible_subspace():
    rng = np.random.default_rng(19)
    n, m = 6, 2
    A = rng.standard_normal((n, n)) * 0.8
    B = rng.standard_normal((n, m))
    x0 = rng.standard_normal(n)

    P, k = metrics.visible_subspace(A, B, x0)
    Pv = P @ P.T
    I = np.eye(n)

    # --- Case 1: modify only V^⊥ block (should still be OK)
    Ahat_ok = A + (I - Pv) @ rng.standard_normal((n, n)) @ (I - Pv)
    ok1, info1 = metrics.same_equiv_class_dt_rel(A, B, Ahat_ok, B, x0, rtol_eq=1e-2)
    assert ok1, f"unexpected fail: dA_V={info1['dA_V']} thrA={info1['thrA']}"

    # --- Case 2: modify the restriction on V (should FAIL)
    R = rng.standard_normal((n, n))
    E_onV = Pv @ R @ Pv
    # scale so projected error > threshold
    dA_unit = np.linalg.norm(P.T @ E_onV @ P, "fro") + 1e-12
    thrA = 1e-2 * max(1.0, np.linalg.norm(P.T @ A @ P, "fro"))
    Ahat_bad = A + (2.0 * thrA / dA_unit) * E_onV

    ok2, info2 = metrics.same_equiv_class_dt_rel(A, B, Ahat_bad, B, x0, rtol_eq=1e-2)
    assert (not ok2) and (info2["dA_V"] > info2["thrA"]), f"expected fail on V; dA_V={info2['dA_V']} thrA={info2['thrA']}"

