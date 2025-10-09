import numpy as np
from .. import ensembles, estimators, metrics

def _simulate_dt(Ad, Bd, x0, U):
    # U: (m, T) -> X: (n, T+1)
    T = U.shape[1]
    n = Ad.shape[0]
    X = np.zeros((n, T+1))
    X[:,0] = x0
    for t in range(T):
        X[:,t+1] = Ad @ X[:,t] + Bd @ U[:,t]
    return X


def test_moesp_equals_one_step_ls_fullstate(rng):
    n,m = 6,2
    A,B = ensembles.ginibre(n, m, rng)  # DT-ish scale is fine, we simulate DT
    # Make it DT stable-ish for long sim
    # (not required for one-step fit, but keeps norms in check)
    A *= 0.8
    mU, T = m, 500
    U = rng.standard_normal((mU, T))
    x0 = rng.standard_normal(n)
    # Simulate DT
    X = np.zeros((n, T+1)); X[:,0]=x0
    for t in range(T): X[:,t+1]=A@X[:,t]+B@U[:,t]
    # Compare MOESP(full-state) vs DMDc pinv
    A1,B1 = estimators.moesp_fit(X[:,:-1], X[:,1:], U)
    A2,B2 = estimators.dmdc_pinv(X[:,:-1], X[:,1:], U)
    assert np.linalg.norm(A1-A2, "fro") <= 1e-10
    assert np.linalg.norm(B1-B2, "fro") <= 1e-10

from .. import estimators, metrics, ensembles
from .conftest import ensure_pe_on_V

def test_dmdc_exact_recovery_on_V_good_x0(rng):
    n,m,rk = 10, 2, 6
    A,B,_ = ensembles.draw_with_ctrb_rank(
        n=n, m=m, r=rk, rng=rng,
        base_c="stable", base_u="stable"  
    )

    x0 = ensembles.initial_state_classifier(A,B)["sample_good"](rng)
    Ad, Bd = metrics.cont2discrete_zoh(A, B, dt=0.05)

    T = 1200
    U = rng.standard_normal((m, T))
    X = _simulate_dt(Ad, Bd, x0, U)
    X0, X1 = X[:, :-1], X[:, 1:]
    # Ensure the PE-on-V hypothesis of the theorem holds
    P, k = ensure_pe_on_V(Ad, Bd, x0, X0, U, tol=1e-12)
    # Numerically gentle solve (tiny ridge), avoids pinv truncation artifacts
    Ahat, Bhat = estimators.dmdc_ridge(X0, X1, U, lam=1e-12)
    dA_V, dB_V = metrics.projected_errors(Ahat, Bhat, Ad, Bd, P)
    assert dA_V <= 1e-10
    assert dB_V <= 1e-10

from .. import estimators, metrics, ensembles
from .conftest import ensure_pe_on_V

def test_dmdc_exact_recovery_on_V_with_two_runs(rng):
    n,m,rk = 10, 2, 6
    A,B,_ = ensembles.draw_with_ctrb_rank(
        n=n, m=m, r=rk, rng=rng,
        base_c="stable", base_u="stable"
    )
    x0 = ensembles.initial_state_classifier(A,B)["sample_good"](rng)

    Ad, Bd = metrics.cont2discrete_zoh(A, B, dt=0.05)
    T = 400
    U1 = rng.standard_normal((m, T))
    U2 = rng.standard_normal((m, T))
    X1 = _simulate_dt(Ad, Bd, x0, U1)
    X2 = _simulate_dt(Ad, Bd, x0, U2)
    X01, X11 = X1[:, :-1], X1[:, 1:]
    X02, X12 = X2[:, :-1], X2[:, 1:]
 
    # Stack regressors
    Z  = np.hstack([np.vstack([X01, U1]), np.vstack([X02, U2])])
    Y  = np.hstack([X11, X12])

    # Check PE-on-V for the stacked experiment
    P, k = ensure_pe_on_V(Ad, Bd, x0,
                          X0=np.hstack([X01, X02]),
                          U=np.hstack([U1,  U2]),
                          tol=1e-12)
    
    # Solve (tiny ridge) for stability
    ZZt = Z @ Z.T
    lam = 1e-12
    Theta_hat = (Y @ Z.T) @ np.linalg.solve(ZZt + lam*np.eye(ZZt.shape[0]), np.eye(ZZt.shape[0]))
    Ahat, Bhat = Theta_hat[:, :n], Theta_hat[:, n:]
     
    dA_V, dB_V = metrics.projected_errors(Ahat, Bhat, Ad, Bd, P)
    assert dA_V <= 1e-7
    assert dB_V <= 1e-7