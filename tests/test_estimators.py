import numpy as np
import pytest
from ..metrics import cont2discrete_zoh, unified_generator, visible_subspace_basis, projected_errors
from ..estimators.dmdc import dmdc_fit, dmdc_tls_fit, dmdc_iv_fit

def _noiseless_system(n=5, m=2, T=120, dt=0.05, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n,n))*0.15 - 0.6*np.eye(n)
    B = rng.standard_normal((n,m))
    x0 = rng.standard_normal(n)
    Ad,Bd = cont2discrete_zoh(A,B,dt)
    u = rng.standard_normal((T,m))
    X = np.empty((n, T+1)); X[:,0] = x0
    for k in range(T):
        X[:,k+1] = Ad@X[:,k] + Bd@u[k,:]
    return A,B,Ad,Bd,u,X

def test_dmdc_noiseless_recovers_Ad_Bd():
    n,m,T = 5,2,180
    A,B,Ad,Bd,u,X = _noiseless_system(n,m,T)
    Xtrain, Xp, Utrain = X[:, :-1], X[:, 1:], u.T
    Ahat, Bhat = dmdc_fit(Xtrain, Xp, Utrain)
    # Projected errors on unrestricted V (span K with x0,B)
    from ..metrics import unified_generator
    K = unified_generator(A, B, X[:,0], mode="unrestricted")
    PV,_ = np.linalg.qr(K, mode="reduced")
    errA, errB = projected_errors(Ahat, Bhat, Ad, Bd, Vbasis=PV)
    assert errA <= 1e-10 and errB <= 1e-10

def test_tls_and_iv_do_not_error_with_noise():
    n,m,T = 5,2,200
    A,B,Ad,Bd,u,X = _noiseless_system(n,m,T)
    rng = np.random.default_rng(1)
    # add small noise
    Xn  = X[:, :-1] + 1e-3*rng.standard_normal(X[:, :-1].shape)
    Xpn = X[:, 1:]  + 1e-3*rng.standard_normal(X[:, 1:].shape)
    Un  = u.T + 1e-3*rng.standard_normal(u.T.shape)
    A1,B1 = dmdc_tls_fit(Xn, Xpn, Un, rcond=1e-10, energy=0.99)
    A2,B2 = dmdc_iv_fit(Xn, Xpn, Un, lag=1, rcond=1e-10)
    # shapes OK
    assert A1.shape == (n,n) and B1.shape == (n,m)
    assert A2.shape == (n,n) and B2.shape == (n,m)
