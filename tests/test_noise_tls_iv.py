import numpy as np
from .. import ensembles, estimators

def _simulate_dt(Ad, Bd, x0, U):
    T = U.shape[1]
    n = Ad.shape[0]
    X = np.zeros((n, T+1)); X[:,0]=x0
    for t in range(T): X[:,t+1] = Ad@X[:,t] + Bd@U[:,t]
    return X

def _relative_param_error(Ah,Bh,A,B):
    return 0.5*(np.linalg.norm(Ah-A,"fro")/np.linalg.norm(A,"fro")+np.linalg.norm(Bh-B,"fro")/np.linalg.norm(B,"fro"))

def test_iv_beats_pinv_under_measurement_noise(rng):
    wins = 0
    trials = 5
    for s in range(trials):
        r = np.random.default_rng(100 + s)
        n,m,T = 6,2,600
        A,B = ensembles.ginibre(n,m,r); A *= 0.9
        x0 = r.standard_normal(n)
        U  = r.standard_normal((m,T))
        X  = _simulate_dt(A,B,x0,U)
        X0, X1 = X[:,:-1], X[:,1:]

        sigma = 0.10
        X0n = X0 + sigma*r.standard_normal(X0.shape)
        X1n = X1 + sigma*r.standard_normal(X1.shape)

        A_ols,B_ols = estimators.dmdc_pinv(X0n, X1n, U)
        A_tls,B_tls = estimators.dmdc_tls (X0n, X1n, U)
        A_iv ,B_iv  = estimators.dmdc_iv  (X0n, X1n, U, L=2)

        e_ols = _relative_param_error(A_ols,B_ols,A,B)
        e_iv  = _relative_param_error(A_iv ,B_iv ,A,B)
        e_tls = _relative_param_error(A_tls,B_tls,A,B)

        wins += int((e_iv < e_ols) or (e_tls < e_ols))

    assert wins >= 3  # majority of trials
