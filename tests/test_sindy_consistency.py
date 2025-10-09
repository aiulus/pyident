import numpy as np
import pytest
from .. import ensembles, estimators, metrics

pysindy = pytest.importorskip("pysindy")

def _simulate_dt(Ad, Bd, x0, U):
    T = U.shape[1]
    n = Ad.shape[0]
    X = np.zeros((n, T+1)); X[:,0]=x0
    for t in range(T): X[:,t+1] = Ad@X[:,t] + Bd@U[:,t]
    return X

def sindy_fit_from_deriv(X: np.ndarray, U: np.ndarray, Xdot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Continuous-time linear identification using provided exact/numerical derivatives:
        Xdot ≈ A X + B U.
    """
    Z = np.vstack([X, U])                 # (n+m, T)
    Theta = Xdot @ np.linalg.pinv(Z, rcond=1e-12)
    n = X.shape[0]
    return Theta[:, :n], Theta[:, n:]


def test_sindy_ct_to_dt_matches_dmdc_in_noiseless_limit(rng):
    n, m = 5, 2
    A_ct, B_ct = ensembles.stable_continuous(n, m, rng)
    dt = 0.02
    Ad, Bd = metrics.cont2discrete_zoh(A_ct, B_ct, dt)

    T = 800
    U = rng.standard_normal((m, T))
    x0 = rng.standard_normal(n)
    X = _simulate_dt(Ad, Bd, x0, U)
    X0, X1 = X[:, :-1], X[:, 1:]

    # DMDc (DT)
    Ahat_dt, Bhat_dt = estimators.dmdc_pinv(X0, X1, U)

    # CT OLS with *true* derivatives at sample times
    Xdot_true = A_ct @ X0 + B_ct @ U
    A_ct_hat, B_ct_hat = sindy_fit_from_deriv(X0, U, Xdot_true)

    # Map CT→DT via ZOH and compare to DMDc
    Ad_hat, Bd_hat = metrics.cont2discrete_zoh(A_ct_hat, B_ct_hat, dt)
    assert np.linalg.norm(Ad_hat - Ahat_dt, "fro") <= 1e-8
    assert np.linalg.norm(Bd_hat - Bhat_dt, "fro") <= 1e-8
