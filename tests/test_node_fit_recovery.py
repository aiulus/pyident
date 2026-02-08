import numpy as np
import numpy.linalg as npl

from pyident.metrics import cont2discrete_zoh
from pyident.simulation import simulate_dt
from pyident.estimators import node_fit


def _rel_err(Ahat, A):
    return float(npl.norm(Ahat - A, "fro") / (npl.norm(A, "fro") + 1e-12))


def test_node_fit_recovers_simple_system():
    rng = np.random.default_rng(123)

    # Simple stable continuous-time system
    A_ct = np.array([[-0.5, 0.1],
                     [0.0, -0.2]], dtype=float)
    B_ct = np.array([[1.0],
                     [0.5]], dtype=float)
    dt = 0.1

    Ad, Bd = cont2discrete_zoh(A_ct, B_ct, dt)

    T = 60
    m = B_ct.shape[1]
    U = rng.standard_normal((T, m))  # time-major (T, m)
    x0 = rng.standard_normal(A_ct.shape[0])

    X = simulate_dt(x0, Ad, Bd, U)
    X0 = X[:, :-1]
    X1 = X[:, 1:]

    Ahat, Bhat = node_fit(
        X0,
        X1,
        U,
        dt,
        epochs=120,
        lr=1e-2,
        verbose=False,
        early_stopping=True,
        patience=20,
        min_delta=1e-6,
        convergence_tol=1e-6,
        seed=123,
        warmstart_lstsq=True,
        use_scheduler=False,
    )

    # Recovery quality: should be reasonably close in the noiseless case.
    assert _rel_err(Ahat, Ad) < 0.2
    assert _rel_err(Bhat, Bd) < 0.2
