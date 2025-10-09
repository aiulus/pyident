import numpy as np
from .. import estimators

def test_project_identifiable_removes_unexcited_directions():
    rng = np.random.default_rng(23)
    n,m,T = 6,2,200
    # Build regressors Z with a known nullspace direction e.g., zero-out last row of U
    X = rng.standard_normal((n, T))
    U = rng.standard_normal((m, T))
    U[-1,:] = 0.0  # unexcited input channel
    Z = np.vstack([X, U])  # (n+m, T)
    # Random theta has energy in all directions
    Theta = rng.standard_normal((n, n+m))
    Thetap = estimators.project_identifiable(Theta, Z)
    # Any component on the unexcited input channel (last col-block) should be suppressed
    col_norm_unexcited = np.linalg.norm(Thetap[:, n + (m-1)], 2)
    # It can be small but not necessarily zero numerically; ensure strong reduction
    assert col_norm_unexcited <= 1e-6 or col_norm_unexcited < 0.1*np.linalg.norm(Theta[:, n + (m-1)], 2)
