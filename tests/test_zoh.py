import numpy as np
from ..metrics import cont2discrete_zoh

def test_zoh_first_order_small_dt():
    rng = np.random.default_rng(0)
    n, m = 4, 2
    A = rng.standard_normal((n, n)) * 0.2
    B = rng.standard_normal((n, m))
    dt = 1e-4
    Ad, Bd = cont2discrete_zoh(A, B, dt)
    Ad_lin = np.eye(n) + A * dt
    Bd_lin = B * dt
    assert np.linalg.norm(Ad - Ad_lin) <= 1e-8
    assert np.linalg.norm(Bd - Bd_lin) <= 1e-8
