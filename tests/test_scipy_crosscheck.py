import numpy as np
import pytest
from ..metrics import cont2discrete_zoh

scipy_spec = __import__("importlib").util.find_spec("scipy.signal") if True else None
has_scipy = scipy_spec is not None

@pytest.mark.skipif(not has_scipy, reason="SciPy not installed")
def test_cont2discrete_matches_scipy():
    from scipy.signal import cont2discrete as c2d
    rng = np.random.default_rng(0)
    n, m = 5, 2
    A = rng.standard_normal((n, n)) * 0.1 - 0.5 * np.eye(n)
    B = rng.standard_normal((n, m))
    dt = 0.05
    Ad, Bd = cont2discrete_zoh(A, B, dt)
    Ad_s, Bd_s, _, _, _ = c2d((A, B, np.eye(n), np.zeros((n, m))), dt, method="zoh")
    assert np.allclose(Ad, Ad_s, atol=1e-8, rtol=1e-8)
    assert np.allclose(Bd, Bd_s, atol=1e-8, rtol=1e-8)
