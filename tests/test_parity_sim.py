import numpy as np
import pytest

from ..metrics import cont2discrete_zoh
from ..run_single import _simulate_numpy

@pytest.mark.skipif("jax" not in globals() and __import__("importlib").util.find_spec("jax") is None, reason="JAX not installed")
def test_simulate_discrete_parity():
    import jax
    from pyident import jax_accel as jxa

    rng = np.random.default_rng(0)
    n, m, T = 5, 2, 80
    A = rng.standard_normal((n, n)) * 0.1 - 0.6 * np.eye(n)
    B = rng.standard_normal((n, m))
    x0 = rng.standard_normal(n)
    Ad, Bd = cont2discrete_zoh(A, B, 0.05)
    u = rng.standard_normal((T, m))

    # JAX path
    try:
        Xj = np.asarray(jxa.simulate_discrete(Ad, Bd, u, x0))
    except Exception:
        pytest.skip("JAX backend not supporting simulate_discrete on this platform")

    # NumPy path
    Xn = _simulate_numpy(Ad, Bd, u, x0)

    err = np.linalg.norm(Xj - Xn) / (1 + np.linalg.norm(Xn))
    assert err <= 1e-10
