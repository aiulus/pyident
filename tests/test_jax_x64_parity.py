import numpy as np
import pytest

jax_spec = __import__("importlib").util.find_spec("jax") if True else None
has_jax = jax_spec is not None

@pytest.mark.skipif(not has_jax, reason="JAX not installed")
def test_x64_parity_against_numpy_if_supported():
    import jax
    from . import jax_accel as jxa
    # Skip on Metal (no eig / limited dtypes), and on platforms without x64
    plat = jax.devices()[0].platform
    if plat == "metal":
        pytest.skip("x64 not fully supported on METAL")
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pytest.skip("Cannot enable JAX x64 on this platform")

    rng = np.random.default_rng(0)
    n, m, T = 4, 2, 60
    A = rng.standard_normal((n, n)) * 0.1 - 0.6 * np.eye(n)
    B = rng.standard_normal((n, m))
    x0 = rng.standard_normal(n)
    dt = 0.05

    # ZOH by NumPy path (pyident)
    from pyident.metrics import cont2discrete_zoh
    Ad, Bd = cont2discrete_zoh(A, B, dt)

    u = rng.standard_normal((T, m))
    # JAX simulate in x64
    Xj = np.asarray(jxa.simulate_discrete(Ad, Bd, u, x0))
    # NumPy simulate
    Xn = np.empty((n, T+1)); Xn[:, 0] = x0
    for k in range(T):
        Xn[:, k+1] = Ad @ Xn[:, k] + Bd @ u[k, :]
    rel = np.linalg.norm(Xj - Xn) / (1 + np.linalg.norm(Xn))
    assert rel <= 1e-12
