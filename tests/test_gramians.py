import numpy as np
from ..metrics import (
    cont2discrete_zoh,
    gramian_ct_infinite,
    gramian_dt_infinite,
    gramian_dt_finite,
)

def is_psd(M, tol=1e-10):
    ev = np.linalg.eigvalsh((M + M.T) / 2.0)
    return float(ev.min()) >= -tol

def test_ct_infinite_psd_when_hurwitz():
    rng = np.random.default_rng(0)
    n, m = 5, 2
    A = rng.standard_normal((n, n)) * 0.1 - 0.8 * np.eye(n)  # Hurwitz
    K = rng.standard_normal((n, m + 1))
    W = gramian_ct_infinite(A, K)
    assert W is not None and is_psd(W)

def test_dt_infinite_exists_only_when_stable():
    rng = np.random.default_rng(1)
    n, m = 4, 2
    A = -0.5 * np.eye(n)
    B = rng.standard_normal((n, m))
    Ad, Bd = cont2discrete_zoh(A, B, 0.1)
    K = np.concatenate([rng.standard_normal((n, 1)), Bd], axis=1)

    W_inf = gramian_dt_infinite(Ad, K)
    assert W_inf is not None and is_psd(W_inf)

    # Make an unstable Ad explicitly
    Ad_bad = 1.05 * np.eye(n)
    W_inf_bad = gramian_dt_infinite(Ad_bad, K)
    assert W_inf_bad is None

def test_dt_finite_monotone_in_T():
    rng = np.random.default_rng(2)
    n, m = 5, 2
    A = -0.1 * np.eye(n)
    B = rng.standard_normal((n, m))
    Ad, Bd = cont2discrete_zoh(A, B, 0.1)
    K = np.concatenate([rng.standard_normal((n, 1)), Bd], axis=1)
    W5 = gramian_dt_finite(Ad, K, 5)
    W10 = gramian_dt_finite(Ad, K, 10)
    assert is_psd(W5) and is_psd(W10)
    D = W10 - W5
    assert is_psd(D)  # monotone nondecreasing
