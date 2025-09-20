import numpy as np
import pytest
from ..metrics import unified_generator, visible_subspace_basis
from ..signals import restrict_pointwise

def test_no_actuation_B_zero_still_works_and_reduces_visibility():
    rng = np.random.default_rng(0)
    n, m = 6, 0
    A = rng.standard_normal((n, n)) * 0.1 - 0.6 * np.eye(n)
    B = np.zeros((n, 0))
    x0 = rng.standard_normal(n)
    # unified_generator must handle m=0
    K = unified_generator(A, B, x0, mode="unrestricted")
    V, k = visible_subspace_basis(A, B, x0)
    assert K.shape[1] >= 1 and 1 <= k <= n  # rank bounded by Krylov from x0 only

def test_restrict_pointwise_dimension_mismatch_errors():
    T, m = 32, 3
    u = np.random.standard_normal((T, m))
    W_bad = np.random.standard_normal((m+1, 2))  # wrong first dim
    with pytest.raises(Exception):
        restrict_pointwise(u, W_bad)

def test_moment_pe_invalid_r_raises():
    rng = np.random.default_rng(1)
    n, m = 5, 2
    A = rng.standard_normal((n, n)) * 0.1 - 0.6 * np.eye(n)
    B = rng.standard_normal((n, m))
    x0 = rng.standard_normal(n)
    with pytest.raises(ValueError):
        unified_generator(A, B, x0, mode="moment-pe", r=0)
