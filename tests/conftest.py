import numpy as np
import pytest

@pytest.fixture(scope="session")
def seed():
    return 12345

@pytest.fixture
def rng(seed):
    return np.random.default_rng(seed)

@pytest.fixture
def dt():
    return 0.05

@pytest.fixture
def horizon():
    return 400  # long enough for PE & stable sims

@pytest.fixture
def pe_inputs(rng, horizon):
    def _make(m, T=None):
        T = T or horizon
        # PE: i.i.d. Gaussian; you can swap for PRBS if you like
        return rng.standard_normal((m, T))
    return _make

# tests/utils_pe.py
import numpy as np
from .. import metrics


def ensure_pe_on_V(Ad, Bd, x0, X0, U, *, tol=1e-12):
    """
    Verify 'richness on the visible subspace' for a single run:
      rank([P^T X0; U]) = k + m and it's not numerically singular
      (after benign column normalization).
    Returns (P, k) for reuse in the test.
    """
    P, k = metrics.visible_subspace(Ad, Bd, x0)
    m = U.shape[0]

    ZV = np.vstack([P.T @ X0, U])     # (k+m, T)

    # Column-normalize to avoid artificial conditioning from energy growth/decay
    coln = np.linalg.norm(ZV, axis=0)
    coln[coln == 0.0] = 1.0
    ZVn = ZV / coln

    s = np.linalg.svd(ZVn, compute_uv=False)
    assert (s > tol).sum() == (k + m), f"rank deficient on V: got {(s>tol).sum()} < {k+m}"
    cond = s[0] / (s[-1] + 1e-30)
    assert cond < 1e8, f"ill-conditioned on V even after normalization: cond={cond:.2e}"

    return P, k


