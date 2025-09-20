import numpy as np
from ..ensembles import stable, sparse_continuous, ginibre, binary

def spectral_abscissa(A):
    lam = np.linalg.eigvals(A)
    return float(np.max(np.real(lam)))

def density(M):
    return (np.count_nonzero(M) / M.size) if M.size else 0.0

def test_stable_is_hurwitzish():
    rng = np.random.default_rng(0)
    A, B = stable(6, 2, rng)
    assert spectral_abscissa(A) < 0.0

def test_sparse_density_targets():
    rng = np.random.default_rng(1)
    n, m = 20, 5
    p = 0.3
    A, B = sparse_continuous(n=n, m=m, rng=rng, which="both", p_density=p)
    # Allow slack due to randomness
    assert abs(density(A) - p) <= 0.1
    assert abs(density(B) - p) <= 0.1

def test_ginibre_full_rank_high_prob():
    rng = np.random.default_rng(2)
    n, m = 8, 3
    A, B = ginibre(n, m, rng)
    rA = np.linalg.matrix_rank(A)
    rB = np.linalg.matrix_rank(B)
    assert rA == n and rB == m

def test_binary_values_support():
    rng = np.random.default_rng(3)
    A, B = binary(6, 2, rng)
    assert set(np.unique(A)).issubset({-1, 0, 1})
    assert set(np.unique(B)).issubset({-1, 0, 1})
