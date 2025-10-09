import numpy as np
import pytest
from .. import ensembles

def test_ginibre_shapes(rng):
    A,B = ensembles.ginibre(7, 3, rng)
    assert A.shape == (7,7) and B.shape == (7,3)
    # Rough scale check (1/sqrt(n))
    assert np.isclose(np.std(A), 1/np.sqrt(7), rtol=0.5)

def test_sparse_pair_no_zero_rows_when_requested(rng):
    A,B = ensembles.sparse_continuous(
        n=10, m=3, rng=rng,
        which="both", p_density=0.2,
        check_zero_rows=True, max_attempts=50,
    )
    assert A.shape == (10,10) and B.shape == (10,3)
    assert np.all(np.linalg.norm(B, axis=1) > 0)

def test_draw_with_ctrb_rank_exact(rng):
    n,m,r = 8, 2, 5
    A,B,meta = ensembles.draw_with_ctrb_rank(n=n, m=m, r=r, rng=rng)
    from ..ensembles import controllability_rank
    rk,_ = controllability_rank(A, B, order=n)
    assert rk == r
    assert meta["rank"] == r
    assert meta["R_basis"].shape == (n, r)
