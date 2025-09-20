import numpy as np
from ..loggers.tolerances import TolerancePolicy

def test_rank_threshold_behavior():
    tol = TolerancePolicy(svd_rtol=1e-6, svd_atol=1e-12)
    s = np.array([1.0, 1e-7, 1e-9, 1e-12])
    r = tol.rank_from_singulars(s)
    # threshold = max(atol, rtol*smax) = max(1e-12, 1e-6) = 1e-6 -> only first > thresh
    assert r == 1

def test_eigen_clustering_merges_close_pairs():
    tol = TolerancePolicy(pbh_cluster_tol=1e-6)
    lam = np.array([1.0+0j, 1.0+5e-7j, 1.0-2e-7j, -0.2+0j, -0.2000004+0j])
    reps = tol.cluster_eigs(lam)
    # Should produce ~3 representatives: around 1.0, and around -0.2
    assert reps.size in (2, 3)  # depending on averaging, near-duplicates collapse
    # Check the first cluster near 1.0
    assert np.isclose(reps.real.max(), 1.0, atol=1e-6) or np.isclose(reps.real.min(), 1.0, atol=1e-6)
