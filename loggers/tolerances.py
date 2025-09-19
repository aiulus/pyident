from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class TolerancePolicy:
    """Centralized numerical tolerances (logged to the ledger).

    svd_rtol: relative cutoff for singular values, as multiple of s_max
    svd_atol: absolute floor for singular values
    ev_atol:  absolute threshold for eigenvalue clustering on the complex plane
    pbh_cluster_tol: cluster radius for deduplicating near-repeated eigenvalues
    """
    svd_rtol: float = 1e-9
    svd_atol: float = 1e-12
    ev_atol: float = 1e-10
    pbh_cluster_tol: float = 1e-7

    def rank_from_singulars(self, s: np.ndarray) -> int:
        if s.size == 0:
            return 0
        smax = float(s[0])
        thresh = max(self.svd_atol, self.svd_rtol * smax)
        return int((s > thresh).sum())

    def cluster_eigs(self, lam: np.ndarray) -> np.ndarray:
        """Return representative eigenvalues after clustering within pbh_cluster_tol."""
        if lam.size == 0:
            return lam
        used = np.zeros(lam.shape, dtype=bool)
        reps = []
        for i in range(len(lam)):
            if used[i]:
                continue
            group = [i]
            for j in range(i+1, len(lam)):
                if abs(lam[j] - lam[i]) <= self.pbh_cluster_tol:
                    group.append(j)
            used[group] = True
            reps.append(lam[group].mean())
        return np.array(reps, dtype=lam.dtype)
