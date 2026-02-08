from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class TolerancePolicy:
    """Centralized numerical tolerances (logged to the ledger)."""
    svd_rtol: float = 1e-9
    svd_atol: float = 1e-12
    ev_atol: float = 1e-10
    pbh_cluster_tol: float = 1e-8

    def rank_from_singulars(self, s: np.ndarray) -> int:
        if s.size == 0:
            return 0
        s = np.asarray(s, dtype=float)
        smax = float(s[0])
        thresh = max(self.svd_atol, self.svd_rtol * smax)
        return int((s > thresh).sum())

    def cluster_eigs(self, lam: np.ndarray) -> np.ndarray:
        lam = np.asarray(lam)
        if lam.size == 0:
            return lam
        used = np.zeros(lam.shape, dtype=bool)
        reps = []
        for i in range(len(lam)):
            if used[i]:
                continue
            group = [i]
            for j in range(i + 1, len(lam)):
                if abs(lam[j] - lam[i]) <= self.pbh_cluster_tol:
                    group.append(j)
            used[group] = True
            reps.append(lam[group].mean())
        return np.array(reps, dtype=lam.dtype)
