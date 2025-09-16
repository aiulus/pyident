
from __future__ import annotations
import numpy as np
import numpy.linalg as npl

def prbs(m:int, T:int, levels:int=2, period:int=7, rng: np.random.Generator|None=None):
    rng = rng or np.random.default_rng()
    U = np.zeros((m, T))
    for i in range(m):
        bits = rng.integers(0, levels, size=T) * 2 - (levels-1)
        for t in range(0, T, period):
            bits[t:t+period] = bits[t]
        U[i,:] = bits
    return U

def is_PE(U: np.ndarray, r:int, tol: float=1e-8):
    m, T = U.shape
    if r <= 0 or r > T:
        return False, 0, 0.0
    blocks = [U[:, i:T-r+i+1] for i in range(r)]
    H = np.vstack(blocks)
    s = np.linalg.svd(H, compute_uv=False)
    rank = int((s > tol).sum())
    return rank == min(H.shape), rank, float(s[-1])
