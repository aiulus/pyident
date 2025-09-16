import numpy as np

def ginibre(n: int, m: int, rng: np.random.Generator):
    A = rng.normal(size=(n,n)) / np.sqrt(n)
    B = rng.normal(size=(n,m)) / np.sqrt(n)
    meta = dict(ensemble="ginibre", density=None, scale=1.0)
    return A,B,meta

def binary(n:int, m:int, rng: np.random.Generator):
    A = rng.integers(0,2,size=(n,n)).astype(float)
    B = rng.integers(0,2,size=(n,m)).astype(float)
    meta = dict(ensemble="binary", density=None, scale=1.0)
    return A,B,meta

def sparse(n:int, m:int, density:float, rng: np.random.Generator):
    A = (rng.random((n,n)) < density) * rng.normal(size=(n,n))
    B = (rng.random((n,m)) < density) * rng.normal(size=(n,m))
    A /= max(1, np.sqrt(n)); B /= max(1, np.sqrt(n))
    meta = dict(ensemble="sparse", density=float(density), scale=1.0)
    return A,B,meta

# Ensures a Hurwitz A matrix
def stable(n:int, m:int, rng: np.random.Generator):
    G = rng.normal(size=(n,n))/np.sqrt(n)
    # shift spectrum to left half-plane
    w = np.linalg.eigvals(G)
    gamma = 0.5 + max(0.0, np.max(np.real(w)))
    A = G - gamma*np.eye(n)
    B = rng.normal(size=(n,m))/np.sqrt(n)
    meta = dict(ensemble="stable", density=None, scale=1.0)
    return A,B,meta

def sample_system_instance(cfg, rng: np.random.Generator):
    if cfg.ensemble == "ginibre":
        return ginibre(cfg.n, cfg.m, rng)
    if cfg.ensemble == "binary":
        return binary(cfg.n, cfg.m, rng)
    if cfg.ensemble == "sparse":
        return sparse(cfg.n, cfg.m, cfg.density, rng)
    if cfg.ensemble == "stable":
        return stable(cfg.n, cfg.m, rng)
    raise ValueError(f"unknown ensemble: {cfg.ensemble}")