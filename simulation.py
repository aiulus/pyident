from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
from numpy.random import Generator

def simulate_dt(T: int,
                x0: np.ndarray,
                Ad: np.ndarray,
                Bd: np.ndarray,
                U: np.ndarray,
                noise_std: float = 0.0,
                rng: Optional[Generator] = None) -> np.ndarray:
    """
    Simulate discrete-time system with optional noise.
    
    Returns:
        X: array of shape (n, T+1) including initial state
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n = Ad.shape[0]
    X = np.zeros((n, T+1))
    X[:, 0] = x0.flatten()
    
    for t in range(T):
        X[:, t+1] = Ad @ X[:, t] + Bd @ U[:, t]
        if noise_std > 0:
            X[:, t+1] += rng.normal(0, noise_std, size=n)
    
    return X

def prbs(m: int,
         T: int,
         scale: float = 1.0,
         dwell: int = 1,
         rng: Optional[Generator] = None) -> np.ndarray:
    """
    Generate PRBS input signal.
    
    Returns:
        U: array of shape (m, T)
    """
    if rng is None:
        rng = np.random.default_rng()
        
    U = np.zeros((m, T))
    for i in range(m):
        base = rng.choice([-1, 1], size=(T // dwell + 1))
        U[i] = np.repeat(base, dwell)[:T]
    return scale * U