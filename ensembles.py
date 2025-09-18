import numpy as np
import numpy.linalg as npl
from typing import Tuple, Literal, Optional

def ginibre(n: int, m: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    A = rng.normal(size=(n,n)) / np.sqrt(n)
    B = rng.normal(size=(n,m)) / np.sqrt(n)
    return A,B

def binary(n: int, m: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    A = rng.choice([0.0, 1.0], size=(n, n))
    B = rng.choice([0.0, 1.0], size=(n, m))
    return A,B

def rademacher(n: int, m: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    A = rng.choice([-1.0, 1.0], size=(n, n))
    B = rng.choice([-1.0, 1.0], size=(n, m))
    return A,B


def sparse_continuous(
    n: int,
    m: int,
    p_density: float,
    rng: np.random.Generator,
    which: Literal["A", "B", "both"] = "both",
    b_density: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw a sparse–continuous ensemble:
      A_ij = Z_ij * M_ij,  Z_ij ~ N(0,1),  M_ij ~ Bernoulli(density)
      B similarly if requested.

    Parameters
    ----------
    n, m : dimensions (A is n×n, B is n×m)
    p_density : float in [0,1]
        Nonzero fraction for A (and for B too if which='both' and b_density is None).
    rng : np.random.Generator
    which : {'A','B','both'}
        Which matrix to sparsify.
    b_density : Optional[float] in [0,1]
        If provided and which='both', use this for B’s nonzero fraction; otherwise B uses p_density.

    Returns
    -------
    A, B : np.ndarray, np.ndarray
        Drawn matrices with requested sparsity pattern(s).
    """
    if not (0.0 <= p_density <= 1.0):
        raise ValueError("p_density must be in [0,1].")
    if b_density is not None and not (0.0 <= b_density <= 1.0):
        raise ValueError("b_density must be in [0,1].")

    # Dense Gaussian draws
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, m))

    if which in ("A", "both"):
        Amask = rng.binomial(1, p_density, size=(n, n)).astype(float)
        A *= Amask

    if which in ("B", "both"):
        densB = p_density if b_density is None else b_density
        Bmask = rng.binomial(1, densB, size=(n, m)).astype(float)
        B *= Bmask

    return A, B


# Ensures a Hurwitz A matrix
def stable(n: int, m: int, rng: np.random.Generator, spectral_radius: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    # draw and scale A to have spectral radius < spectral_radius (DT stability)
    A = rng.standard_normal((n, n))
    vals = npl.eigvals(A)
    rho = max(1e-12, np.max(np.abs(vals)))
    A = (spectral_radius / rho) * A
    B = rng.standard_normal((n, m))
    return A, B

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

def draw_initial_state(n: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode == "gaussian":
        x0 = rng.standard_normal(n)
    elif mode == "rademacher":
        x0 = rng.choice([-1.0, 1.0], size=n)
    elif mode == "ones":
        x0 = np.ones(n)
    elif mode == "zero":
        x0 = np.zeros(n)
    else:
        raise ValueError(f"Unknown x0_mode={mode}")
    return x0