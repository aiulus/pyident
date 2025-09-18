import warnings
from typing import Tuple, Literal, Optional

import numpy as np
import numpy.linalg as npl


# ---------------------------------------------------------------------
# Random-matrix factories for (A, B).
# All generators return numpy arrays with shapes:
#   A: (n, n),  B: (n, m)
# ---------------------------------------------------------------------


def ginibre(n: int, m: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    A = rng.normal(size=(n, n)) / np.sqrt(n)
    B = rng.normal(size=(n, m)) / np.sqrt(n)
    return A, B


def binary(n: int, m: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Binary ensemble.This may induce strong degeneracies (e.g., repeated singular values).
      Prefer rademacher() if you need symmetric +/- 1."""
    A = rng.choice([0.0, 1.0], size=(n, n))
    B = rng.choice([0.0, 1.0], size=(n, m))
    return A, B


def rademacher(n: int, m: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    A = rng.choice([-1.0, 1.0], size=(n, n))
    B = rng.choice([-1.0, 1.0], size=(n, m))
    return A, B


def sparse_continuous(
    n: int,
    m: int,
    p_density: float,
    rng: np.random.Generator,
    which: Literal["A", "B", "both"] = "both",
    p_density_A: Optional[float] = None,
    p_density_B: Optional[float] = None,
    check_zero_rows: bool = False,
    max_attempts: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sparse–continuous ensemble.

    A_ij = Z_ij * M_ij where Z_ij ~ N(0,1), M_ij ~ Ber(p_density_A).
    B is treated similarly if requested.

    Disclaimer:
    - If check_zero_rows=True, we resample masks up to max_attempts to avoid fully
      zero rows (for A and/or B depending on `which`). This slightly biases the
      mask distribution but prevents degenerate, non-identifiable edge cases
      unrelated to the phenomena of interest.

    Parameters
    ----------
    n, m : int
    p_density_A : float in [0,1]
        Nonzero fraction for A.
    rng : np.random.Generator
    which : {'A','B','both'}
        Which matrix to sparsify.
    p_density_B : Optional[float] in [0,1]
        Nonzero fraction for B (if None and 'both' or 'B', uses p_density_A).
    check_zero_rows : bool
        Enforce no all-zero rows in the sparsified matrices (see note above).
    max_attempts : int
        Max resampling attempts for zero-row avoidance.

    Returns
    -------
    A, B : np.ndarray
    """
    p_density_A = p_density_A if p_density_A is not None else p_density
    p_density_B = p_density_B if p_density_B is not None else p_density

    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, m))

    def _mask(shape, p, avoid_zero_rows=False) -> np.ndarray:
        if not avoid_zero_rows:
            return (rng.binomial(1, p, size=shape)).astype(float)
        # Resample until no zero rows OR attempts exhausted.
        attempts = 0
        while True:
            M = (rng.binomial(1, p, size=shape)).astype(float)
            if np.all(M.sum(axis=1) > 0) or attempts >= max_attempts:
                if attempts >= max_attempts:
                    warnings.warn(
                        "sparse_continuous: zero-row avoidance max_attempts hit; "
                        "returning current mask (may contain zero rows)."
                    )
                return M
            attempts += 1

    if which in ("A", "both"):
        Amask = _mask((n, n), p_density_A, check_zero_rows)
        A *= Amask

    if which in ("B", "both"):
        densB = p_density_A if p_density_B is None else p_density_B
        Bmask = _mask((n, m), densB, check_zero_rows)
        B *= Bmask

    return A, B


def stable(
    n: int,
    m: int,
    rng: np.random.Generator,
    spectral_radius: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Discrete-time ‘stable’ ensemble: scale A to have spectral radius < spectral_radius.

    Disclaimer:
    - This is a *post-hoc spectral scaling* (A ← \alpha A). It preserves the random
      direction structure but *alters* the singular/eigen value distribution.
      We use it only to avoid blow-up in DT simulations. If your study
      is sensitive to eigenvalue statistics, prefer reporting both the raw
      (unstable) and scaled cases.

    Returns
    -------
    A, B : np.ndarray
    """
    A = rng.standard_normal((n, n))
    vals = npl.eigvals(A)
    rho = max(1e-12, np.max(np.abs(vals)))
    A = (spectral_radius / rho) * A
    B = rng.standard_normal((n, m))
    return A, B


# ---------------------------------------------------------------------
# Convenience adapter used by run_single / CLI
# ---------------------------------------------------------------------
def sample_system_instance(cfg, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch by cfg.ensemble with cfg’s density fields.

    Supported:
    - 'ginibre', 'binary', 'stable', 'sparse'
    """
    if cfg.ensemble == "ginibre":
        return ginibre(cfg.n, cfg.m, rng)
    if cfg.ensemble == "binary":
        return binary(cfg.n, cfg.m, rng)
    if cfg.ensemble == "sparse":
        return sparse_continuous(
            n=cfg.n,
            m=cfg.m,
            p_density_A=cfg._density_A,
            rng=rng,
            which=cfg.sparse_which,
            p_density_B=cfg._density_B,
            check_zero_rows=False,
        )
    if cfg.ensemble == "stable":
        return stable(cfg.n, cfg.m, rng)
    raise ValueError(f"unknown ensemble: {cfg.ensemble}")


def draw_initial_state(n: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Initial condition sampler."""
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
