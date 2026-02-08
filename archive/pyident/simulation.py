from __future__ import annotations
from typing import Optional, Tuple, Union
import warnings
import numpy as np
from numpy.random import Generator


def simulate_dt(
    *args: Union[int, np.ndarray],
    noise_std: float = 0.0,
    rng: Optional[Generator] = None,
) -> np.ndarray:
    """Simulate a discrete-time LTI system with optional Gaussian process noise.

    Args:
        x0: Initial state vector ``(n,)``.
        Ad: Discrete-time state matrix ``(n, n)``.
        Bd: Discrete-time input matrix ``(n, m)``.
        U: Time-major input sequence with shape ``(T, m)``.
        noise_std: Optional standard deviation for additive process noise.
        rng: Optional ``numpy.random.Generator`` used when ``noise_std > 0``.

    Returns:
        X: array of shape (n, T+1) including initial state
    Notes:
        The preferred call signature is ``simulate_dt(x0, Ad, Bd, U, ...)``.
        For backwards compatibility, the legacy
        ``simulate_dt(T, x0, Ad, Bd, U, ...)`` form is also accepted but will
        emit a ``DeprecationWarning``.
    """

    legacy_T: Optional[int] = None

    if len(args) == 4:
        x0, Ad, Bd, U = args  # type: ignore[misc]
    elif len(args) == 5:
        # Backwards-compatible call pattern simulate_dt(T, x0, Ad, Bd, U, ...)
        legacy_T, x0, Ad, Bd, U = args  # type: ignore[misc]
        warnings.warn(
            "simulate_dt(T, x0, Ad, Bd, U, ...) is deprecated; "
            "call simulate_dt(x0, Ad, Bd, U, ...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    else:
        raise TypeError(
            "simulate_dt expected 4 positional arguments (x0, Ad, Bd, U); "
            "optionally accept legacy 5-argument form (T, x0, Ad, Bd, U)."
        )

    U = np.asarray(U, dtype=float)

    if legacy_T is not None:
        if U.ndim != 2:
            raise ValueError(
                "Legacy simulate_dt(T, ...) call received non 2-D U with shape {}.".format(
                    U.shape
                )
            )
        if U.shape[0] != legacy_T and U.shape[1] == legacy_T:
            warnings.warn(
                "simulate_dt detected channel-major inputs (m, T) in the deprecated "
                "signature and will transpose them to time-major (T, m).",
                DeprecationWarning,
                stacklevel=2,
            )
            U = np.asarray(U.T, dtype=float)
        elif U.shape[0] != legacy_T:
            raise ValueError(
                "Input horizon mismatch: provided T={} but U has shape {}.".format(
                    legacy_T, U.shape
                )
            )

    if U.ndim != 2:
        raise ValueError(f"U must be 2-D (T, m); got shape {U.shape}.")

    T, m = U.shape
    if Bd.shape[1] != m:
        raise ValueError(
            f"Bd has {Bd.shape[1]} input columns but U provides m={m} channels."
        )

    if rng is None:
        rng = np.random.default_rng()
    
    n = Ad.shape[0]

    if Ad.shape[1] != n:
        raise ValueError("Ad must be square (n, n).")

    X = np.zeros((n, T + 1), dtype=float)
    X[:, 0] = np.asarray(x0).reshape(n)

    for t in range(T):
        X[:, t + 1] = Ad @ X[:, t] + Bd @ U[t, :]
        if noise_std > 0.0:
            X[:, t + 1] += rng.normal(0.0, noise_std, size=n)
    
    return X

def prbs(
    T: int,
    m: int,
    scale: float = 1.0,
    dwell: int = 1,
    rng: Optional[Generator] = None,
) -> np.ndarray:
    """Generate a pseudo-random binary sequence (PRBS) input.

    Args:
        T: Horizon length (number of time steps).
        m: Number of input channels.
        scale: Amplitude applied to the ±1 sequence.
        dwell: Number of repeated samples per binary draw.
        rng: Optional ``numpy.random.Generator``.

    Returns:
        U: array of shape (T, m)
    """
    if rng is None:
        rng = np.random.default_rng()
        
    if dwell <= 0:
        raise ValueError("dwell must be positive.")

    U = np.zeros((T, m), dtype=float)
    blocks = int(np.ceil(T / dwell)) + 1
    draws = rng.choice([-1.0, 1.0], size=(blocks, m))
    tiled = np.repeat(draws, dwell, axis=0)[:T, :]
    U[:, :] = tiled
    
    return scale * U