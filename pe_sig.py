from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, Literal

import numpy as np

from .signals import estimate_pe_order, estimate_moment_pe_order


ArrayLike = np.ndarray


@dataclass
class SignalBundle:
    """Container for a continuous-time input signal and its sampled sequence."""

    t: np.ndarray
    u: np.ndarray  # time-major samples, shape (T, m)
    u_of_t: Callable[[ArrayLike], np.ndarray]
    meta: Dict[str, object]
    pe_order_est: Optional[int] = None


@dataclass
class PRBSSpec:
    """Continuous-time PRBS specification.

    Notes:
        - PRBS is generated with a linear feedback shift register (LFSR) when
          `taps` are provided. Otherwise, a random ±1 pattern is used.
        - `clock` is the hold time (seconds) of each PRBS chip.
    """

    register: int = 7
    clock: float = 1.0
    levels: Tuple[float, float] = (-1.0, 1.0)
    taps: Optional[Sequence[int]] = None
    seed_state: Optional[Sequence[int]] = None


@dataclass
class MultisineSpec:
    """Continuous-time multisine specification."""

    k_lines: int = 8
    freqs: Optional[Sequence[float]] = None  # rad/s
    amps: Optional[Sequence[float]] = None


def _as_1d_time(t: ArrayLike) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    if t_arr.ndim == 0:
        return t_arr.reshape(1)
    return t_arr.reshape(-1)


def _zoh_callable(u: np.ndarray, dt: float, t0: float = 0.0) -> Callable[[ArrayLike], np.ndarray]:
    if dt <= 0:
        raise ValueError("dt must be positive for zero-order hold.")
    T, m = u.shape

    def u_of_t(t_query: ArrayLike) -> np.ndarray:
        tq = _as_1d_time(t_query)
        idx = np.floor((tq - t0) / dt).astype(int)
        idx = np.clip(idx, 0, T - 1)
        out = u[idx]
        if np.asarray(t_query).ndim == 0:
            return out.reshape(m)
        return out

    return u_of_t


def _lfsr_bits(length: int, register: int, taps: Sequence[int], seed_state: Optional[Sequence[int]] = None) -> np.ndarray:
    """Generate LFSR bits using a Fibonacci LFSR with 1-based tap positions.

    Taps are counted from the least significant bit (LSB). For example, taps
    (register, 1) corresponds to the polynomial x^register + x + 1.
    """
    if register <= 1:
        raise ValueError("register must be >= 2.")
    if length <= 0:
        raise ValueError("length must be positive.")
    if not taps:
        raise ValueError("taps must be a non-empty sequence of 1-based positions.")
    for t in taps:
        if t < 1 or t > register:
            raise ValueError(f"tap {t} is out of range for register length {register}.")

    if seed_state is None:
        state = np.ones(register, dtype=np.uint8)
    else:
        state = np.asarray(seed_state, dtype=np.uint8).reshape(-1)
        if state.size != register:
            raise ValueError("seed_state length must equal register length.")
        if not np.any(state):
            raise ValueError("seed_state must be non-zero.")
        state = state.copy()

    bits = np.zeros(length, dtype=np.uint8)
    for k in range(length):
        bits[k] = state[-1]
        fb = 0
        for t in taps:
            fb ^= state[-t]
        state[1:] = state[:-1]
        state[0] = fb
    return bits


def generate_prbs(
    T: int,
    m: int,
    dt: float,
    rng: np.random.Generator,
    spec: PRBSSpec,
    pe_order: Optional[int] = None,
) -> SignalBundle:
    """Generate a continuous-time PRBS signal and its samples.

    The output samples are time-major with shape (T, m).
    """
    if T <= 0 or m <= 0:
        raise ValueError("T and m must be positive.")
    if spec.clock <= 0:
        raise ValueError("PRBS clock must be positive.")

    if pe_order is not None:
        spec = PRBSSpec(
            register=max(int(pe_order), spec.register),
            clock=spec.clock,
            levels=spec.levels,
            taps=spec.taps,
            seed_state=spec.seed_state,
        )

    dwell = max(1, int(np.round(spec.clock / dt)))
    chips_needed = int(np.ceil(T / dwell))

    levels_arr = np.asarray(spec.levels, dtype=float)
    if levels_arr.shape != (2,):
        raise ValueError("levels must be a pair of floats (low, high).")

    if spec.taps is not None:
        period = (2 ** spec.register) - 1
        seq = _lfsr_bits(period, spec.register, spec.taps, seed_state=spec.seed_state)
        chips = np.zeros((chips_needed, m), dtype=float)
        for j in range(m):
            shift = int(rng.integers(0, period)) if m > 1 else 0
            shifted = np.roll(seq, shift)
            tiled = np.tile(shifted, int(np.ceil(chips_needed / period)))[:chips_needed]
            chips[:, j] = levels_arr[tiled]
    else:
        idx = rng.integers(0, 2, size=(chips_needed, m))
        chips = levels_arr[idx]

    U = np.repeat(chips, dwell, axis=0)[:T]

    t = np.arange(T, dtype=float) * dt
    u_of_t = _zoh_callable(U, dt=dt)
    meta = {
        "family": "prbs",
        "register": spec.register,
        "clock": spec.clock,
        "levels": spec.levels,
        "taps": tuple(spec.taps) if spec.taps is not None else None,
        "dwell": dwell,
    }
    return SignalBundle(t=t, u=U, u_of_t=u_of_t, meta=meta)


def _multisine_callable(
    freqs: np.ndarray,
    phases: np.ndarray,
    amps: np.ndarray,
    m: int,
) -> Callable[[ArrayLike], np.ndarray]:
    def u_of_t(t_query: ArrayLike) -> np.ndarray:
        tq = _as_1d_time(t_query)
        out = np.zeros((tq.size, m), dtype=float)
        for j in range(m):
            for w, ph, a in zip(freqs, phases[j], amps[j]):
                out[:, j] += a * np.sin(w * tq + ph)
        if np.asarray(t_query).ndim == 0:
            return out.reshape(m)
        return out

    return u_of_t


def generate_multisine(
    T: int,
    m: int,
    dt: float,
    rng: np.random.Generator,
    spec: MultisineSpec,
    pe_order: Optional[int] = None,
    normalize: bool = True,
) -> SignalBundle:
    if T <= 0 or m <= 0:
        raise ValueError("T and m must be positive.")
    horizon = T * dt

    k_lines = spec.k_lines
    if pe_order is not None:
        k_lines = max(int(pe_order), k_lines)
    if k_lines <= 0:
        raise ValueError("k_lines must be positive.")

    if spec.freqs is None:
        max_bin = max(2, T // 8)
        k_use = min(k_lines, max_bin - 1)
        bins = rng.choice(np.arange(1, max_bin), size=k_use, replace=False)
        freqs = 2.0 * np.pi * bins / max(horizon, dt)
    else:
        freqs = np.asarray(spec.freqs, dtype=float)
        if freqs.ndim != 1 or freqs.size == 0:
            raise ValueError("freqs must be a 1D array of length >= 1.")

    k_use = freqs.size
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(m, k_use))
    if spec.amps is None:
        amps = np.ones((m, k_use), dtype=float)
    else:
        amps_vec = np.asarray(spec.amps, dtype=float)
        if amps_vec.ndim != 1 or amps_vec.size != k_use:
            raise ValueError("amps must be 1D and match number of freqs.")
        amps = np.repeat(amps_vec[None, :], m, axis=0)

    u_of_t = _multisine_callable(freqs, phases, amps, m)
    t = np.arange(T, dtype=float) * dt
    U = u_of_t(t)
    if normalize:
        scale = np.maximum(np.max(np.abs(U), axis=0), 1.0)
        U = U / scale

    meta = {
        "family": "multisine",
        "k_lines": int(k_use),
        "freqs": freqs,
        "phases": phases,
        "amps": amps,
    }
    return SignalBundle(t=t, u=U, u_of_t=u_of_t, meta=meta)


def generate_pe_signal(
    family: Literal["prbs", "multisine"],
    T: int,
    m: int,
    dt: float = 1.0,
    pe_order: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    prbs: Optional[PRBSSpec] = None,
    multisine: Optional[MultisineSpec] = None,
    ensure_pe: bool = True,
    pe_method: Literal["block", "moment"] = "block",
    pe_tol: float = 1e-8,
    max_tries: int = 128,
) -> SignalBundle:
    """Generate a PE control input signal with a desired PE order.

    The returned bundle includes a continuous-time callable and sampled data
    compatible with the rest of the codebase (time-major (T, m)).
    """
    if rng is None:
        rng = np.random.default_rng()

    prbs = prbs if prbs is not None else PRBSSpec()
    multisine = multisine if multisine is not None else MultisineSpec()

    last_bundle: Optional[SignalBundle] = None
    last_pe: Optional[int] = None

    for _ in range(max_tries):
        if family == "prbs":
            bundle = generate_prbs(T, m, dt, rng, spec=prbs, pe_order=pe_order)
        elif family == "multisine":
            bundle = generate_multisine(T, m, dt, rng, spec=multisine, pe_order=pe_order)
        else:
            raise ValueError("family must be 'prbs' or 'multisine'.")

        if not ensure_pe or pe_order is None:
            return bundle

        if pe_method == "block":
            last_pe = estimate_pe_order(bundle.u, s_max=pe_order, tol=pe_tol)
        elif pe_method == "moment":
            last_pe = estimate_moment_pe_order(bundle.u, r_max=pe_order, dt=dt, tol=pe_tol)
        else:
            raise ValueError("pe_method must be 'block' or 'moment'.")

        bundle.pe_order_est = int(last_pe)
        last_bundle = bundle
        if last_pe >= pe_order:
            return bundle

    if last_bundle is None:
        raise RuntimeError("Failed to generate any signal bundle.")
    raise RuntimeError(
        f"Unable to reach PE order {pe_order} after {max_tries} tries; "
        f"best achieved order={last_pe}."
    )
