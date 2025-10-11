"""Experiments: signal knobs (length & frequency) vs. error; post-hoc PE diagnostics.

This version replaces PRBS with a simple square-wave (0/1) alternating input.
We vary *frequency* (Hz) and *length* (number of samples) of the signal and
plot MSE vs. the chosen signal property, while also reporting the resulting
persistency-of-excitation (estimated) as a diagnostic.

Panels:
  Left  : k = dim V(x0) = 3 < n = 5 (partially identifiable)
  Right : k = n = 5 (fully visible)

Also saves a PE-vs-signal-property figure for sanity checking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..estimators import dmdc_tls
from ..metrics import projected_errors
from ..signals import estimate_moment_pe_order, estimate_pe_order
from ..simulation import simulate_dt
from .visible_sampling import VisibleDrawConfig, draw_system_state_with_visible_dim


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PEVisibleConfig(ExperimentConfig):
    """Config for the length/frequency sweep with square-wave input."""
    n: int = 5
    m: int = 2
    dt: float = 0.1
    u_scale: float = 3.0
    noise_std: float = 0.0

    # --- NEW: knobs you vary ---
    freqs: Sequence[float] = (0.25, 0.5, 1.0, 2.0)   # Hz
    lengths: Sequence[int] = (60, 120, 240)                 # number of samples

    # Plot x-axis: "freq" or "length". (We draw both if both have >1 unique values.)
    x_axis: str = "freq"

    # PE estimation caps/tols
    pe_s_max: int = 12
    pe_moment_r_max: int = 12
    pe_svd_tol: float = 1e-9

    # Trials per (length, frequency)
    trials: int = 25

    # Visible-dimension drawing
    max_system_attempts: int = 500
    max_x0_attempts: int = 256
    visible_tol: float = 1e-8
    deterministic_x0: bool = False

    # Output
    outdir: str = "out_signal_knobs"

    def __post_init__(self) -> None:  # type: ignore[override]
        super().__post_init__()
        if self.trials < 1:
            raise ValueError("trials must be positive.")
        if min(self.freqs) <= 0:
            raise ValueError("freqs must be positive.")
        if min(self.lengths) <= 2:
            raise ValueError("lengths must be >= 3.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_system_with_visible_dim(
    cfg: PEVisibleConfig,
    target_dim: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw (A, B, Ad, Bd, x0, P) with dim span(P) == target_dim."""
    draw_cfg = VisibleDrawConfig(
        n=cfg.n, m=cfg.m, dt=cfg.dt, dim_visible=target_dim,
        ensemble="stable",
        max_system_attempts=cfg.max_system_attempts,
        max_x0_attempts=cfg.max_x0_attempts,
        tol=cfg.visible_tol,
        deterministic_x0=cfg.deterministic_x0,
    )
    return draw_system_state_with_visible_dim(draw_cfg, rng)


def _period_samples_from_freq(freq_hz: float, dt: float) -> int:
    """Even number of samples per period for 50% duty; at least 2."""
    p = max(2, int(round(1.0 / (freq_hz * dt))))
    if p % 2:  # make it even
        p += 1
    return p


def square01_signal(T: int, m: int, period_samples: int, scale: float = 1.0) -> np.ndarray:
    """
    Build a (T, m) square wave that alternates 0 and 1 with 50% duty.
    All channels identical (simplest interpretation).
    """
    half = period_samples // 2
    base = np.concatenate([np.zeros(half), np.ones(half)])
    reps = int(np.ceil(T / period_samples))
    sig = np.tile(base, reps)[:T]  # shape (T,)
    U = (scale * sig)[:, None] * np.ones((1, m))  # (T, m)
    return U


def _mse(mat: np.ndarray) -> float:
    return float(np.sum(mat * mat) / mat.size)


def _aggregate(series: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = series.groupby(level=0)
    x = g.median().index.to_numpy()
    med = g.median().to_numpy()
    q1 = g.quantile(0.25).to_numpy()
    q3 = g.quantile(0.75).to_numpy()
    return x, med, q1, q3


def _plot_standard_by_visibility_mse(
    df: pd.DataFrame,
    outfile: pathlib.Path,
    x_col: str,
    xlabel: str,
) -> None:
    """Two panels: k=3 (partial) and k=5 (full); MSE in standard basis."""
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    panels = [("partial", "k=3 (partial)"), ("full", "k=5 (full)")]

    for ax, (scenario, title) in zip(axes, panels):
        sub = df[df["scenario"] == scenario]
        xA, medA, q1A, q3A = _aggregate(sub.set_index(x_col)["mseA"])
        ax.plot(xA, medA, label="MSE(A)")
        ax.fill_between(xA, q1A, q3A, alpha=0.2)

        xB, medB, q1B, q3B = _aggregate(sub.set_index(x_col)["mseB"])
        ax.plot(xB, medB, linestyle="--", label="MSE(B)")
        ax.fill_between(xB, q1B, q3B, alpha=0.2)

        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(xlabel)

    axes[0].set_ylabel("Mean squared error")
    axes[0].legend(title="Standard-basis errors")
    fig.suptitle(f"MSE vs. {xlabel} (standard basis)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outfile, dpi=160)
    plt.close(fig)


def _plot_posthoc_pe(df: pd.DataFrame, outfile: pathlib.Path, x_col: str, xlabel: str) -> None:
    """Report 'used' PE (median ± IQR) vs. the chosen signal knob."""
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    panels = [("partial", "k=3 (partial)"), ("full", "k=5 (full)")]
    for ax, (scenario, title) in zip(axes, panels):
        sub = df[df["scenario"] == scenario]
        x, med, q1, q3 = _aggregate(sub.set_index(x_col)["pe_order_actual"])
        ax.plot(x, med, label="block-PE (Hankel)")
        ax.fill_between(x, q1, q3, alpha=0.2)
        x2, med2, q12, q32 = _aggregate(sub.set_index(x_col)["pe_order_moment"])
        ax.plot(x2, med2, linestyle="--", label="moment-PE")
        ax.fill_between(x2, q12, q32, alpha=0.2)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Estimated PE order")
    axes[0].legend()
    fig.suptitle(f"Post-hoc PE vs. {xlabel}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outfile, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def _run_trial(
    label: str,
    U: np.ndarray,
    freq_hz: float,
    period_samples: int,
    system: Dict[str, np.ndarray],
    cfg: PEVisibleConfig,
):
    """Simulate one scenario; compute MSEs + PE diagnostics."""
    x0 = system["x0"]; Ad = system["Ad"]; Bd = system["Bd"]; P = system["P"]

    # Simulate and fit
    X = simulate_dt(x0, Ad, Bd, U, noise_std=cfg.noise_std)
    X0, X1 = X[:, :-1], X[:, 1:]
    Ahat, Bhat = dmdc_tls(X0, X1, U)

    # Standard-basis errors
    mseA = _mse(Ahat - Ad)
    mseB = _mse(Bhat - Bd)

    # (Kept for completeness; not used in main plot now)
    dA_V, dB_V = projected_errors(Ahat, Bhat, Ad, Bd, P)

    # Post-hoc PE
    s_max = min(cfg.pe_s_max, U.shape[0] // 2)  # guard
    r_max = min(cfg.pe_moment_r_max, U.shape[0] // 2)
    pe_block = int(estimate_pe_order(U, s_max=s_max, svd_tol=cfg.pe_svd_tol))
    pe_moment = int(estimate_moment_pe_order(U, r_max=r_max, dt=cfg.dt, tol=cfg.pe_svd_tol))

    return {
        "scenario": label,
        "freq_hz": freq_hz,
        "period_samples": period_samples,
        "T": U.shape[0],
        "cycles": U.shape[0] / period_samples,
        "pe_order_actual": pe_block,
        "pe_order_moment": pe_moment,
        "mseA": mseA,
        "mseB": mseB,
        "errA_V_rel": float(dA_V),  # stored if you still want it later
        "errB_V_rel": float(dB_V),
        "dim_visible": system["dim_visible"],
    }


def run_experiment(cfg: PEVisibleConfig) -> pd.DataFrame:
    base_rng = np.random.default_rng(cfg.seed)
    sys_rng = np.random.default_rng(base_rng.integers(0, 2**32))
    input_rng = np.random.default_rng(base_rng.integers(0, 2**32))  # kept for parity

    # Systems
    partial = dict(zip(
        ("A", "B", "Ad", "Bd", "x0", "P"),
        _draw_system_with_visible_dim(cfg, target_dim=3, rng=sys_rng)
    ))
    partial["dim_visible"] = 3

    full = dict(zip(
        ("A", "B", "Ad", "Bd", "x0", "P"),
        _draw_system_with_visible_dim(cfg, target_dim=cfg.n, rng=sys_rng)
    ))
    full["dim_visible"] = cfg.n

    records: List[Dict[str, float]] = []

    for T in cfg.lengths:
        for f in cfg.freqs:
            period = _period_samples_from_freq(f, cfg.dt)
            for _ in range(cfg.trials):
                U = square01_signal(T=T, m=cfg.m, period_samples=period, scale=cfg.u_scale)
                # Run both scenarios on EXACTLY the same input
                records.append(_run_trial("partial", U, f, period, partial, cfg))
                records.append(_run_trial("full",    U, f, period, full,    cfg))

    df = pd.DataFrame.from_records(records)

    # Save CSV
    outdir = pathlib.Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "signal_knobs.csv", index=False)

    # Choose x-axis and plot
    if cfg.x_axis == "freq":
        x_col, xlabel = "freq_hz", "Frequency (Hz)"
    else:
        x_col, xlabel = "T", "Signal length (samples)"

    # If x-axis chosen has only one unique value, auto-fallback to the other
    if df[x_col].nunique() < 2:
        if x_col == "freq_hz" and df["T"].nunique() > 1:
            x_col, xlabel = "T", "Signal length (samples)"
        elif x_col == "T" and df["freq_hz"].nunique() > 1:
            x_col, xlabel = "freq_hz", "Frequency (Hz)"

    _plot_standard_by_visibility_mse(
        df, outdir / f"mse_vs_{x_col}.png", x_col=x_col, xlabel=xlabel
    )
    _plot_posthoc_pe(
        df, outdir / f"posthoc_pe_vs_{x_col}.png", x_col=x_col, xlabel=xlabel
    )

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="out_signal_knobs")
    parser.add_argument("--x-axis", type=str, default="freq", choices=["freq", "length"])
    parser.add_argument("--trials", type=int, default=25)
    # Simple comma-separated overrides
    parser.add_argument("--freqs", type=str, default="")
    parser.add_argument("--lengths", type=str, default="")
    args = parser.parse_args(argv)

    freqs = None
    if args.freqs:
        freqs = tuple(float(s) for s in args.freqs.split(",") if s)

    lens = None
    if args.lengths:
        lens = tuple(int(s) for s in args.lengths.split(",") if s)

    cfg = PEVisibleConfig(
        seed=args.seed,
        outdir=args.outdir,
        x_axis="freq" if args.x_axis == "freq" else "length",
        trials=args.trials,
        freqs=freqs or PEVisibleConfig.freqs,
        lengths=lens or PEVisibleConfig.lengths,
    )

    df = run_experiment(cfg)
    # Small console sanity check
    print(df.groupby(["scenario", cfg.x_axis == "freq" and "freq_hz" or "T"])["pe_order_actual"].median())

if __name__ == "__main__":
    main()
