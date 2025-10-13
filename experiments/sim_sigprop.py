"""Signal-property sweeps vs visible subspace dimension.

This experiment mirrors :mod:`pyident.experiments.sim_pe` but replaces the
explicit PEness sweep with targeted variations of a single signal property.

Motivation
----------
Previous runs focused on varying the block persistency-of-excitation order in
order to relate theoretical bounds with practical identification accuracy.  To
understand how *other* signal covariates confound the relationship between
visibility and estimation error, we expose the main PRBS signal knobs and vary
one at a time.  For each configuration we draw ensembles of
``(A, B, x0)`` whose visible subspace dimension takes one of three canonical
values:

``k ∈ {n, n-2, n-4}``

For every signal property value we simulate trajectories, fit the system with a
consistent MOESP estimator, and record Frobenius relative errors.  Results are
summarized via box plots showing the distribution of estimation errors across
the three visibility regimes.  Post-hoc PE diagnostics are still reported to
aid interpretation.
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
from ..estimators import moesp_fit
from ..metrics import projected_errors, build_visible_basis_dt
from ..signals import estimate_moment_pe_order, estimate_pe_order
from ..simulation import prbs, simulate_dt
from .visible_sampling import VisibleDrawConfig, draw_system_state_with_visible_dim


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROPERTY_METADATA = {
    "T": {
        "label": "Horizon length",
        "type": int,
        "values": (120, 200, 320),
    },
    "dwell": {
        "label": "PRBS dwell",
        "type": int,
        "values": (1, 2, 4, 8),
    },
    "u_scale": {
        "label": "Input scale",
        "type": float,
        "values": (1.0, 2.5, 4.0),
    },
}


@dataclass
class SignalPropertyConfig(ExperimentConfig):
    """Configuration for the signal-property sweep."""

    n: int = 6
    m: int = 2
    dt: float = 0.1
    T: int = 200
    dwell: int = 1
    u_scale: float = 3.0
    noise_std: float = 0.0

    signal_property: str = "u_scale"
    property_values: Sequence[float | int] | None = None

    ntrials: int = 25
    n_systems: int = 100

    max_system_attempts: int = 500
    max_x0_attempts: int = 256
    visible_tol: float = 1e-8
    eps_norm: float = 1e-12
    deterministic_x0: bool = False

    pe_s_max: int = 50  # Increased from 12 to allow measurement of true PE orders  
    pe_moment_r_max: int = 50  # Increased from 12 to allow measurement of true PE orders

    outdir: str = "out_signal_props"

    def __post_init__(self) -> None:  # type: ignore[override]
        super().__post_init__()

        if self.signal_property not in _PROPERTY_METADATA:
            allowed = ", ".join(sorted(_PROPERTY_METADATA))
            raise ValueError(
                f"Unknown signal property '{self.signal_property}'. "
                f"Choose one of: {allowed}."
            )

        if self.ntrials < 1:
            raise ValueError("ntrials must be positive.")
        if self.n_systems < 1:
            raise ValueError("n_systems must be positive.")
        if self.T <= 2:
            raise ValueError("T must be at least 3 samples.")
        if self.dwell <= 0:
            raise ValueError("dwell must be positive.")
        if self.u_scale <= 0:
            raise ValueError("u_scale must be positive.")

        # Canonical visibility regimes: n, n-2, n-4 (filtered to valid values)
        dims = [self.n]
        if self.n - 2 > 0:
            dims.append(self.n - 2)
        if self.n - 4 > 0 and self.n - 4 not in dims:
            dims.append(self.n - 4)
        self.visible_dims: Tuple[int, ...] = tuple(sorted(dims, reverse=True))

        meta = _PROPERTY_METADATA[self.signal_property]
        if self.property_values is None:
            self.property_values = meta["values"]

        if not self.property_values:
            raise ValueError("property_values cannot be empty.")

        # Ensure proper typing and monotonicity for nicer plots
        parser = meta["type"]
        parsed: List[float | int] = [parser(v) for v in self.property_values]
        self.property_values = tuple(parsed)

        if self.signal_property == "T":
            if min(int(v) for v in self.property_values) <= 2:
                raise ValueError("All horizon lengths must be >= 3 samples.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_system_dict(
    A: np.ndarray,
    B: np.ndarray,
    Ad: np.ndarray,
    Bd: np.ndarray,
    x0: np.ndarray,
    P: np.ndarray,
    dim_visible: int,
    system_id: int,
) -> Dict[str, float | int | np.ndarray]:
    return {
        "A": A,
        "B": B,
        "Ad": Ad,
        "Bd": Bd,
        "x0": x0,
        "P": P,
        "dim_visible": dim_visible,
        "system_id": system_id,
    }


def _draw_system_with_visible_dim(
    cfg: SignalPropertyConfig,
    target_dim: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    draw_cfg = VisibleDrawConfig(
        n=cfg.n,
        m=cfg.m,
        dt=cfg.dt,
        dim_visible=target_dim,
        ensemble="stable",
        max_system_attempts=cfg.max_system_attempts,
        max_x0_attempts=cfg.max_x0_attempts,
        tol=cfg.visible_tol,
        deterministic_x0=cfg.deterministic_x0,
    )

    for _ in range(cfg.max_system_attempts):
        A, B, Ad, Bd, x0, P = draw_system_state_with_visible_dim(draw_cfg, rng)
        Qv = build_visible_basis_dt(Ad, Bd, x0, tol=cfg.visible_tol)
        if Qv.shape[1] == target_dim:
            return A, B, Ad, Bd, x0, Qv

    raise RuntimeError(
        "Failed to draw a system with the requested visible subspace dimension "
        f"{target_dim}."
    )


def _relative_norm(err: float, ref: float, eps: float) -> float:
    return float(err / max(ref, eps))


def _format_property_value(values: Sequence[float | int]) -> List[str]:
    formatted = []
    for v in values:
        if isinstance(v, int) or float(v).is_integer():
            formatted.append(f"{int(v)}")
        else:
            formatted.append(f"{float(v):.3g}")
    return formatted


def _make_signal(
    cfg: SignalPropertyConfig,
    prop: str,
    value: float | int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    T = cfg.T
    dwell = cfg.dwell
    scale = cfg.u_scale

    if prop == "T":
        T = int(value)
    elif prop == "dwell":
        dwell = int(value)
    elif prop == "u_scale":
        scale = float(value)
    else:
        raise ValueError(f"Unsupported property '{prop}'.")

    if dwell <= 0:
        raise ValueError("dwell must remain positive during sweep.")
    if T <= 2:
        raise ValueError("T must remain at least 3 during sweep.")

    U = prbs(T, cfg.m, scale=scale, dwell=dwell, rng=rng)

    s_cap = min(cfg.pe_s_max, max(1, T // 2))
    r_cap = min(cfg.pe_moment_r_max, max(1, T // 2))
    pe_block = int(estimate_pe_order(U, s_max=s_cap))
    pe_moment = int(estimate_moment_pe_order(U, r_max=r_cap, dt=cfg.dt))

    info = {
        "T": float(T),
        "dwell": float(dwell),
        "u_scale": float(scale),
        "pe_order_actual": float(pe_block),
        "pe_order_moment": float(pe_moment),
    }
    return U, info


def _run_trial(
    label: str,
    U: np.ndarray,
    info: Dict[str, float],
    system: Dict[str, float | int | np.ndarray],
    cfg: SignalPropertyConfig,
) -> Dict[str, float]:
    x0 = system["x0"]
    Ad = system["Ad"]
    Bd = system["Bd"]
    P = system["P"]
    assert isinstance(x0, np.ndarray)
    assert isinstance(Ad, np.ndarray)
    assert isinstance(Bd, np.ndarray)
    assert isinstance(P, np.ndarray)

    X = simulate_dt(x0, Ad, Bd, U, noise_std=cfg.noise_std)
    X0, X1 = X[:, :-1], X[:, 1:]
    # Ensure channel-major inputs for the wrapper
    Ahat, Bhat = moesp_fit(X0, X1, U.T, n=cfg.n)

    errA = float(np.linalg.norm(Ahat - Ad, ord="fro"))
    errB = float(np.linalg.norm(Bhat - Bd, ord="fro"))
    normA = float(np.linalg.norm(Ad, ord="fro"))
    normB = float(np.linalg.norm(Bd, ord="fro"))

    Q, _ = np.linalg.qr(P)
    dA_V, dB_V = projected_errors(Ahat, Bhat, Ad, Bd, Q)
    A_vis = float(np.linalg.norm(Q.T @ Ad @ Q, ord="fro"))
    B_vis = float(np.linalg.norm(Q.T @ Bd, ord="fro"))

    return {
        "scenario": label,
        "errA_rel": _relative_norm(errA, normA, cfg.eps_norm),
        "errB_rel": _relative_norm(errB, normB, cfg.eps_norm),
        "errA_V_rel": _relative_norm(float(dA_V), max(A_vis, cfg.eps_norm), cfg.eps_norm),
        "errB_V_rel": _relative_norm(float(dB_V), max(B_vis, cfg.eps_norm), cfg.eps_norm),
        "dim_visible": int(system["dim_visible"]),
        "system_id": int(system["system_id"]),
        **info,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_error_boxplots(
    df: pd.DataFrame,
    cfg: SignalPropertyConfig,
    outfile: pathlib.Path,
) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)

    prop_values = list(cfg.property_values)
    labels = _format_property_value(prop_values)
    prop_label = _PROPERTY_METADATA[cfg.signal_property]["label"]

    fig, axes = plt.subplots(
        2,
        len(cfg.visible_dims),
        figsize=(4 * len(cfg.visible_dims), 8),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    for col, dim in enumerate(cfg.visible_dims):
        scenario_label = f"k={dim}"
        sub = df[df["dim_visible"] == dim]

        dataA = [
            sub[np.isclose(sub["property_value"], float(v))]["errA_rel"].to_numpy()
            for v in prop_values
        ]
        dataB = [
            sub[np.isclose(sub["property_value"], float(v))]["errB_rel"].to_numpy()
            for v in prop_values
        ]

        axes[0, col].boxplot(dataA, labels=labels, patch_artist=True)
        axes[1, col].boxplot(dataB, labels=labels, patch_artist=True)

        axes[0, col].set_title(scenario_label)
        axes[1, col].set_xlabel(prop_label)
        if col == 0:
            axes[0, col].set_ylabel("‖Â−A‖₍F₎ / ‖A‖₍F₎")
            axes[1, col].set_ylabel("‖ B̂−B ‖₍F₎ / ‖B‖₍F₎")
        axes[0, col].grid(True, axis="y", alpha=0.3)
        axes[1, col].grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Estimation error vs {prop_label} (MOESP, n={cfg.n}, property={cfg.signal_property})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outfile, dpi=160)
    plt.close(fig)


def _plot_posthoc_pe(
    df: pd.DataFrame,
    cfg: SignalPropertyConfig,
    outfile: pathlib.Path,
) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    prop_values = list(cfg.property_values)
    labels = _format_property_value(prop_values)
    prop_label = _PROPERTY_METADATA[cfg.signal_property]["label"]

    fig, axes = plt.subplots(
        1,
        len(cfg.visible_dims),
        figsize=(4 * len(cfg.visible_dims), 4),
        sharex=True,
        sharey=True,
    )

    positions = np.arange(1, len(prop_values) + 1)

    for ax, dim in zip(np.atleast_1d(axes), cfg.visible_dims):
        sub = df[df["dim_visible"] == dim]
        data_block = [
            sub[np.isclose(sub["property_value"], float(v))]["pe_order_actual"].to_numpy()
            for v in prop_values
        ]
        data_moment = [
            sub[np.isclose(sub["property_value"], float(v))]["pe_order_moment"].to_numpy()
            for v in prop_values
        ]

        ax.boxplot(data_block, positions=positions - 0.15, widths=0.25, labels=labels, patch_artist=True)
        ax.boxplot(data_moment, positions=positions + 0.15, widths=0.25, patch_artist=True)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_title(f"k={dim}")
        ax.set_xlabel(prop_label)
        ax.set_ylabel("Estimated PE order")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Post-hoc PE vs {prop_label}")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(outfile, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------


def run_experiment(cfg: SignalPropertyConfig) -> pd.DataFrame:
    base_rng = np.random.default_rng(cfg.seed)
    sys_rng = np.random.default_rng(base_rng.integers(0, 2**32))
    input_rng = np.random.default_rng(base_rng.integers(0, 2**32))

    scenarios: List[Dict[str, float | int | np.ndarray]] = []
    for idx, dim in enumerate(cfg.visible_dims):
        systems_for_dim: List[Dict[str, float | int | np.ndarray]] = []
        for j in range(cfg.n_systems):
            A, B, Ad, Bd, x0, P = _draw_system_with_visible_dim(cfg, target_dim=dim, rng=sys_rng)
            systems_for_dim.append(
                _create_system_dict(A, B, Ad, Bd, x0, P, dim_visible=dim, system_id=j)
            )
        scenarios.append({"dim": dim, "systems": systems_for_dim})

    property_values = [float(v) for v in cfg.property_values]
    records: List[Dict[str, float]] = []

    for val in property_values:
        for trial in range(cfg.ntrials):
            U, info = _make_signal(cfg, cfg.signal_property, val, input_rng)
            info["property_value"] = val
            info["signal_property"] = cfg.signal_property

            for scenario in scenarios:
                dim = scenario["dim"]
                systems = scenario["systems"]
                label = f"k={dim}"
                for system in systems:
                    result = _run_trial(label, U, info, system, cfg)
                    result["property_value"] = val
                    result["signal_property"] = cfg.signal_property
                    result["trial"] = float(trial)
                    records.append(result)

    df = pd.DataFrame.from_records(records)

    outdir = pathlib.Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"signal_property_{cfg.signal_property}.csv"
    df.to_csv(outfile, index=False)

    _plot_error_boxplots(df, cfg, outdir / f"errors_vs_{cfg.signal_property}.png")
    _plot_posthoc_pe(df, cfg, outdir / f"pe_vs_{cfg.signal_property}.png")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_values(raw: str, prop: str) -> Sequence[float | int] | None:
    if not raw:
        return None
    parser = _PROPERTY_METADATA[prop]["type"]
    values: List[float | int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(parser(token))
    return tuple(values)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--outdir", type=str, default="out_signal_props", help="Output directory")
    parser.add_argument("--n", type=int, default=6, help="State dimension")
    parser.add_argument("--m", type=int, default=2, help="Input dimension")
    parser.add_argument("--dt", type=float, default=0.1, help="Sampling time step")
    parser.add_argument("--T", type=int, default=200, help="Baseline signal horizon")
    parser.add_argument("--dwell", type=int, default=1, help="Baseline PRBS dwell time")
    parser.add_argument("--u-scale", type=float, default=3.0, help="Baseline PRBS amplitude")
    parser.add_argument("--noise-std", type=float, default=0.0, help="Process noise standard deviation")

    parser.add_argument(
        "--signal-property",
        type=str,
        default="u_scale",
        choices=sorted(_PROPERTY_METADATA.keys()),
        help="Signal property to sweep",
    )
    parser.add_argument(
        "--property-values",
        type=str,
        default="",
        help="Comma-separated list of property values (at most one property per run)",
    )

    parser.add_argument("--n-trials", type=int, default=100, help="Trials per property value")
    parser.add_argument("--n-systems", type=int, default=100, help="Systems per visibility scenario")
    parser.add_argument("--max-system-attempts", type=int, default=500, help="Maximum system draw attempts")
    parser.add_argument("--max-x0-attempts", type=int, default=256, help="Maximum x0 draw attempts")
    parser.add_argument("--visible-tol", type=float, default=1e-8, help="Visible subspace tolerance")
    parser.add_argument("--eps-norm", type=float, default=1e-12, help="Epsilon for relative norms")
    parser.add_argument("--det", action="store_true", help="Use deterministic x0 construction")
    parser.add_argument("--pe-s-max", type=int, default=50, help="Maximum block-PE depth for diagnostics")  # Increased from 12
    parser.add_argument("--pe-moment-r-max", type=int, default=50, help="Maximum moment-PE order for diagnostics")  # Increased from 12

    args = parser.parse_args(argv)

    prop_values = _parse_values(args.property_values, args.signal_property)

    cfg = SignalPropertyConfig(
        seed=args.seed,
        outdir=args.outdir,
        n=args.n,
        m=args.m,
        dt=args.dt,
        T=args.T,
        dwell=args.dwell,
        u_scale=args.u_scale,
        noise_std=args.noise_std,
        signal_property=args.signal_property,
        property_values=prop_values,
        ntrials=args.n_trials,
        n_systems=args.n_systems,
        max_system_attempts=args.max_system_attempts,
        max_x0_attempts=args.max_x0_attempts,
        visible_tol=args.visible_tol,
        eps_norm=args.eps_norm,
        deterministic_x0=args.det,
        pe_s_max=args.pe_s_max,
        pe_moment_r_max=args.pe_moment_r_max,
    )

    df = run_experiment(cfg)
    group_col = "property_value"
    print(
        df.groupby(["scenario", group_col])["errA_rel"].median().unstack("scenario")
    )


if __name__ == "__main__":
    main()