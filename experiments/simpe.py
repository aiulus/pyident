"""Persistency-of-excitation identifiability sweep.

This experiment follows the workflow described in the user request:

1. Build two ensembles of (A, B, x0) triples with prescribed visible
   subspace dimensions (one equal to ``n`` and one equal to a target dim).
2. Generate ensembles of persistently exciting input signals with *exact*
   block-PE order drawn from ``--peorders`` using either PRBS or multisine
   generators.
3. Simulate trajectories for every combination of system, initial condition,
   and input signal.
4. Estimate (A, B) using user-selected algorithms and compute relative
   estimation errors (REE).
5. Aggregate the data into CSV tables and produce plots of REE versus PE order
   for both visible-dimension scenarios.

The theoretical minimum PE order required for identifiability with state-input
measurements is the state dimension ``n``.  Classical references (e.g.,
Chapter 2 of Van Overschee & De Moor, *Subspace Identification for Linear
Systems*, 1996; Markovsky, *Structured Low-Rank Approximation and its
Applications*, 2019; Ljung, *System Identification*, 2nd ed., 1999)
show that block Hankel matrices of depth ``n`` (order ``n`` in the PE sense)
are necessary and sufficient to guarantee that the regressor matrix built from
``[x_k; u_k]`` attains full row rank, making (A, B) identifiable from state-input
trajectories.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..estimators import dmdc_fit, dmdc_iv, dmdc_tls, moesp_fit
from ..experiments.visible_sampling import (
    VisibleDrawConfig,
    construct_x0_with_dim_visible,
    prepare_system_with_visible_dim,
)
from ..signals import estimate_pe_order, multisine as multisine_signal, prbs as prbs_signal
from ..simulation import simulate_dt


def minimal_pe_order_for_identifiability(n: int, m: int) -> int:
    """Return the theoretical minimum block-PE order required for DD-ID."""

    # For discrete-time LTI systems with full state measurements, the augmented
    # regressor matrix [X; U] must have rank n + m.  Order-n block Hankel
    # matrices guarantee full rank for the input component, enabling unique
    # least-squares solutions for (A, B).
    if n <= 0:
        raise ValueError("State dimension must be positive.")
    if m <= 0:
        raise ValueError("Input dimension must be positive.")
    return int(n)


@dataclass
class IdentifiabilityConfig:
    """Configuration for the PE identifiability sweep."""

    n: int = 5
    m: int = 2
    dt: float = 0.05
    ntrials: int = 200
    target_visible: int = 3
    seed: int = 0
    peorders: Sequence[int] = dataclasses.field(default_factory=lambda: tuple(range(1, 6)))
    sigtype: str = "prbs"
    algos: Sequence[str] = dataclasses.field(default_factory=lambda: ("dmdc_tls",))
    noise_std: float = 0.0
    outdir: str = "out_pe_ident"
    max_signal_tries: int = 512
    moesp_s: int | None = None
    csv_name: str = "ree_results.csv"

    # Visible subspace draw controls
    max_system_attempts: int = 256
    max_x0_attempts: int = 512
    visible_tol: float = 1e-10

    def __post_init__(self) -> None:
        if self.n <= 0 or self.m <= 0:
            raise ValueError("n and m must be positive integers.")
        if self.target_visible <= 0 or self.target_visible > self.n:
            raise ValueError("target_visible must satisfy 1 <= target_visible <= n.")
        if self.ntrials <= 0:
            raise ValueError("ntrials must be positive.")
        if not self.peorders:
            raise ValueError("peorders cannot be empty.")
        if min(self.peorders) <= 0:
            raise ValueError("peorders must contain positive integers.")
        if self.sigtype not in {"prbs", "multisine"}:
            raise ValueError("sigtype must be 'prbs' or 'multisine'.")
        self.algos = tuple(a.lower() for a in self.algos)


def _minimal_horizon(order: int, m: int) -> int:
    # Block Hankel with depth 'order' has (m*order) rows and (T - order + 1)
    # columns.  Full row rank requires columns >= rows ⇒ T >= m*order + order - 1.
    return int(m * order + order - 1)


def _check_exact_order(U: np.ndarray, order: int) -> bool:
    r = int(estimate_pe_order(U, s_max=order))
    if r != order:
        return False
    # Minimal horizon choice guarantees order+1 cannot be full row rank, but we
    # still verify numerically for robustness.
    r_plus = int(estimate_pe_order(U, s_max=order + 1))
    return r_plus < order + 1


def _draw_prbs_exact(order: int, m: int, rng: np.random.Generator, max_tries: int) -> np.ndarray:
    T = _minimal_horizon(order, m)
    period = max(2, 2 * order)
    for _ in range(max_tries):
        U = prbs_signal(T, m, rng, period=period)
        if _check_exact_order(U, order):
            return U
    raise RuntimeError(
        f"Failed to draw PRBS input with exact PE order {order} after {max_tries} attempts."
    )


def _draw_multisine_exact(order: int, m: int, rng: np.random.Generator, max_tries: int) -> np.ndarray:
    T = _minimal_horizon(order, m)
    k_lines = max(4, min(T // 2, order * m))
    for _ in range(max_tries):
        U = multisine_signal(T, m, rng, k_lines=k_lines)
        if _check_exact_order(U, order):
            return U
    raise RuntimeError(
        f"Failed to draw multisine input with exact PE order {order} after {max_tries} attempts."
    )


def generate_signal(order: int, cfg: IdentifiabilityConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.sigtype == "prbs":
        return _draw_prbs_exact(order, cfg.m, rng, cfg.max_signal_tries)
    return _draw_multisine_exact(order, cfg.m, rng, cfg.max_signal_tries)


def _relative_estimation_error(A: np.ndarray, B: np.ndarray, Ahat: np.ndarray, Bhat: np.ndarray) -> float:
    ref = np.hstack([A, B])
    est = np.hstack([Ahat, Bhat])
    denom = float(np.linalg.norm(ref, ord="fro"))
    return float(np.linalg.norm(est - ref, ord="fro") / (denom + 1e-12))


def _available_estimators(cfg: IdentifiabilityConfig) -> Dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    def wrap_dmdc(X: np.ndarray, Xp: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return dmdc_fit(X, Xp, U)

    def wrap_dmdc_tls(X: np.ndarray, Xp: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return dmdc_tls(X, Xp, U)

    def wrap_dmdc_iv(X: np.ndarray, Xp: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        L = max(1, min(cfg.n, U.shape[0] // 4))
        return dmdc_iv(X, Xp, U, L=L)

    def wrap_moesp(X: np.ndarray, Xp: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = cfg.moesp_s or max(cfg.n, min(U.shape[0] // 2, cfg.n + cfg.m))
        Ahat, Bhat, *_ = moesp_fit(U, X.T, s=s, n=cfg.n)
        return Ahat, Bhat

    mapping: Dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = {
        "dmdc": wrap_dmdc,
        "dmdc_tls": wrap_dmdc_tls,
        "dmdc_iv": wrap_dmdc_iv,
        "moesp": wrap_moesp,
    }
    return mapping


def _draw_systems(cfg: IdentifiabilityConfig, target_dim: int, rng: np.random.Generator) -> List[Dict[str, np.ndarray]]:
    systems: List[Dict[str, np.ndarray]] = []
    draw_cfg = VisibleDrawConfig(
        n=cfg.n,
        m=cfg.m,
        dt=cfg.dt,
        dim_visible=target_dim,
        max_system_attempts=cfg.max_system_attempts,
        max_x0_attempts=cfg.max_x0_attempts,
        tol=cfg.visible_tol,
        deterministic_x0=False,
    )
    for _ in range(cfg.ntrials):
        A, B, Ad, Bd, Rbasis = prepare_system_with_visible_dim(draw_cfg, rng)
        x0, Vbasis = construct_x0_with_dim_visible(
            Ad,
            Bd,
            Rbasis,
            target_dim,
            rng,
            tol=cfg.visible_tol,
            max_tries=cfg.max_x0_attempts,
        )
        systems.append({
            "A": Ad,
            "B": Bd,
            "x0": x0,
            "Vbasis": Vbasis,
            "dim_visible": Vbasis.shape[1],
        })
    return systems


def run_experiment(cfg: IdentifiabilityConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    visible_target = _draw_systems(cfg, cfg.target_visible, rng)
    visible_full = _draw_systems(cfg, cfg.n, rng)

    estimator_map = _available_estimators(cfg)
    missing = [a for a in cfg.algos if a not in estimator_map]
    if missing:
        raise ValueError(f"Unknown estimators requested: {missing}. Available: {sorted(estimator_map)}")

    rows: List[Dict[str, object]] = []

    signals: Dict[int, np.ndarray] = {}
    for order in sorted(set(cfg.peorders)):
        signals[order] = generate_signal(order, cfg, rng)

    combos: Iterable[Tuple[str, Dict[str, np.ndarray]]] = (
        [("target", sys) for sys in visible_target]
        + [("full", sys) for sys in visible_full]
    )

    for scenario, sys in combos:
        A = sys["A"]
        B = sys["B"]
        x0 = sys["x0"]
        dim_visible = int(sys["dim_visible"])

        for order, U in signals.items():
            T = U.shape[0]
            X = simulate_dt(x0, A, B, U, noise_std=cfg.noise_std, rng=rng)
            Xk = X[:, :-1]
            Xkp1 = X[:, 1:]

            for algo_name in cfg.algos:
                estimator = estimator_map[algo_name]
                try:
                    Ahat, Bhat = estimator(Xk, Xkp1, U)
                except Exception as exc:  # pragma: no cover - defensive
                    rows.append({
                        "scenario": scenario,
                        "algo": algo_name,
                        "pe_order": order,
                        "ree": np.nan,
                        "error": str(exc),
                        "dim_visible": dim_visible,
                    })
                    continue

                ree = _relative_estimation_error(A, B, Ahat, Bhat)
                rows.append({
                    "scenario": scenario,
                    "algo": algo_name,
                    "pe_order": order,
                    "ree": ree,
                    "dim_visible": dim_visible,
                })

    df = pd.DataFrame(rows)
    return df


def _plot_group(df: pd.DataFrame, scenario: str, outpath: str) -> None:
    subset = df[df["scenario"] == scenario]
    if subset.empty:
        return

    grouped = (
        subset.groupby(["algo", "pe_order"])  # type: ignore[arg-type]
        ["ree"].median()
        .reset_index()
    )

    plt.figure(figsize=(6.4, 4.8))
    for algo in sorted(subset["algo"].unique()):
        series = grouped[grouped["algo"] == algo]
        plt.plot(series["pe_order"], series["ree"], marker="o", label=algo.upper())
    plt.xlabel("PE order")
    plt.ylabel("Relative Estimation Error")
    plt.title(f"Scenario: {scenario} (median REE)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def save_outputs(df: pd.DataFrame, cfg: IdentifiabilityConfig) -> None:
    os.makedirs(cfg.outdir, exist_ok=True)
    csv_path = os.path.join(cfg.outdir, cfg.csv_name)
    df.to_csv(csv_path, index=False)

    plot_target = os.path.join(cfg.outdir, "ree_vs_pe_target.png")
    plot_full = os.path.join(cfg.outdir, "ree_vs_pe_full.png")
    _plot_group(df, "target", plot_target)
    _plot_group(df, "full", plot_full)

    meta = {
        "config": dataclasses.asdict(cfg),
        "pe_requirement": minimal_pe_order_for_identifiability(cfg.n, cfg.m),
        "csv": csv_path,
        "plots": {"target": plot_target, "full": plot_full},
    }
    with open(os.path.join(cfg.outdir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def parse_args(argv: Sequence[str] | None = None) -> IdentifiabilityConfig:
    parser = argparse.ArgumentParser(description="PE identifiability sweep")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--ntrials", type=int, default=200)
    parser.add_argument("--target-visible", type=int, default=3)
    parser.add_argument("--peorders", type=str, default="1:6")
    parser.add_argument("--sigtype", type=str, default="prbs", choices=["prbs", "multisine"])
    parser.add_argument("--algos", type=str, default="dmdc_tls")
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="out_pe_ident")
    parser.add_argument("--max-signal-tries", type=int, default=512)
    parser.add_argument("--moesp-s", type=int, default=None)

    args = parser.parse_args(argv)

    if ":" in args.peorders:
        parts = [int(p) for p in args.peorders.split(":") if p]
        if len(parts) == 2:
            start, stop = parts
            orders = tuple(range(start, stop))
        elif len(parts) == 3:
            start, stop, step = parts
            orders = tuple(range(start, stop, step))
        else:
            raise ValueError("Bad --peorders range. Use start:stop[:step].")
    else:
        orders = tuple(int(x) for x in args.peorders.replace(",", " ").split())

    algos = [token.strip() for token in args.algos.replace(",", " ").split() if token.strip()]
    cfg = IdentifiabilityConfig(
        n=args.n,
        m=args.m,
        dt=args.dt,
        ntrials=args.ntrials,
        target_visible=args.target_visible,
        seed=args.seed,
        peorders=orders,
        sigtype=args.sigtype,
        algos=tuple(algos),
        noise_std=args.noise_std,
        outdir=args.outdir,
        max_signal_tries=args.max_signal_tries,
        moesp_s=args.moesp_s,
    )
    return cfg


def main(argv: Sequence[str] | None = None) -> None:
    cfg = parse_args(argv)
    df = run_experiment(cfg)
    save_outputs(df, cfg)


if __name__ == "__main__":  # pragma: no cover
    main()