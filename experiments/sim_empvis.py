from __future__ import annotations

from dataclasses import dataclass, field
import pathlib
from typing import List

import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..projectors import projector_from_basis, visible_from_traj
from ..signals import estimate_pe_order_block
from ..simulation import prbs, simulate_dt
from .visible_sampling import (
    VisibleDrawConfig,
    construct_x0_with_dim_visible,
    draw_system_state_with_visible_dim,
    reachable_basis,
    sample_visible_initial_state,
)


@dataclass
class VisibleSubspaceApproxConfig(ExperimentConfig):
    """
    Empirical visible subspace approximation experiment.

    Objective: quantify how well V_emp(x0) = span{x_0,...,x_{T-1}} recovers the
    structural visible subspace under ZOH sampling and PRBS excitation.
    """

    n: int = 8
    m: int = 2
    dim_visible_grid: tuple[int, ...] = (2, 4, 6)

    dt: float = 0.1
    T: int = 200
    T_grid: tuple[int, ...] = (10, 20, 40, 80, 160)

    n_systems: int = 10
    n_x0_per_system: int = 4
    n_signal_realizations: int = 4

    u_scale: float = 1.0
    pe_order_max: int = 16

    ensemble: str = "stable"

    tol: float = 1e-10
    deterministic_x0: bool = False
    force_hurwitz: bool = True
    stability_margin: float = 0.05
    max_system_attempts: int = 128
    max_x0_attempts: int = 256

    save_dir: pathlib.Path = field(
        default_factory=lambda: pathlib.Path("out_vis_subspace")
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.save_dir.mkdir(parents=True, exist_ok=True)


def draw_prbs_for_dim(
    T: int,
    m: int,
    dim_visible: int,
    u_scale: float,
    rng: np.random.Generator,
    s_max: int,
) -> tuple[np.ndarray, int, int]:
    """
    Draw a PRBS input for the given visible dimension and record PE diagnostics.
    """
    dwell = 1
    U = prbs(T=T, m=m, scale=u_scale, dwell=dwell, rng=rng)
    try:
        pe_est = int(estimate_pe_order_block(U, s_max=s_max))
    except Exception:
        pe_est = 0
    return U, dwell, pe_est


def _visible_subspace_trials(
    cfg: VisibleSubspaceApproxConfig, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Core experiment loop testing convergence of the empirical visible subspace.

    For each target visible dimension k, systems (A, B) are drawn with reachable
    rank k, initial states with dim V(x0) = k are sampled, the system is
    simulated under PRBS input, and span{x_0,...,x_{T-1}} is compared to the
    structural visible subspace via dimension error and projector distance.
    """
    records: List[dict[str, float | int]] = []

    for k in cfg.dim_visible_grid:
        draw_cfg = VisibleDrawConfig(
            n=cfg.n,
            m=cfg.m,
            dt=cfg.dt,
            dim_visible=int(k),
            ensemble=cfg.ensemble,
            max_system_attempts=cfg.max_system_attempts,
            max_x0_attempts=cfg.max_x0_attempts,
            tol=cfg.tol,
            deterministic_x0=cfg.deterministic_x0,
            force_hurwitz=cfg.force_hurwitz,
            stability_margin=cfg.stability_margin,
        )

        for sys_idx in range(cfg.n_systems):
            A, B, Ad, Bd, x0_base, V_true_base = draw_system_state_with_visible_dim(
                draw_cfg, rng
            )
            Rbasis = reachable_basis(Ad, Bd, tol=cfg.tol)

            for x_idx in range(cfg.n_x0_per_system):
                if x_idx == 0:
                    x0 = np.asarray(x0_base, dtype=float)
                    V_true = V_true_base
                else:
                    if cfg.deterministic_x0:
                        x0, V_true = construct_x0_with_dim_visible(
                            Ad, Bd, Rbasis, k, rng, tol=cfg.tol
                        )
                    else:
                        x0, V_true = sample_visible_initial_state(
                            Ad,
                            Bd,
                            Rbasis,
                            k,
                            rng,
                            max_attempts=cfg.max_x0_attempts,
                            tol=cfg.tol,
                        )

                P_true = projector_from_basis(V_true)
                k_true = V_true.shape[1]

                for sig_idx in range(cfg.n_signal_realizations):
                    U, dwell, pe_est = draw_prbs_for_dim(
                        T=cfg.T,
                        m=cfg.m,
                        dim_visible=k,
                        u_scale=cfg.u_scale,
                        rng=rng,
                        s_max=min(cfg.pe_order_max, cfg.T // 2),
                    )

                    X = simulate_dt(x0, Ad, Bd, U, noise_std=0.0, rng=rng)
                    X_full = X[:, :-1]

                    for T_eff in cfg.T_grid:
                        if T_eff > X_full.shape[1]:
                            continue

                        X_slice = X_full[:, :T_eff]
                        V_emp = visible_from_traj(X_slice, tol=cfg.tol)
                        P_emp = projector_from_basis(V_emp)
                        k_emp = V_emp.shape[1]

                        if P_emp.shape == P_true.shape:
                            proj_err = float(
                                np.linalg.norm(P_true - P_emp, ord="fro")
                            )
                        else:
                            proj_err = float("nan")

                        records.append(
                            {
                                "dim_visible_target": int(k),
                                "dim_visible_true": int(k_true),
                                "dim_visible_emp": int(k_emp),
                                "dim_err": int(abs(k_emp - k_true)),
                                "proj_err": proj_err,
                                "T_eff": int(T_eff),
                                "T": int(cfg.T),
                                "dwell": int(dwell),
                                "pe_order_est": int(pe_est),
                                "n": int(cfg.n),
                                "m": int(cfg.m),
                                "sys_idx": int(sys_idx),
                                "x_idx": int(x_idx),
                                "sig_idx": int(sig_idx),
                            }
                        )

    return pd.DataFrame.from_records(records)


def run_visible_subspace_experiment(
    cfg: VisibleSubspaceApproxConfig,
) -> pd.DataFrame:
    """
    Public entry point: run the experiment and save a CSV of the results.
    """
    rng = np.random.default_rng(cfg.seed)
    df = _visible_subspace_trials(cfg, rng)

    out_csv = cfg.save_dir / "vis_subspace_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    return df
