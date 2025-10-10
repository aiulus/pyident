"""Estimator consistency experiment.

This module implements the experiment sketched in the project notes:

1.  Correlate identifiability scores computed directly from ``(A, B, x0)``
    with the estimation errors obtained by a set of learners (DMDc variants
    and MOESP).  We also record whether each estimate lies inside the
    predicted equivalence class ``[A, B]_{x0}``.
2.  Use the equivalence-class membership tests to assess estimator
    consistency on partially identifiable instances.
3.  Map the regimes of ``(A, B)`` – sparsity, state dimension ``n`` and
    underactuation – that lead to poor identifiability according to the
    proposed score.
4.  Probe the effect of persistency of excitation by varying the PE order
    of a PRBS input signal and comparing estimation errors when the visible
    subspace dimension is ``k < n`` versus the fully visible case ``k = n``.

The experiment relies exclusively on core modules from :mod:`pyident`.
Results are returned as pandas DataFrames and are also written to
``out_estimator_consistency`` for convenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pathlib

import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..ensembles import (
    ginibre,
    sparse_continuous,
    stable,
    draw_with_ctrb_rank,
)
from ..metrics import (
    cont2discrete_zoh,
    eta0,
    projected_errors,
    regressor_stats,
    same_equiv_class_dt_rel,
    unified_generator,
)
from ..simulation import prbs as prbs_dt
from ..simulation import simulate_dt
from ..signals import estimate_pe_order
from ..estimators import (
    dmdc_pinv,
    dmdc_tls,
    moesp_fit,
)
from ..projectors import build_projected_x0


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


@dataclass
class EstimatorConsistencyConfig(ExperimentConfig):
    """Configuration specific to the estimator-consistency experiment."""

    n_systems: int = 12
    """Number of distinct (A, B) pairs sampled for the correlation study."""

    n_x0_per_system: int = 24
    """Number of initial states sampled on the unit sphere per system."""

    # Sweep settings ----------------------------------------------------
    sparsity_grid: Sequence[float] = (0.15, 0.3, 0.5, 0.7, 0.9)
    state_dim_grid: Sequence[int] = (4, 6, 8, 10, 12)
    underactuation_grid: Sequence[int] = (1, 2, 3, 4)
    joint_grid: Sequence[Tuple[int, int, float]] = (
        (4, 1, 0.25),
        (6, 2, 0.35),
        (8, 2, 0.25),
        (10, 3, 0.15),
        (12, 4, 0.15),
    )
    grid_systems: int = 10
    grid_initial_states: int = 20

    # PE sweep ----------------------------------------------------------
    pe_order_max: int = 10
    pe_trials_per_order: int = 8
    pe_dwell_scale: float = 8.0

    noise_std: float = 0.0  # inherit default but keep explicit

    save_dir: pathlib.Path = field(default_factory=lambda: pathlib.Path("out_estimator_consistency"))

    def __post_init__(self) -> None:
        super().__post_init__()
        self.save_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _sample_unit_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(n)
    nrm = float(np.linalg.norm(v))
    if nrm == 0.0:
        return _sample_unit_sphere(n, rng)
    return v / nrm


def _visible_basis(Ad: np.ndarray, Bd: np.ndarray, x0: np.ndarray) -> np.ndarray:
    from ..metrics import build_visible_basis_dt

    return build_visible_basis_dt(Ad, Bd, x0, tol=1e-12)


def _identifiability_summary(Ad: np.ndarray, Bd: np.ndarray, x0: np.ndarray) -> Dict[str, float]:
    """Return identifiability diagnostics derived directly from (Ad, Bd, x0)."""

    K = unified_generator(Ad, Bd, x0, mode="unrestricted")
    if K.size:
        svals = np.linalg.svd(K, compute_uv=False)
        sigma_min = float(svals[-1])
        cond = float(svals[0] / (svals[-1] + 1e-15))
    else:
        sigma_min = 0.0
        cond = np.inf

    Vbasis = _visible_basis(Ad, Bd, x0)
    dim_V = int(Vbasis.shape[1])

    eta = float(eta0(Ad, Bd, x0, rtol=1e-12))

    return {
        "sigma_min": sigma_min,
        "sigma_cond": cond,
        "dim_V": dim_V,
        "eta0": eta,
    }


def _estimation_errors(
    Ahat: np.ndarray,
    Bhat: np.ndarray,
    Ad: np.ndarray,
    Bd: np.ndarray,
    Vbasis: np.ndarray,
) -> Dict[str, float]:
    nrmA = float(np.linalg.norm(Ad, ord="fro") + 1e-15)
    nrmB = float(np.linalg.norm(Bd, ord="fro") + 1e-15)
    errA = float(np.linalg.norm(Ahat - Ad, ord="fro") / nrmA)
    errB = float(np.linalg.norm(Bhat - Bd, ord="fro") / nrmB)

    dA_V, dB_V = projected_errors(Ahat, Bhat, Ad, Bd, Vbasis)
    scale_VA = float(np.linalg.norm(Vbasis.T @ Ad @ Vbasis, ord="fro") + 1e-15)
    scale_VB = float(np.linalg.norm(Vbasis.T @ Bd, ord="fro") + 1e-15)

    return {
        "errA_rel": errA,
        "errB_rel": errB,
        "errA_V_rel": float(dA_V / scale_VA if scale_VA > 0 else dA_V),
        "errB_V_rel": float(dB_V / scale_VB if scale_VB > 0 else dB_V),
    }


def _estimate_algorithms() -> Mapping[str, Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    return {
        "dmdc_pinv": dmdc_pinv,
        "dmdc_tls": dmdc_tls,
        "moesp": moesp_fit,
    }


def _system_draw(cfg: EstimatorConsistencyConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if cfg.ensemble == "ginibre":
        return ginibre(cfg.n, cfg.m, rng)
    if cfg.ensemble == "sparse":
        return sparse_continuous(
            n=cfg.n,
            m=cfg.m,
            rng=rng,
            which=getattr(cfg, "sparse_which", "both"),
            p_density=getattr(cfg, "p_density", 0.5),
            p_density_A=getattr(cfg, "_density_A", None),
            p_density_B=getattr(cfg, "_density_B", None),
        )
    if cfg.ensemble == "stable":
        return stable(cfg.n, cfg.m, rng)
    if cfg.ensemble == "A_stbl_B_ctrb":
        A, B, _meta = draw_with_ctrb_rank(
            cfg.n,
            cfg.m,
            cfg.n,
            rng,
            ensemble_type="stable",
            embed_random_basis=True,
        )
        return A, B
    raise ValueError(f"Unsupported ensemble '{cfg.ensemble}' for consistency experiment.")


def _correlation_trials(cfg: EstimatorConsistencyConfig, rng: np.random.Generator) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    algs = _estimate_algorithms()

    for sys_idx in range(cfg.n_systems):
        A, B = _system_draw(cfg, rng)
        Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)

        for x_idx in range(cfg.n_x0_per_system):
            x0 = _sample_unit_sphere(cfg.n, rng)
            Vbasis = _visible_basis(Ad, Bd, x0)
            ident = _identifiability_summary(Ad, Bd, x0)

            U = prbs_dt(cfg.T, cfg.m, scale=cfg.u_scale, dwell=cfg.dwell, rng=rng)
            X = simulate_dt(x0, Ad, Bd, U, noise_std=cfg.noise_std, rng=rng)
            X0, X1 = X[:, :-1], X[:, 1:]
            U_cm = U.T
            zstats = regressor_stats(X0, U_cm, rtol_rank=1e-12)

            for name, estimator in algs.items():
                try:
                    Ahat, Bhat = estimator(X0, X1, U_cm)
                except Exception as exc:  # pragma: no cover - defensive guard
                    rec = {
                        "system": sys_idx,
                        "x0_id": x_idx,
                        "algo": name,
                        "failed": 1,
                        "failure_msg": str(exc),
                        **ident,
                    }
                    rec.update({k: np.nan for k in ("errA_rel", "errB_rel", "errA_V_rel", "errB_V_rel")})
                    rec.update({k: np.nan for k in zstats})
                    records.append(rec)
                    continue

                errs = _estimation_errors(Ahat, Bhat, Ad, Bd, Vbasis)
                ok_eqv, info_eqv = same_equiv_class_dt_rel(
                    Ad,
                    Bd,
                    Ahat,
                    Bhat,
                    x0,
                    rtol_eq=cfg.rtol_eq_rel if hasattr(cfg, "rtol_eq_rel") else 1e-2,
                    rtol_rank=1e-12,
                    use_leak=True,
                )

                rec = {
                    "system": sys_idx,
                    "x0_id": x_idx,
                    "algo": name,
                    "failed": 0,
                    "eqv_ok": int(ok_eqv),
                    **ident,
                    **errs,
                    **zstats,
                    "eqv_dim_V": info_eqv.get("dim_V", np.nan),
                    "eqv_dA_V": info_eqv.get("dA_V", np.nan),
                    "eqv_dB": info_eqv.get("dB", np.nan),
                    "eqv_leak": info_eqv.get("leak", np.nan),
                }
                records.append(rec)

    df = pd.DataFrame.from_records(records)
    return df


def _sweep_single_axes(cfg: EstimatorConsistencyConfig, rng: np.random.Generator) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    # Sparsity sweep ----------------------------------------------------
    for density in cfg.sparsity_grid:
        for sys_idx in range(cfg.grid_systems):
            A, B = sparse_continuous(cfg.n, cfg.m, rng, which="both", p_density=density)
            Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)
            for x_idx in range(cfg.grid_initial_states):
                x0 = _sample_unit_sphere(cfg.n, rng)
                ident = _identifiability_summary(Ad, Bd, x0)
                rows.append({
                    "axis": "sparsity",
                    "value": float(density),
                    "system": sys_idx,
                    "x0_id": x_idx,
                    **ident,
                })

    # State dimension sweep --------------------------------------------
    for n in cfg.state_dim_grid:
        m = min(cfg.m, max(1, n // 3))
        for sys_idx in range(cfg.grid_systems):
            A, B = ginibre(n, m, rng)
            Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)
            for x_idx in range(cfg.grid_initial_states):
                x0 = _sample_unit_sphere(n, rng)
                ident = _identifiability_summary(Ad, Bd, x0)
                rows.append({
                    "axis": "state_dim",
                    "value": float(n),
                    "system": sys_idx,
                    "x0_id": x_idx,
                    "m": float(m),
                    **ident,
                })

    # Underactuation sweep ---------------------------------------------
    for m in cfg.underactuation_grid:
        m_eff = min(int(m), cfg.n)
        for sys_idx in range(cfg.grid_systems):
            A, B = ginibre(cfg.n, m_eff, rng)
            Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)
            for x_idx in range(cfg.grid_initial_states):
                x0 = _sample_unit_sphere(cfg.n, rng)
                ident = _identifiability_summary(Ad, Bd, x0)
                rows.append({
                    "axis": "underactuation",
                    "value": float(m_eff),
                    "system": sys_idx,
                    "x0_id": x_idx,
                    **ident,
                })

    return pd.DataFrame.from_records(rows)


def _sweep_joint_axes(cfg: EstimatorConsistencyConfig, rng: np.random.Generator) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    for n, m, density in cfg.joint_grid:
        for sys_idx in range(cfg.grid_systems):
            A, B = sparse_continuous(n, m, rng, which="both", p_density=density)
            Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)
            for x_idx in range(cfg.grid_initial_states):
                x0 = _sample_unit_sphere(n, rng)
                ident = _identifiability_summary(Ad, Bd, x0)
                rows.append({
                    "axis": "joint",
                    "value_n": float(n),
                    "value_m": float(m),
                    "value_density": float(density),
                    "system": sys_idx,
                    "x0_id": x_idx,
                    **ident,
                })

    return pd.DataFrame.from_records(rows)


def _prepare_pe_system(
    cfg: EstimatorConsistencyConfig,
    target_dim: int,
    rng: np.random.Generator,
    max_tries: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a system and initial state with ``dim V(x0) == target_dim``."""

    n = 5
    m = min(cfg.m, 2)
    k_off = max(0, n - target_dim)

    for _ in range(max_tries):
        A, B, _meta = draw_with_ctrb_rank(
            n=n,
            m=m,
            r=target_dim,
            rng=rng,
            ensemble_type="stable",
            embed_random_basis=True,
        )
        Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)

        if k_off > 0:
            seed = rng.standard_normal(n)
            x0, _, _ = build_projected_x0(Ad, Bd, seed, k_off=k_off, rng=rng, tol=1e-12)
        else:
            x0 = _sample_unit_sphere(n, rng)

        Vbasis = _visible_basis(Ad, Bd, x0)
        if Vbasis.shape[1] == target_dim:
            return Ad, Bd, x0

    raise RuntimeError(
        f"Unable to synthesize system with dim V(x0) = {target_dim} after {max_tries} attempts."
    )


def _pe_order_sweep(cfg: EstimatorConsistencyConfig, rng: np.random.Generator) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    # Base systems ------------------------------------------------------
    Ad_k3, Bd_k3, x0_k3 = _prepare_pe_system(cfg, target_dim=3, rng=rng)
    Ad_k5, Bd_k5, x0_k5 = _prepare_pe_system(cfg, target_dim=5, rng=rng)

    systems = (
        ("k3", Ad_k3, Bd_k3, x0_k3),
        ("k5", Ad_k5, Bd_k5, x0_k5),
    )

    # Ensure inputs share a consistent number of channels
    m_input = systems[0][2].shape[1]
    if any(Bd.shape[1] != m_input for _, _, Bd, _ in systems):
        raise RuntimeError("PE sweep systems must share the same input dimension.")

    algs = _estimate_algorithms()

    for target_order in range(1, cfg.pe_order_max + 1):
        dwell = max(1, int(round(cfg.pe_dwell_scale / max(1, target_order))))
        for trial in range(cfg.pe_trials_per_order):
            U = prbs_dt(cfg.T, m_input, scale=cfg.u_scale, dwell=dwell, rng=rng)
            pe_est = estimate_pe_order(U, s_max=min(cfg.pe_order_max, cfg.T // 2))

            for label, Ad, Bd, x0 in systems:
                Vbasis = _visible_basis(Ad, Bd, x0)
                X = simulate_dt(x0, Ad, Bd, U, noise_std=cfg.noise_std, rng=rng)
                X0, X1 = X[:, :-1], X[:, 1:]
                U_cm = U.T
                zstats = regressor_stats(X0, U_cm, rtol_rank=1e-12)

                for name, estimator in algs.items():
                    Ahat, Bhat = estimator(X0, X1, U_cm)
                    errs = _estimation_errors(Ahat, Bhat, Ad, Bd, Vbasis)
                    rows.append({
                        "system_label": label,
                        "algo": name,
                        "target_pe": target_order,
                        "estimated_pe": pe_est,
                        "dwell": dwell,
                        "dim_V": Vbasis.shape[1],
                        **errs,
                        **zstats,
                    })

    return pd.DataFrame.from_records(rows)


# ---------------------------------------------------------------------------
# Public experiment entry point
# ---------------------------------------------------------------------------


def run_experiment(cfg: Optional[EstimatorConsistencyConfig] = None) -> Dict[str, pd.DataFrame]:
    """Run the estimator consistency experiment.

    Args:
        cfg: Optional configuration.  When ``None`` a default
            :class:`EstimatorConsistencyConfig` is instantiated.

    Returns:
        Dictionary with the following DataFrames:

        ``correlation``
            Trial-wise identifiability scores, estimation errors and
            equivalence-class checks.
        ``single_axis``
            Identifiability scores for one-axis sweeps (sparsity, ``n``, ``m``).
        ``joint_axes``
            Identifiability scores when varying multiple axes jointly.
        ``pe_sweep``
            Estimation errors as a function of the PE order for the ``k=3`` and
            ``k=5`` systems.
    """

    if cfg is None:
        cfg = EstimatorConsistencyConfig()

    rng = np.random.default_rng(cfg.seed)

    correlation = _correlation_trials(cfg, rng)
    single_axis = _sweep_single_axes(cfg, rng)
    joint_axes = _sweep_joint_axes(cfg, rng)
    pe_sweep = _pe_order_sweep(cfg, rng)

    outputs = {
        "correlation": correlation,
        "single_axis": single_axis,
        "joint_axes": joint_axes,
        "pe_sweep": pe_sweep,
    }

    for name, df in outputs.items():
        df.to_csv(cfg.save_dir / f"{name}.csv", index=False)

    return outputs


def main() -> None:
    cfg = EstimatorConsistencyConfig()
    run_experiment(cfg)


if __name__ == "__main__":
    main()