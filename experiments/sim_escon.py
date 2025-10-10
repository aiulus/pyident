from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pathlib

import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..ensembles import draw_with_ctrb_rank
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


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


@dataclass
class EstimatorConsistencyConfig(ExperimentConfig):
    """Configuration for the partially identifiable consistency study."""

    dims_invisible: Sequence[int] = (1,)
    """How many invisible directions ``d`` to test (``dim V = n - d``)."""

    n_systems: int = 8
    """Number of distinct systems drawn for each ``d``."""

    n_x0_per_system: int = 6
    """Number of initial conditions sampled per system."""

    n_signal_realizations: int = 4
    """Independent PRBS inputs simulated per ``(system, x0)`` pair."""

    pe_order_max: int = 16
    """Maximum order used when estimating PE of the generated inputs."""

    prbs_dwell_scale: float = 8.0
    """Scale used to adapt the PRBS dwell time to the visible dimension."""

    max_system_draws: int = 128
    """Maximum attempts allowed when searching for a system with target dim ``V``."""

    max_x0_draws: int = 256
    """Maximum attempts allowed when sampling an initial state with target dim ``V``."""

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

def _orthonormal_basis(M: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Return an orthonormal basis for the column space of ``M``."""

    if M.size == 0:
        return np.zeros((M.shape[0], 0))

    U, s, _ = np.linalg.svd(M, full_matrices=False)
    if s.size == 0:
        return np.zeros((M.shape[0], 0))
    
    cutoff = tol * max(M.shape)
    rank = int(np.sum(s > cutoff))
    return U[:, :rank]

def _reachable_basis(A: np.ndarray, B: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Compute an orthonormal basis for the reachable subspace of ``(A, B)``."""
    """Return an orthonormal basis for the reachable subspace of ``(A, B)``."""

    n = A.shape[0]
    zero = np.zeros(n)
    # ``unrestricted`` spans the iterated images of both ``A`` and ``B``
    K = unified_generator(A, B, zero, mode="unrestricted")
    return _orthonormal_basis(K, tol=tol)


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


def _adapted_basis(Vbasis: np.ndarray, n: int, tol: float = 1e-12) -> Tuple[np.ndarray, int]:
    """Return an orthonormal basis whose first ``k`` columns span ``Vbasis``."""

    if Vbasis.size == 0:
        return np.eye(n), 0

    U, s, _ = np.linalg.svd(Vbasis, full_matrices=True)
    k = int(np.sum(s > tol))
    return U, k


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

    n = Ad.shape[0]
    Q, k = _adapted_basis(Vbasis, n)
    A_tilde = Q.T @ Ad @ Q
    Ahat_tilde = Q.T @ Ahat @ Q
    B_tilde = Q.T @ Bd
    Bhat_tilde = Q.T @ Bhat

    if k > 0:
        scale_vis_A = float(np.linalg.norm(A_tilde[:k, :k], ord="fro") + 1e-15)
        scale_vis_B = float(np.linalg.norm(B_tilde[:k, :], ord="fro") + 1e-15)
        errA_vis_block = float(np.linalg.norm(Ahat_tilde[:k, :k] - A_tilde[:k, :k], ord="fro") / scale_vis_A)
        errB_vis_block = float(np.linalg.norm(Bhat_tilde[:k, :] - B_tilde[:k, :], ord="fro") / scale_vis_B)
    else:
        errA_vis_block = float(np.linalg.norm(Ahat_tilde - A_tilde, ord="fro"))
        errB_vis_block = float(np.linalg.norm(Bhat_tilde - B_tilde, ord="fro"))

    if k < n:
        scale_dark_A = float(np.linalg.norm(A_tilde[k:, k:], ord="fro") + 1e-15)
        scale_dark_B = float(np.linalg.norm(B_tilde[k:, :], ord="fro") + 1e-15)
        errA_dark_block = float(np.linalg.norm(Ahat_tilde[k:, k:] - A_tilde[k:, k:], ord="fro") / scale_dark_A)
        errB_dark_block = float(np.linalg.norm(Bhat_tilde[k:, :] - B_tilde[k:, :], ord="fro") / scale_dark_B)
    else:
        errA_dark_block = 0.0
        errB_dark_block = 0.0

    return {
        "errA_rel": errA,
        "errB_rel": errB,
        "errA_V_rel": float(dA_V / scale_VA if scale_VA > 0 else dA_V),
        "errB_V_rel": float(dB_V / scale_VB if scale_VB > 0 else dB_V),
        "errA_vis_block_rel": errA_vis_block,
        "errB_vis_block_rel": errB_vis_block,
        "errA_dark_block_rel": errA_dark_block,
        "errB_dark_block_rel": errB_dark_block,
    }


def _estimate_algorithms() -> Mapping[str, Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    return {
        "dmdc_pinv": dmdc_pinv,
        "dmdc_tls": dmdc_tls,
        "moesp": moesp_fit,
    }


def _prepare_partially_identifiable_system(
    cfg: EstimatorConsistencyConfig,
    rng: np.random.Generator,
    dim_visible: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``(Ad, Bd)`` with ``dim V(x0)`` matching ``dim_visible`` for some ``x0``."""

    if not (0 < dim_visible <= cfg.n):
        raise ValueError(f"Visible dimension must lie in (0, n]; got {dim_visible} with n={cfg.n}.")

    base = getattr(cfg, "partial_base_ensemble", None) or cfg.ensemble
    if base == "A_stbl_B_ctrb":
        base = "stable"

    attempts = 0
    while attempts < cfg.max_system_draws:
        attempts += 1
        A, B, _ = draw_with_ctrb_rank(
            cfg.n,
            cfg.m,
            dim_visible,
            rng,
            ensemble_type=base,
            embed_random_basis=True,
        )
        Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)
        Rbasis = _reachable_basis(Ad, Bd, tol=1e-12)
        if Rbasis.shape[1] != dim_visible:
            continue
        return Ad, Bd, Rbasis
    
    raise RuntimeError(
        f"Unable to synthesise a system with reachable dimension {dim_visible} after {cfg.max_system_draws} attempts."
    )

def _sample_visible_initial_state(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Rbasis: np.ndarray,
    dim_visible: int,
    rng: np.random.Generator,
    max_attempts: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample ``x0`` such that ``dim V(x0)`` equals ``dim_visible``."""
    
    n = Ad.shape[0]
    for _ in range(max_attempts):
        if dim_visible >= n:
            x0 = _sample_unit_sphere(n, rng)
        else:
            coeff = rng.standard_normal(dim_visible)
            x0 = Rbasis @ coeff
            nrm = float(np.linalg.norm(x0))
            if nrm <= 1e-12:
                continue

            x0 = x0 / nrm
        Vbasis = _visible_basis(Ad, Bd, x0)

        if Vbasis.shape[1] == dim_visible:
            return x0, Vbasis
    raise RuntimeError(
        f"Failed to draw x0 with dim V(x0)={dim_visible} after {max_attempts} attempts."
    )


def _draw_prbs(cfg: EstimatorConsistencyConfig, rng: np.random.Generator, dim_visible: int) -> Tuple[np.ndarray, int]:
    dwell = max(1, int(round(cfg.prbs_dwell_scale / max(1, dim_visible))))
    U = prbs_dt(cfg.T, cfg.m, scale=cfg.u_scale, dwell=dwell, rng=rng)
    return U, dwell

def _partially_identifiable_trials(cfg: EstimatorConsistencyConfig, rng: np.random.Generator) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    algs = _estimate_algorithms()

    for hidden_dim in cfg.dims_invisible:
        dim_visible = cfg.n - int(hidden_dim)
        if dim_visible <= 0:
            raise ValueError(
                f"Invisible dimension {hidden_dim} is incompatible with n={cfg.n}."
            )
        for sys_idx in range(cfg.n_systems):
            Ad, Bd, Rbasis = _prepare_partially_identifiable_system(cfg, rng, dim_visible)

            for x_idx in range(cfg.n_x0_per_system):
                x0, Vbasis = _sample_visible_initial_state(
                    Ad,
                    Bd,
                    Rbasis,
                    dim_visible,
                    rng,
                    cfg.max_x0_draws,
                )
                ident = _identifiability_summary(Ad, Bd, x0)

                for signal_idx in range(cfg.n_signal_realizations):
                    U, dwell = _draw_prbs(cfg, rng, dim_visible)
                    pe_est = estimate_pe_order(
                        U,
                        s_max=min(cfg.pe_order_max, cfg.T // 2),
                    )

                    X = simulate_dt(x0, Ad, Bd, U, noise_std=cfg.noise_std, rng=rng)
                    X0, X1 = X[:, :-1], X[:, 1:]
                    U_cm = U.T
                    zstats = regressor_stats(X0, U_cm, rtol_rank=1e-12)

                    for name, estimator in algs.items():
                        try:
                            Ahat, Bhat = estimator(X0, X1, U_cm)
                        except Exception as exc:  # pragma: no cover - defensive guard
                            rec = {
                                "hidden_dim": int(hidden_dim),
                                "dim_visible": float(dim_visible),
                                "system": sys_idx,
                                "x0_id": x_idx,
                                "signal_id": signal_idx,
                                "algo": name,
                                "failed": 1,
                                "failure_msg": str(exc),
                                "dwell": dwell,
                                "pe_estimated": pe_est,
                                "target_pe": dim_visible,
                                **ident,
                            }
                            rec.update({
                                k: np.nan
                                for k in (
                                    "errA_rel",
                                    "errB_rel",
                                    "errA_V_rel",
                                    "errB_V_rel",
                                    "errA_vis_block_rel",
                                    "errB_vis_block_rel",
                                    "errA_dark_block_rel",
                                    "errB_dark_block_rel",
                                )
                            })
                            rec.update({k: np.nan for k in ("eqv_ok", "eqv_dA_V", "eqv_dB", "eqv_leak")})
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
                            rtol_eq=getattr(cfg, "rtol_eq_rel", 1e-2),
                            rtol_rank=1e-12,
                            use_leak=True,
                        )

                        rec = {
                            "hidden_dim": int(hidden_dim),
                            "dim_visible": float(dim_visible),
                            "system": sys_idx,
                            "x0_id": x_idx,
                            "signal_id": signal_idx,
                            "algo": name,
                            "failed": 0,
                            "eqv_ok": int(ok_eqv),
                            "dwell": dwell,
                            "pe_estimated": pe_est,
                            "target_pe": dim_visible,
                            **ident,
                            **errs,
                            **zstats,
                            "eqv_dim_V": info_eqv.get("dim_V", np.nan),
                            "eqv_dA_V": info_eqv.get("dA_V", np.nan),
                            "eqv_dB": info_eqv.get("dB", np.nan),
                            "eqv_leak": info_eqv.get("leak", np.nan),
                        }
                        records.append(rec)

    return pd.DataFrame.from_records(records)


def _summaries_from_trials(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    metrics = [
        "errA_rel",
        "errB_rel",
        "errA_V_rel",
        "errB_V_rel",
        "errA_vis_block_rel",
        "errB_vis_block_rel",
        "errA_dark_block_rel",
        "errB_dark_block_rel",
    ]
    grouped = (
        df.loc[df["failed"] == 0]
        .groupby(["hidden_dim", "dim_visible", "algo"])[metrics]
        .agg(["mean", "median"])
        .reset_index()
    )

    flat_columns: List[str] = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            name = "_".join(str(part) for part in col if str(part))
            flat_columns.append(name)
        else:
            flat_columns.append(str(col))
    grouped.columns = flat_columns
    return grouped


# ---------------------------------------------------------------------------
# Public experiment entry point
# ---------------------------------------------------------------------------


def run_experiment(cfg: Optional[EstimatorConsistencyConfig] = None) -> Dict[str, pd.DataFrame]:
    """Run the estimator consistency experiment on partially identifiable systems."""

    if cfg is None:
        cfg = EstimatorConsistencyConfig()

    rng = np.random.default_rng(cfg.seed)

    trials = _partially_identifiable_trials(cfg, rng)
    summary = _summaries_from_trials(trials)

    outputs = {
        "trials": trials,
        "summary": summary,
    }

    for name, df in outputs.items():
        df.to_csv(cfg.save_dir / f"{name}.csv", index=False)

    return outputs


def main() -> None:
    cfg = EstimatorConsistencyConfig()
    run_experiment(cfg)


if __name__ == "__main__":
    main()