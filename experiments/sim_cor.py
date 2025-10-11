from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..ensembles import sample_system_instance
from ..estimators import dmdc_pinv, dmdc_tls, moesp_fit, node_fit, sindy_fit
from ..metrics import (
    build_visible_basis_dt,
    cont2discrete_zoh,
    eta0,
    unified_generator,
)
from ..signals import estimate_pe_order
from ..simulation import prbs, simulate_dt
from .visible_sampling import (
    VisibleDrawConfig,
    construct_x0_with_dim_visible,
    prepare_system_with_visible_dim,
    sample_visible_initial_state,
)


Estimator = Callable[[np.ndarray, np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray]]


# ---------------------------------------------------------------------------
# Configuration & CLI
# ---------------------------------------------------------------------------


def _available_estimators() -> Dict[str, Estimator]:
    return {
        "dmdc_tls": lambda X0, X1, U_cm, dt: dmdc_tls(X0, X1, U_cm),
        "dmdc_pinv": lambda X0, X1, U_cm, dt: dmdc_pinv(X0, X1, U_cm),
        "moesp": lambda X0, X1, U_cm, dt: moesp_fit(X0, X1, U_cm),
        "sindy": lambda X0, X1, U_cm, dt: sindy_fit(X0, X1, U_cm, dt),
        "node": lambda X0, X1, U_cm, dt: node_fit(X0, X1, U_cm, dt, epochs=200),
    }


@dataclass
class IdentifiabilityCorrelationConfig(ExperimentConfig):
    """Configuration for the identifiability/error correlation experiment."""

    alg: str = "dmdc_tls"
    ens_size: int = 24
    x0_samples: int = 8
    engine: str = "stoch"  # "det" or "stoch"
    x0_amp: float = 1.0
    outdir: Path = Path("out_identifiability_corr")

    def __post_init__(self) -> None:  # type: ignore[override]
        super().__post_init__()
        if self.ens_size <= 0:
            raise ValueError("ens_size must be positive.")
        if self.x0_samples <= 0:
            raise ValueError("x0_samples must be positive.")
        if self.engine not in {"det", "stoch"}:
            raise ValueError("engine must be either 'det' or 'stoch'.")
        if self.x0_amp <= 0.0:
            raise ValueError("x0_amp must be positive.")
        if isinstance(self.outdir, str):
            self.outdir = Path(self.outdir)


def _parse_args(argv: Sequence[str] | None = None) -> IdentifiabilityCorrelationConfig:
    estimators = _available_estimators()

    parser = argparse.ArgumentParser(
        description=(
            "Correlate identifiability scores with reconstruction accuracy "
            "for deterministic and stochastic ensemble samplers."
        )
    )
    parser.add_argument("--n", type=int, default=6, help="State dimension.")
    parser.add_argument("--m", type=int, default=3, help="Input dimension.")
    parser.add_argument("--T", type=int, default=400, help="Trajectory horizon (minimum enforced automatically).")
    parser.add_argument("--dt", type=float, default=0.05, help="Sampling time for discretisation.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    parser.add_argument(
        "--ensemble",
        type=str,
        default="ginibre",
        choices=["ginibre", "sparse", "stable", "binary", "A_stbl_B_ctrb"],
        help="Base ensemble for (A, B).",
    )
    parser.add_argument("--p-density", type=float, default=0.8, dest="p_density", help="Density for sparse ensemble.")
    parser.add_argument(
        "--sparse-which",
        "--sparse_which",
        dest="sparse_which",
        type=str,
        default="both",
        choices=["A", "B", "both"],
        help="Which matrices are sparsified when ensemble=sparse.",
    )
    parser.add_argument(
        "--p-density-B",
        "--p_density_B",
        dest="p_density_B",
        type=float,
        default=None,
        help="Optional density for B when ensemble=sparse.",
    )

    parser.add_argument(
        "--alg",
        type=str,
        default="dmdc_tls",
        choices=sorted(estimators.keys()),
        help="Estimator used to recover (A, B).",
    )
    parser.add_argument("--ens-size", type=int, default=24, help="Base ensemble size (number of systems).")
    parser.add_argument("--x0-samples", type=int, default=8, help="Number of initial states per system.")
    parser.add_argument(
        "--x0amp",
        type=float,
        default=1.0,
        help="Scale factor applied to unit-sphere initial states.",
    )
    parser.add_argument("--noise-std", type=float, default=0.0, dest="noise_std", help="Process noise standard deviation.")
    parser.add_argument("--u-scale", type=float, default=3.0, dest="u_scale", help="PRBS amplitude.")

    engine_group = parser.add_mutually_exclusive_group()
    engine_group.add_argument(
        "--det",
        dest="engine",
        action="store_const",
        const="det",
        help="Use deterministic, stratified visible-dimension sampling.",
    )
    engine_group.add_argument(
        "--stoch",
        dest="engine",
        action="store_const",
        const="stoch",
        help="Use stochastic ensemble sampling (default).",
    )
    parser.set_defaults(engine="stoch")

    parser.add_argument(
        "--outdir",
        type=str,
        default="out_identifiability_corr",
        help="Directory for CSV outputs and plots.",
    )

    args = parser.parse_args(argv)

    cfg = IdentifiabilityCorrelationConfig(
        n=args.n,
        m=args.m,
        T=args.T,
        dt=args.dt,
        seed=args.seed,
        ensemble=args.ensemble,
        p_density=args.p_density,
        sparse_which=args.sparse_which,
        p_density_B=args.p_density_B,
        alg=args.alg,
        ens_size=args.ens_size,
        x0_samples=args.x0_samples,
        engine=args.engine,
        x0_amp=args.x0amp,
        noise_std=args.noise_std,
        u_scale=args.u_scale,
        outdir=Path(args.outdir),
    )
    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_estimator(name: str) -> Estimator:
    estimators = _available_estimators()
    if name not in estimators:
        raise ValueError(f"Unknown estimator '{name}'. Available: {sorted(estimators)}")
    return estimators[name]


def _sample_unit_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(n)
    nrm = float(np.linalg.norm(v))
    if nrm == 0.0:
        return _sample_unit_sphere(n, rng)
    return v / nrm


def _identifiability_metrics(Ad: np.ndarray, Bd: np.ndarray, x0: np.ndarray) -> Dict[str, float]:
    generator = unified_generator(Ad, Bd, x0, mode="unrestricted")
    if generator.size:
        svals = np.linalg.svd(generator, compute_uv=False)
        sigma_min = float(svals[-1])
    else:
        sigma_min = 0.0
    eta = float(eta0(Ad, Bd, x0, rtol=1e-12))
    Vbasis = build_visible_basis_dt(Ad, Bd, x0, tol=1e-10)
    dim_visible = int(Vbasis.shape[1])
    return {
        "sigma_min": sigma_min,
        "eta0": eta,
        "dim_visible": dim_visible,
    }


def _relative_errors(Ahat: np.ndarray, Bhat: np.ndarray, Ad: np.ndarray, Bd: np.ndarray) -> Dict[str, float]:
    errA = float(np.linalg.norm(Ahat - Ad, ord="fro"))
    errB = float(np.linalg.norm(Bhat - Bd, ord="fro"))
    nrmA = float(np.linalg.norm(Ad, ord="fro") + 1e-15)
    nrmB = float(np.linalg.norm(Bd, ord="fro") + 1e-15)
    relA = errA / nrmA
    relB = errB / nrmB
    rel_mean = 0.5 * (relA + relB)
    return {
        "errA_rel": relA,
        "errB_rel": relB,
        "err_mean": rel_mean,
    }


def _target_order(cfg: IdentifiabilityCorrelationConfig) -> int:
    return 2 * (cfg.n + cfg.m)


def _minimum_horizon(cfg: IdentifiabilityCorrelationConfig, order: int) -> int:
    # Block Hankel rank condition: T >= m*r + r - 1
    T_needed = cfg.m * order + order - 1
    return max(cfg.T, T_needed)


def _prbs_with_order(cfg: IdentifiabilityCorrelationConfig, rng: np.random.Generator) -> Tuple[np.ndarray, int]:
    target = _target_order(cfg)
    T = _minimum_horizon(cfg, target)
    best_U: np.ndarray | None = None
    best_order = -1
    for _ in range(25):
        U = prbs(T, cfg.m, scale=cfg.u_scale, dwell=1, rng=rng)
        order_est = int(estimate_pe_order(U, s_max=min(target, T // 2)))
        if order_est >= target:
            return U, order_est
        if order_est > best_order:
            best_order = order_est
            best_U = U
    if best_U is None:
        best_U = prbs(T, cfg.m, scale=cfg.u_scale, dwell=1, rng=rng)
        best_order = int(estimate_pe_order(best_U, s_max=min(target, T // 2)))
    return best_U, best_order


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


def _run_stochastic_engine(
    cfg: IdentifiabilityCorrelationConfig,
    estimator: Estimator,
    rng: np.random.Generator,
    U: np.ndarray,
    pe_order: int,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    U_cm = U.T

    for sys_idx in range(cfg.ens_size):
        A, B = sample_system_instance(cfg, rng)
        Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)
        for x_idx in range(cfg.x0_samples):
            x0 = _sample_unit_sphere(cfg.n, rng) * cfg.x0_amp
            ident = _identifiability_metrics(Ad, Bd, x0)

            X = simulate_dt(x0, Ad, Bd, U, noise_std=cfg.noise_std, rng=rng)
            X0, X1 = X[:, :-1], X[:, 1:]

            try:
                Ahat, Bhat = estimator(X0, X1, U_cm, cfg.dt)
            except Exception:
                continue

            errs = _relative_errors(Ahat, Bhat, Ad, Bd)

            results.append(
                {
                    "engine": "stoch",
                    "system_index": sys_idx,
                    "x0_index": x_idx,
                    "deficiency": int(cfg.n - ident["dim_visible"]),
                    "dim_visible": ident["dim_visible"],
                    "sigma_min": ident["sigma_min"],
                    "eta0": ident["eta0"],
                    "errA_rel": errs["errA_rel"],
                    "errB_rel": errs["errB_rel"],
                    "err_mean": errs["err_mean"],
                    "pe_order": pe_order,
                }
            )
    return results


def _deterministic_dim_sequence(n: int) -> Iterable[int]:
    for deficiency in range(0, 4):
        dim_visible = n - deficiency
        if dim_visible <= 0:
            continue
        yield dim_visible


def _run_deterministic_engine(
    cfg: IdentifiabilityCorrelationConfig,
    estimator: Estimator,
    rng: np.random.Generator,
    U: np.ndarray,
    pe_order: int,
) -> List[Dict[str, object]]:
    per_def_records: Dict[int, List[Dict[str, object]]] = {}
    U_cm = U.T

    base_ensemble = "stable" if cfg.ensemble == "A_stbl_B_ctrb" else cfg.ensemble

    for dim_visible in _deterministic_dim_sequence(cfg.n):
        deficiency = cfg.n - dim_visible
        per_def_records.setdefault(deficiency, [])
        systems_target = cfg.ens_size
        # Oversample per deficiency (×4 overall) to enable stratified down-selection
        tuples_needed = cfg.ens_size * cfg.x0_samples
        systems_built = 0
        tuples_collected = 0

        while tuples_collected < tuples_needed:
            if systems_built >= systems_target and tuples_collected < tuples_needed:
                systems_target += 1  # allow extra systems if earlier ones failed

            draw_cfg = VisibleDrawConfig(
                n=cfg.n,
                m=cfg.m,
                dt=cfg.dt,
                dim_visible=dim_visible,
                ensemble=base_ensemble,
                max_system_attempts=512,
                max_x0_attempts=1024,
                tol=1e-12,
            )

            try:
                A, B, Ad, Bd, Rbasis = prepare_system_with_visible_dim(draw_cfg, rng)
            except RuntimeError:
                continue

            systems_built += 1

            x0_records: List[np.ndarray] = []
            try:
                x0_det, _ = construct_x0_with_dim_visible(
                    Ad,
                    Bd,
                    Rbasis,
                    dim_visible,
                    rng,
                    tol=1e-12,
                )
                x0_records.append(x0_det)
            except Exception:
                pass

            attempts = 0
            while len(x0_records) < cfg.x0_samples and attempts < 4 * cfg.x0_samples:
                attempts += 1
                try:
                    x0_s, _ = sample_visible_initial_state(
                        Ad,
                        Bd,
                        Rbasis,
                        dim_visible,
                        rng,
                        max_attempts=256,
                        tol=1e-12,
                    )
                except Exception:
                    continue
                x0_records.append(x0_s)

            for x0_raw in x0_records:
                x0 = x0_raw * cfg.x0_amp
                ident = _identifiability_metrics(Ad, Bd, x0)

                X = simulate_dt(x0, Ad, Bd, U, noise_std=cfg.noise_std, rng=rng)
                X0, X1 = X[:, :-1], X[:, 1:]

                try:
                    Ahat, Bhat = estimator(X0, X1, U_cm, cfg.dt)
                except Exception:
                    continue

                errs = _relative_errors(Ahat, Bhat, Ad, Bd)

                per_def_records[deficiency].append(
                    {
                        "engine": "det",
                        "system_index": systems_built - 1,
                        "x0_index": len(per_def_records[deficiency]),
                        "deficiency": deficiency,
                        "dim_visible": ident["dim_visible"],
                        "sigma_min": ident["sigma_min"],
                        "eta0": ident["eta0"],
                        "errA_rel": errs["errA_rel"],
                        "errB_rel": errs["errB_rel"],
                        "err_mean": errs["err_mean"],
                        "pe_order": pe_order,
                    }
                )
                tuples_collected += 1
                if tuples_collected >= tuples_needed:
                    break

    total_target = cfg.ens_size * cfg.x0_samples
    return _stratified_subset(per_def_records, total_target)


def _stratified_subset(
    per_def_records: Dict[int, List[Dict[str, object]]], total_target: int
) -> List[Dict[str, object]]:
    deficiencies = sorted(per_def_records.keys())
    if not deficiencies:
        raise RuntimeError("Deterministic engine did not record any tuples.")

    bins = len(deficiencies)
    if total_target < bins:
        raise ValueError(
            "Requested total tuples fewer than number of deficiency bins; increase ens-size or x0-samples."
        )

    base = total_target // bins
    remainder = total_target % bins

    selected: List[Dict[str, object]] = []
    for idx, deficiency in enumerate(deficiencies):
        need = base + (1 if idx < remainder else 0)
        available = per_def_records[deficiency]
        if len(available) < need:
            raise RuntimeError(
                f"Insufficient tuples for deficiency {deficiency}; have {len(available)} need {need}."
            )
        selected.extend(available[:need])

    for idx, rec in enumerate(selected):
        rec["x0_index"] = idx

    return selected


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def _compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for deficiency, sub in df.groupby("deficiency"):
        if len(sub) < 2:
            pearson = np.nan
            spearman = np.nan
        else:
            x = sub["sigma_min"].to_numpy()
            y = sub["err_mean"].to_numpy()
            if np.allclose(x.std(ddof=0), 0.0) or np.allclose(y.std(ddof=0), 0.0):
                pearson = np.nan
            else:
                pearson = float(np.corrcoef(x, y)[0, 1])

            rx = pd.Series(x).rank(method="average").to_numpy()
            ry = pd.Series(y).rank(method="average").to_numpy()
            if np.allclose(rx.std(ddof=0), 0.0) or np.allclose(ry.std(ddof=0), 0.0):
                spearman = np.nan
            else:
                spearman = float(np.corrcoef(rx, ry)[0, 1])

        rows.append(
            {
                "deficiency": int(deficiency),
                "pearson": pearson,
                "spearman": spearman,
                "count": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def _scatter_plot(df: pd.DataFrame, cfg: IdentifiabilityCorrelationConfig, outfile: Path) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    deficiencies = sorted(df["deficiency"].unique())
    n_plots = max(len(deficiencies), 1)
    nrows = 2
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax in axes[n_plots:]:
        ax.axis("off")

    for ax, deficiency in zip(axes, deficiencies):
        sub = df[df["deficiency"] == deficiency]
        ax.scatter(sub["sigma_min"], sub["err_mean"], alpha=0.65, s=18)
        ax.set_title(f"deficiency = {deficiency}")
        ax.set_xlabel("identifiability score σ_min")
        ax.set_ylabel("mean relative error")
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        f"Identifiability vs error (engine={cfg.engine}, alg={cfg.alg}, ensemble={cfg.ensemble})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    cfg = _parse_args(argv)
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(cfg.seed))
    estimator = _select_estimator(cfg.alg)

    U, order_est = _prbs_with_order(cfg, rng)

    if order_est < _target_order(cfg):
        print(
            f"[warn] PRBS achieved order {order_est} < target {_target_order(cfg)}. Results may be conservative."
        )

    if cfg.engine == "stoch":
        records = _run_stochastic_engine(cfg, estimator, rng, U, order_est)
    else:
        records = _run_deterministic_engine(cfg, estimator, rng, U, order_est)

    if not records:
        raise RuntimeError("No successful trials were recorded. Check configuration or estimator stability.")

    df = pd.DataFrame.from_records(records)
    df["ident_score"] = df["sigma_min"]

    corr_df = _compute_correlations(df)

    scatter_path = cfg.outdir / f"scatter_{cfg.engine}.png"
    _scatter_plot(df, cfg, scatter_path)

    csv_path = cfg.outdir / f"results_{cfg.engine}.csv"
    df.to_csv(csv_path, index=False)

    corr_path = cfg.outdir / f"correlations_{cfg.engine}.csv"
    corr_df.to_csv(corr_path, index=False)

    meta = {
        "config": {
            "n": cfg.n,
            "m": cfg.m,
            "dt": cfg.dt,
            "T": cfg.T,
            "ensemble": cfg.ensemble,
            "alg": cfg.alg,
            "ens_size": cfg.ens_size,
            "x0_samples": cfg.x0_samples,
            "engine": cfg.engine,
            "x0_amp": cfg.x0_amp,
            "noise_std": cfg.noise_std,
            "u_scale": cfg.u_scale,
            "seed": cfg.seed,
        },
        "pe_order_est": order_est,
        "records": int(len(df)),
    }
    with open(cfg.outdir / f"summary_{cfg.engine}.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Saved scatter plot to {scatter_path}")
    print(f"Saved results to {csv_path}")
    print(f"Saved correlations to {corr_path}")


if __name__ == "__main__":
    main()