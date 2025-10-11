"""Stratified identifiability/error correlation experiment.

This script reproduces the deterministic engine of ``sim_cor`` where
systems are stratified by the visible subspace dimension of the
initial state.  Each retained trial is additionally annotated with the
continuous-time metrics used by ``sim1`` so that we can compare the two
families of proxies side-by-side.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..estimators import dmdc_pinv, dmdc_tls, moesp_fit, node_fit, sindy_fit
from ..metrics import pair_distance
from ..simulation import simulate_dt
from .simcor_common import (
    CoreExperimentConfig,
    compute_core_metrics,
    compute_identifiability_metrics,
    prbs_with_order,
    relative_errors,
    target_pe_order,
)
from .visible_sampling import (
    VisibleDrawConfig,
    construct_x0_with_dim_visible,
    prepare_system_with_visible_dim,
    sample_visible_initial_state,
)


EstimatorMap = Dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, float], tuple[np.ndarray, np.ndarray]]]


# ---------------------------------------------------------------------------
# Configuration & CLI
# ---------------------------------------------------------------------------


@dataclass
class StratifiedCoreConfig(CoreExperimentConfig):
    """Configuration for the stratified deterministic experiment."""

    estimators: tuple[str, ...] = ("dmdc_pinv",)
    outdir: Path = Path("out_stratcore")
    dark_dims: tuple[int, ...] = ()

    def __post_init__(self) -> None:  # type: ignore[override]
        super().__post_init__()
        if not self.estimators:
            raise ValueError("At least one estimator must be specified.")
        if isinstance(self.outdir, str):
            self.outdir = Path(self.outdir)
        if not self.dark_dims:
            self.dark_dims = tuple(range(1, self.n))
        cleaned: List[int] = []
        for deficiency in self.dark_dims:
            if deficiency < 0:
                raise ValueError("dark_dims entries must be non-negative.")
            if deficiency >= self.n:
                raise ValueError("dark_dims entries must be less than n.")
            if deficiency not in cleaned:
                cleaned.append(deficiency)
        self.dark_dims = tuple(cleaned)


def _available_estimators() -> EstimatorMap:
    return {
        "dmdc_tls": lambda X0, X1, U_cm, dt: dmdc_tls(X0, X1, U_cm),
        "dmdc_pinv": lambda X0, X1, U_cm, dt: dmdc_pinv(X0, X1, U_cm),
        "moesp": lambda X0, X1, U_cm, dt: moesp_fit(X0, X1, U_cm),
        "sindy": lambda X0, X1, U_cm, dt: sindy_fit(X0, X1, U_cm, dt),
        "node": lambda X0, X1, U_cm, dt: node_fit(X0, X1, U_cm, dt, epochs=200),
    }


def _parse_args(argv: Sequence[str] | None = None) -> StratifiedCoreConfig:
    estimators = _available_estimators()

    parser = argparse.ArgumentParser(
        description="Deterministic, stratified identifiability/error correlation experiment."
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

    parser.add_argument("--ens-size", type=int, default=24, dest="ens_size", help="Target systems per deficiency bin.")
    parser.add_argument("--x0-samples", type=int, default=8, dest="x0_samples", help="Initial states per system.")
    parser.add_argument(
        "--x0amp",
        type=float,
        default=1.0,
        help="Uniform scaling factor applied to the unit-sphere initial states.",
    )
    parser.add_argument("--noise-std", type=float, default=0.0, dest="noise_std", help="Process noise standard deviation.")
    parser.add_argument("--u-scale", type=float, default=3.0, dest="u_scale", help="PRBS amplitude.")
    parser.add_argument("--dwell", type=int, default=1, help="PRBS dwell time (samples per draw).")
    parser.add_argument(
        "--dark-dims",
        type=int,
        nargs="+",
        default=None,
        dest="dark_dims",
        help=(
            "List of dark-dimension counts n-dim(V(x0)) to target during stratified sampling. "
            "Defaults to the integers from 1 to n-1."
        ),
    )
    parser.add_argument(
        "--estimators",
        type=str,
        nargs="+",
        default=["dmdc_pinv"],
        choices=sorted(estimators.keys()),
        help="Estimators evaluated for each retained trial.",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="out_stratcore",
        help="Directory for CSV outputs and plots.",
    )

    args = parser.parse_args(argv)

    cfg = StratifiedCoreConfig(
        n=args.n,
        m=args.m,
        T=args.T,
        dt=args.dt,
        seed=args.seed,
        ensemble=args.ensemble,
        p_density=args.p_density,
        sparse_which=args.sparse_which,
        p_density_B=args.p_density_B,
        ens_size=args.ens_size,
        x0_samples=args.x0_samples,
        x0_amp=args.x0amp,
        noise_std=args.noise_std,
        u_scale=args.u_scale,
        dwell=args.dwell,
        estimators=tuple(args.estimators),
        dark_dims=tuple(args.dark_dims) if args.dark_dims is not None else (),
        outdir=Path(args.outdir),
    )
    return cfg


# ---------------------------------------------------------------------------
# Deterministic sampling helpers
# ---------------------------------------------------------------------------


def _select_estimators(names: Iterable[str]) -> EstimatorMap:
    available = _available_estimators()
    selected: EstimatorMap = {}
    for name in names:
        if name not in available:
            raise ValueError(f"Unknown estimator '{name}'. Available: {sorted(available)}")
        selected[name] = available[name]
    return selected


def _deterministic_dim_sequence(cfg: StratifiedCoreConfig) -> Iterable[int]:
    for deficiency in cfg.dark_dims:
        dim_visible = cfg.n - deficiency
        if dim_visible <= 0:
            continue
        yield dim_visible


def _stratified_subset(
    per_def_records: Dict[int, List[Dict[str, object]]],
    total_target: int,
    deficiencies: Sequence[int],
) -> List[Dict[str, object]]:
    deficiency_order = [d for d in deficiencies if d in per_def_records]
    if not deficiency_order:
        raise RuntimeError("Deterministic engine did not record any tuples.")

    bins = len(deficiency_order)
    if total_target < bins:
        raise ValueError(
            "Requested total tuples fewer than number of deficiency bins; increase ens-size or x0-samples."
        )

    base = total_target // bins
    remainder = total_target % bins

    for idx, deficiency in enumerate(deficiency_order):
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


def _expand_records(base_records: List[Dict[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for rec in base_records:
        errors = rec.pop("errors")
        for est_name, err_vals in errors.items():
            row = {**rec, **err_vals, "estimator": est_name}
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting & correlation helpers
# ---------------------------------------------------------------------------


def _scatter_plot(df: pd.DataFrame, cfg: StratifiedCoreConfig, outfile: Path) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    estimators = sorted(df["estimator"].unique())
    n_plots = len(estimators)
    ncols = min(3, max(1, n_plots))
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for ax in axes[n_plots:]:
        ax.axis("off")

    for ax, name in zip(axes, estimators):
        sub = df[df["estimator"] == name]
        colors = sub["deficiency"].to_numpy()
        sc = ax.scatter(sub["sigma_min"], sub["err_mean_rel"], c=colors, cmap="viridis", alpha=0.7, s=20)
        ax.set_title(f"{name}")
        ax.set_xlabel(r"identifiability σ_min")
        ax.set_ylabel("mean relative error")
        ax.grid(True, linestyle="--", alpha=0.4)
    if n_plots:
        fig.colorbar(sc, ax=axes[:n_plots], shrink=0.85, label="deficiency")

    fig.suptitle(
        f"Identifiability vs error (stratified, ensemble={cfg.ensemble})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def _compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (estimator, deficiency), sub in df.groupby(["estimator", "deficiency"]):
        if len(sub) < 2:
            pearson = np.nan
            spearman = np.nan
        else:
            x = sub["sigma_min"].to_numpy()
            y = sub["err_mean_rel"].to_numpy()
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
                "estimator": estimator,
                "deficiency": int(deficiency),
                "pearson": pearson,
                "spearman": spearman,
                "count": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main deterministic engine
# ---------------------------------------------------------------------------


def _run_stratified(cfg: StratifiedCoreConfig, rng: np.random.Generator) -> pd.DataFrame:
    estimators = _select_estimators(cfg.estimators)

    U, order_est = prbs_with_order(cfg, rng)
    if order_est < target_pe_order(cfg):
        print(
            f"[warn] PRBS achieved order {order_est} < target {target_pe_order(cfg)}. Results may be conservative."
        )

    per_def_records: Dict[int, List[Dict[str, object]]] = {}
    U_cm = U.T

    base_ensemble = "stable" if cfg.ensemble == "A_stbl_B_ctrb" else cfg.ensemble

    for dim_visible in _deterministic_dim_sequence(cfg):
        deficiency = cfg.n - dim_visible
        per_def_records.setdefault(deficiency, [])
        systems_target = cfg.ens_size
        tuples_needed = cfg.ens_size * cfg.x0_samples
        systems_built = 0
        tuples_collected = 0

        while tuples_collected < tuples_needed:
            if systems_built >= systems_target and tuples_collected < tuples_needed:
                systems_target += 1

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
                ident = compute_identifiability_metrics(Ad, Bd, x0)
                core = compute_core_metrics(A, B, x0)

                X = simulate_dt(x0, Ad, Bd, U, noise_std=cfg.noise_std, rng=rng)
                X0, X1 = X[:, :-1], X[:, 1:]

                err_map: Dict[str, Dict[str, float]] = {}
                for name, estimator in estimators.items():
                    try:
                        Ahat, Bhat = estimator(X0, X1, U_cm, cfg.dt)
                    except Exception:
                        continue

                    rel = relative_errors(Ahat, Bhat, Ad, Bd)
                    err_pair = float(pair_distance(Ahat, Bhat, Ad, Bd))
                    err_map[name] = {
                        "errA_rel": rel["errA_rel"],
                        "errB_rel": rel["errB_rel"],
                        "err_mean_rel": rel["err_mean_rel"],
                        "err_pair": err_pair,
                    }

                if not err_map:
                    continue

                per_def_records[deficiency].append(
                    {
                        "engine": "det",
                        "system_index": systems_built - 1,
                        "deficiency": deficiency,
                        "dim_visible": ident["dim_visible"],
                        "sigma_min": ident["sigma_min"],
                        "eta0": ident["eta0"],
                        "pbh_struct": core["pbh_struct"],
                        "krylov_smin": core["krylov_smin"],
                        "mu_min": core["mu_min"],
                        "pe_order": order_est,
                        "errors": err_map,
                    }
                )

                tuples_collected += 1
                if tuples_collected >= tuples_needed:
                    break

    total_target = cfg.ens_size * cfg.x0_samples
    base_records = _stratified_subset(per_def_records, total_target, cfg.dark_dims)
    df = _expand_records(base_records)
    return df


def main(argv: Sequence[str] | None = None) -> None:
    cfg = _parse_args(argv)
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(cfg.seed))
    df = _run_stratified(cfg, rng)

    scatter_path = cfg.outdir / "scatter_det.png"
    _scatter_plot(df, cfg, scatter_path)

    corr_df = _compute_correlations(df)

    csv_path = cfg.outdir / "results_det.csv"
    df.to_csv(csv_path, index=False)

    corr_path = cfg.outdir / "correlations_det.csv"
    corr_df.to_csv(corr_path, index=False)

    meta = {
        "config": {
            "n": cfg.n,
            "m": cfg.m,
            "dt": cfg.dt,
            "T": cfg.T,
            "ensemble": cfg.ensemble,
            "estimators": list(cfg.estimators),
            "ens_size": cfg.ens_size,
            "x0_samples": cfg.x0_samples,
            "x0_amp": cfg.x0_amp,
            "noise_std": cfg.noise_std,
            "u_scale": cfg.u_scale,
            "dwell": cfg.dwell,
            "seed": cfg.seed,
            "dark_dims": list(cfg.dark_dims),
        },
        "records": int(len(df)),
    }
    with open(cfg.outdir / "summary_det.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Saved scatter plot to {scatter_path}")
    print(f"Saved results to {csv_path}")
    print(f"Saved correlations to {corr_path}")


if __name__ == "__main__":
    main()