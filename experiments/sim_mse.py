"""Stratified ensemble simulation of identifiability criteria vs. estimation error.

This script extends :mod:`experiments.sim_scoree` to sweep multiple controllability
deficiencies and systems.  Instead of drawing a single (A, B) pair it draws
``--ensvol`` systems and stratifies them over the controllability-rank deficiency
range ``0, 1, …, n-1``.  For each sampled system we evaluate ``--x0count`` initial
states drawn from the unit sphere and reuse the core criteria/error computation.

Key differences vs. ``experiments.sim_scoree``:

* ``--ensvol`` controls the number of (A, B) draws.  The script allocates
  ``floor(ensvol / n)`` systems to each deficiency level ``0, …, n-1``.
* ``--x0count`` controls how many initial states are sampled per system.
* Scatter plots colour-code points by deficiency and omit Pearson statistics from
  the legend.  Correlations are written to ``.csv`` and ``.json`` files instead.

Usage mirrors the original script while adding the new volume/stratification
parameters.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

from .sim_scoree import (
    add_transforms,
    compute_core_metrics,
    cont2discrete_zoh,
    draw_system,
    make_estimator,
    prbs,
    relative_error_fro,
    sample_unit_sphere,
    simulate_dt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: float | int | None) -> float | None:
    """Convert ``value`` to ``float`` if finite; otherwise return ``None``."""

    if value is None:
        return None
    if isinstance(value, (float, int)):
        val = float(value)
        if math.isfinite(val):
            return val
    return None


def _build_deficiency_colors(deficiencies: Sequence[int]) -> Dict[int, Tuple[float, ...]]:
    """Map each deficiency to a unique colour tuple."""

    unique = sorted(set(int(d) for d in deficiencies))
    if not unique:
        return {}
    cmap = plt.get_cmap("viridis", len(unique))
    return {d: cmap(idx) for idx, d in enumerate(unique)}


def _deficiency_legend_handles(color_map: Dict[int, Tuple[float, ...]]) -> List[Line2D]:
    """Create legend handles for deficiency colour mapping."""

    handles: List[Line2D] = []
    for deficiency, colour in sorted(color_map.items()):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=6,
                markerfacecolor=colour,
                markeredgecolor="black",
                label=f"deficiency={deficiency}",
            )
        )
    return handles


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def run_ensemble(
    *,
    n: int,
    m: int,
    T: int,
    dt: float,
    ensvol: int,
    x0count: int,
    noise_std: float,
    seed: int,
    ensemble: str,
    estimators: Sequence[str],
):
    """Simulate estimation error/criteria over a stratified ensemble."""

    rng = np.random.default_rng(seed)

    est_funcs: Dict[str, callable] = {}
    for name in estimators:
        try:
            est_funcs[name] = make_estimator(name)
        except Exception as exc:  # pragma: no cover - diagnostic message
            print(f"[warn] Skipping estimator '{name}': {exc}")

    if not est_funcs:
        raise RuntimeError("No valid estimators selected.")

    per_def = ensvol // n
    if per_def <= 0:
        raise ValueError("ensvol must be at least n so that each deficiency has a system.")

    rows: List[dict] = []
    A_list: List[np.ndarray] = []
    B_list: List[np.ndarray] = []
    Ad_list: List[np.ndarray] = []
    Bd_list: List[np.ndarray] = []
    deficiencies: List[int] = []

    trial = 0
    system_index = 0

    for deficiency in range(n):
        for _ in range(per_def):
            A, B, meta = draw_system(
                n=n,
                m=m,
                deficiency=deficiency,
                rng=rng,
                ensemble=ensemble,
            )
            Ad, Bd = cont2discrete_zoh(A, B, dt)

            A_list.append(A)
            B_list.append(B)
            Ad_list.append(Ad)
            Bd_list.append(Bd)
            deficiencies.append(deficiency)

            for x0_idx in range(x0count):
                x0 = sample_unit_sphere(n, rng)
                u = prbs(T, m, rng)
                X = simulate_dt(Ad, Bd, u, x0)
                if noise_std > 0.0:
                    X = X + noise_std * rng.standard_normal(X.shape)

                X0, X1, U = X[:, :-1], X[:, 1:], u.T

                crit = compute_core_metrics(A, B, x0)

                errs: Dict[str, float] = {}
                for est_name, est_fn in est_funcs.items():
                    try:
                        Ahat, Bhat = est_fn(X0, X1, U, n=n, dt=dt)
                        errs[f"err_{est_name}"] = relative_error_fro(Ahat, Bhat, Ad, Bd)
                    except Exception as exc:  # pragma: no cover - diagnostic message
                        print(
                            f"[warn] Estimator '{est_name}' failed on system {system_index}, "
                            f"x0 index {x0_idx}: {exc}"
                        )
                        errs[f"err_{est_name}"] = np.nan

                rows.append(
                    dict(
                        trial=trial,
                        deficiency=deficiency,
                        system_index=system_index,
                        x0_index=x0_idx,
                        **crit,
                        **errs,
                    )
                )
                trial += 1

            system_index += 1

    df = pd.DataFrame(rows)

    meta_out = {
        "A": np.stack(A_list),
        "B": np.stack(B_list),
        "Ad": np.stack(Ad_list),
        "Bd": np.stack(Bd_list),
        "deficiency": np.asarray(deficiencies, dtype=int),
        "ensemble": ensemble,
        "estimators": list(est_funcs.keys()),
        "n": n,
        "m": m,
        "dt": dt,
        "ensvol": ensvol,
        "x0count": x0count,
    }

    return df, meta_out


# ---------------------------------------------------------------------------
# Reporting utilities
# ---------------------------------------------------------------------------


def compute_correlations(df: pd.DataFrame, estimator_cols: Iterable[str]):
    """Compute Pearson statistics for each metric/estimator pair."""

    metrics = [
        ("x_inv_pbh", "1 / PBH (structured)"),
        ("x_inv_krylov_smin", "1 / σ_min(K)"),
        ("x_inv_mu", "1 / mu_min"),
    ]

    records: List[dict] = []
    json_records: List[dict] = []

    for metric_key, metric_label in metrics:
        record = {"metric": metric_key, "metric_label": metric_label}
        json_entry = {"metric": metric_key, "metric_label": metric_label, "estimators": {}}
        for est_col in estimator_cols:
            xvals = df[metric_key].to_numpy()
            yvals = df[est_col].to_numpy()
            mask = np.isfinite(xvals) & np.isfinite(yvals)
            n_eff = int(mask.sum())
            if n_eff >= 3:
                r, p = pearsonr(xvals[mask], yvals[mask])
                r_f = float(r)
                p_f = float(p)
            else:
                r_f = np.nan
                p_f = np.nan

            record[f"{est_col}_r"] = r_f
            record[f"{est_col}_p"] = p_f
            record[f"{est_col}_n"] = n_eff
            json_entry["estimators"][est_col] = {
                "r": _safe_float(r_f),
                "p": _safe_float(p_f),
                "n": n_eff,
            }

        records.append(record)
        json_records.append(json_entry)

    return pd.DataFrame(records), json_records


def scatter_plots(
    df: pd.DataFrame,
    ykey: str,
    outdir: pathlib.Path,
    tag: str,
    color_map: Dict[int, Tuple[float, ...]],
):
    """Scatter plots coloured by deficiency without correlation legends."""

    pairs = [
        ("x_inv_pbh", "1 / PBH (structured)"),
        ("x_inv_krylov_smin", "1 / σ_min(K)"),
        ("x_inv_mu", "1 / mu_min"),
    ]

    handles = _deficiency_legend_handles(color_map)

    for xkey, xlabel in pairs:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        colours = df["deficiency"].apply(lambda d: color_map.get(int(d)))
        ax.scatter(df[xkey].to_numpy(), df[ykey].to_numpy(), s=18, c=list(colours))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"REE ({ykey.replace('err_', '')})")
        ax.set_title("Estimation error vs. identifiability criterion")
        if handles:
            ax.legend(handles=handles, frameon=True, loc="best")
        fig.savefig(outdir / f"{xkey}_vs_{ykey}_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _compute_zoom_limits(xvals: np.ndarray, yvals: np.ndarray, q: float = 0.9):
    """Replicate the zoom-window heuristic from the original script."""

    mask = np.isfinite(xvals) & np.isfinite(yvals)
    if mask.sum() < 3:
        return None
    x = xvals[mask]
    y = yvals[mask]
    xlo = float(np.nanmin(x))
    ylo = float(np.nanmin(y))
    xhi = float(np.nanquantile(x, q))
    yhi = float(np.nanquantile(y, q))
    if not np.isfinite(xhi) or xhi <= xlo:
        xhi = xlo + 1e-12
    if not np.isfinite(yhi) or yhi <= ylo:
        yhi = ylo + 1e-12
    xpad = 0.02 * (xhi - xlo + 1e-12)
    ypad = 0.02 * (yhi - ylo + 1e-12)
    return xlo - xpad, xhi + xpad, ylo - ypad, yhi + ypad


def scatter_plots_zoom(
    df: pd.DataFrame,
    ykey: str,
    outdir: pathlib.Path,
    tag: str,
    color_map: Dict[int, Tuple[float, ...]],
    q_zoom: float,
):
    """Zoomed scatter plots with deficiency colour coding."""

    pairs = [
        ("x_inv_pbh", "1 / PBH (structured)"),
        ("x_inv_krylov_smin", "1 / σ_min(K)"),
        ("x_inv_mu", "1 / mu_min"),
    ]

    handles = _deficiency_legend_handles(color_map)

    for xkey, xlabel in pairs:
        xvals = df[xkey].to_numpy()
        yvals = df[ykey].to_numpy()
        limits = _compute_zoom_limits(xvals, yvals, q=q_zoom)
        if limits is None:
            continue
        xlo, xhi, ylo, yhi = limits
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        colours = df["deficiency"].apply(lambda d: color_map.get(int(d)))
        ax.scatter(xvals, yvals, s=18, c=list(colours))
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"REE ({ykey.replace('err_', '')})")
        ax.set_title(
            f"Estimation error vs. identifiability criterion (zoom {int(100 * q_zoom)}th)"
        )
        if handles:
            ax.legend(handles=handles, frameon=True, loc="best")
        fig.savefig(outdir / f"{xkey}_vs_{ykey}_{tag}_zoom.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_estimators(spec: str) -> List[str]:
    return [token.strip().lower() for token in spec.split(",") if token.strip()]


def main(argv: Sequence[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--ensvol", type=int, default=120)
    ap.add_argument("--x0count", type=int, default=1000)
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=31415)
    ap.add_argument("--outdir", type=str, default="ident_vs_error_strat")
    ap.add_argument("--tag", type=str, default="demo")
    ap.add_argument(
        "--ensemble",
        type=str,
        default="ginibre",
        choices=["ginibre", "stable", "sparse", "binary"],
        help="Base ensemble passed to draw_with_ctrb_rank.",
    )
    ap.add_argument(
        "--estimators",
        type=str,
        default="dmdc,moesp",
        help="Comma-separated subset of {dmdc, moesp, sindy, node}.",
    )
    ap.add_argument("--zoom", action="store_true", help="Generate zoomed scatter plots as well.")
    ap.add_argument(
        "--zoom-q",
        type=float,
        default=0.9,
        help="Quantile defining the zoom rectangle per axis (default 0.9).",
    )

    args = ap.parse_args(argv)

    outdir = pathlib.Path(args.outdir)
    plotdir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)

    estimator_list = parse_estimators(args.estimators)

    df, meta = run_ensemble(
        n=args.n,
        m=args.m,
        T=args.T,
        dt=args.dt,
        ensvol=args.ensvol,
        x0count=args.x0count,
        noise_std=args.noise_std,
        seed=args.seed,
        ensemble=args.ensemble,
        estimators=estimator_list,
    )

    df = add_transforms(df)

    estimator_cols = [c for c in df.columns if c.startswith("err_")]
    corr_table, corr_json = compute_correlations(df, estimator_cols)

    color_map = _build_deficiency_colors(df["deficiency"].tolist())

    tag = (
        f"{args.tag}_ens-{args.ensemble}_ensvol-{args.ensvol}_x0-{args.x0count}_"
        f"ests-{'-'.join(meta['estimators'])}"
    )

    df.to_csv(outdir / f"results_{tag}.csv", index=False)
    corr_table.to_csv(outdir / f"pearson_{tag}.csv", index=False)
    with open(outdir / f"pearson_{tag}.json", "w", encoding="utf-8") as fh:
        json.dump(corr_json, fh, indent=2)

    for ykey in estimator_cols:
        scatter_plots(df, ykey, plotdir, tag, color_map)
        if args.zoom:
            scatter_plots_zoom(df, ykey, plotdir, tag, color_map, args.zoom_q)

    np.savez(
        outdir / f"systems_{tag}.npz",
        A=meta["A"],
        B=meta["B"],
        Ad=meta["Ad"],
        Bd=meta["Bd"],
        deficiency=meta["deficiency"],
        ensemble=np.array(meta["ensemble"], dtype=object),
        estimators=np.array(meta["estimators"], dtype=object),
        n=np.array(meta["n"]),
        m=np.array(meta["m"]),
        dt=np.array(meta["dt"]),
        ensvol=np.array(meta["ensvol"]),
        x0count=np.array(meta["x0count"]),
    )

    print("Saved:")
    print("  ", outdir / f"results_{tag}.csv")
    print("  ", outdir / f"pearson_{tag}.csv")
    print("  ", outdir / f"pearson_{tag}.json")
    print("  ", outdir / f"systems_{tag}.npz")
    print("  plots ->", plotdir)


if __name__ == "__main__":
    main()