"""Pipeline sweeps for identifiability score correlations.

This module reuses the core simulation logic from :mod:`experiments.sim_scoree`
so we can explore how the correlations between identifiability proxies and
estimation error evolve along user-specified grids of parameters.

Supported sweep scenarios
-------------------------
* ``vary_m``  – keep ``n`` fixed and vary the number of inputs ``m``.
* ``vary_def`` – keep ``(n, m)`` fixed and vary the controllability
  deficiency ``d`` (where the controllability rank is ``n-d``).
* ``vary_T``  – keep ``(n, m, d)`` fixed and vary the trajectory length ``T``.

For each grid point the script:
  1. draws a random system, generates input trajectories, runs the estimator
     pipeline, and records the trial-wise scores and estimation errors;
  2. computes Spearman correlations between each score and estimation error;
  3. stores the raw trials, the per-grid correlation statistics, and a JSON
     manifest so future ``--just-plot`` calls can recover the correct files.

The ``--just-plot`` option reuses an existing CSV to create the full-scale and
zoomed scatter plots from :mod:`experiments.sim_scoree` for one grid point.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:  # package-style import
    from .sim_scoree import (
        add_transforms,
        run_trials,
        scatter_plots,
        scatter_plots_zoom,
    )
except ImportError:  # pragma: no cover - fallback for script execution
    from sim_scoree import (  # type: ignore
        add_transforms,
        run_trials,
        scatter_plots,
        scatter_plots_zoom,
    )


METRIC_COLUMNS: Sequence[tuple[str, str]] = (
    ("x_inv_pbh", "1 / PBH (structured)"),
    ("x_inv_krylov_smin", "1 / σ_min(K_n)"),
    ("x_inv_mu", "1 / mu_min"),
)


@dataclass(frozen=True)
class ScenarioInfo:
    name: str
    grid_param: str
    help: str


SCENARIOS: dict[str, ScenarioInfo] = {
    "vary_m": ScenarioInfo("vary_m", "m", "Sweep over the number of inputs m."),
    "vary_def": ScenarioInfo(
        "vary_def", "deficiency", "Sweep over controllability deficiency d."),
    "vary_T": ScenarioInfo("vary_T", "T", "Sweep over the trajectory length T."),
}


def parse_int_list(arg: str | None) -> list[int]:
    if arg is None:
        return []
    vals: list[int] = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    return vals


def ensure_grid(scenario: ScenarioInfo, values: Sequence[int]) -> list[int]:
    if not values:
        raise ValueError(
            f"No grid specified for {scenario.name!r}. Provide --{scenario.grid_param}-grid."
        )
    return list(values)


def _safe_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _sanitize_label(text: str) -> str:
    return (
        text.replace("/", "per")
        .replace(" ", "_")
        .replace("σ", "sigma")
        .replace("μ", "mu")
        .replace("(", "")
        .replace(")", "")
        .replace("_", "_")
    )


def _run_single_grid_point(
    *,
    scenario: ScenarioInfo,
    grid_index: int,
    grid_value: int,
    args: argparse.Namespace,
    outdir: pathlib.Path,
    est_list: Sequence[str],
) -> list[dict[str, object]]:
    """Execute the simulation pipeline for one grid value and save artifacts."""

    # Resolve scenario-dependent parameters.
    n = args.n
    m = args.m
    deficiency = args.deficiency
    T = args.T

    if scenario.name == "vary_m":
        m = grid_value
    elif scenario.name == "vary_def":
        deficiency = grid_value
        if deficiency < 0 or deficiency > n:
            raise ValueError(
                f"Invalid deficiency {deficiency} for n={n}. Must satisfy 0 ≤ d ≤ n."
            )
    elif scenario.name == "vary_T":
        T = grid_value
        if T <= 1:
            raise ValueError("Trajectory length T must be greater than 1.")

    df, meta = run_trials(
        n=n,
        m=m,
        T=T,
        dt=args.dt,
        trials=args.trials,
        noise_std=args.noise_std,
        seed=args.seed + grid_index * args.seed_stride,
        ensemble=args.ensemble,
        deficiency=deficiency,
        estimators=est_list,
    )
    df = add_transforms(df)

    grid_dir = outdir / f"{scenario.grid_param}_{grid_index:03d}"
    plot_dir = grid_dir / "plots"
    grid_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Persist raw data for later inspection / --just-plot runs.
    df.to_csv(grid_dir / "results.csv", index=False)

    manifest = {
        "scenario": scenario.name,
        "grid_param": scenario.grid_param,
        "grid_index": grid_index,
        "grid_value": grid_value,
        "grid_value_repr": str(grid_value),
        "base_parameters": {
            "n": n,
            "m": m,
            "deficiency": deficiency,
            "T": T,
            "dt": args.dt,
            "trials": args.trials,
            "seed": args.seed + grid_index * args.seed_stride,
            "ensemble": args.ensemble,
            "noise_std": args.noise_std,
            "estimators": list(est_list),
        },
    }
    with (grid_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    # Compute correlations for each metric/estimator pair.
    ycols = [c for c in df.columns if c.startswith("err_")]
    records: list[dict[str, object]] = []
    for metric_key, metric_label in METRIC_COLUMNS:
        xvals = df[metric_key].to_numpy()
        for y in ycols:
            yvals = df[y].to_numpy()
            mask = np.isfinite(xvals) & np.isfinite(yvals)
            n_eff = int(mask.sum())
            if n_eff >= 3:
                rho, p = spearmanr(xvals[mask], yvals[mask])
            else:
                rho, p = math.nan, math.nan
            records.append(
                {
                    "scenario": scenario.name,
                    "grid_param": scenario.grid_param,
                    "grid_index": grid_index,
                    "grid_value": grid_value,
                    "grid_value_display": str(grid_value),
                    "grid_value_numeric": _safe_float(grid_value),
                    "metric_key": metric_key,
                    "metric_label": metric_label,
                    "estimator": y.replace("err_", ""),
                    "spearman_rho": float(rho) if np.isfinite(rho) else math.nan,
                    "spearman_p": float(p) if np.isfinite(p) else math.nan,
                    "n_effective": n_eff,
                }
            )

    corr_df = pd.DataFrame.from_records(records)
    corr_df.to_csv(grid_dir / "correlations.csv", index=False)

    # Generate quick-look scatter plots for convenience.
    tag = f"{scenario.name}_{scenario.grid_param}-{grid_value}"
    for y in ycols:
        scatter_plots(df, y, plot_dir, tag)
        scatter_plots_zoom(df, y, plot_dir, tag, q_zoom=args.zoom_q)

    return records


def _plot_correlation_summary(
    corr_df: pd.DataFrame, scenario: ScenarioInfo, outdir: pathlib.Path
) -> None:
    if corr_df.empty:
        return

    plot_dir = outdir / "summary_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for (metric_label, estimator), group in corr_df.groupby(["metric_label", "estimator"]):
        yvals = group.sort_values("grid_index")
        if yvals["spearman_rho"].notna().sum() == 0:
            continue

        if np.isfinite(yvals["grid_value_numeric"].to_numpy()).all():
            x = yvals["grid_value_numeric"].to_numpy(dtype=float)
            xticks = x
            xticklabels = [str(v) for v in xticks]
        else:
            x = yvals["grid_index"].to_numpy(dtype=float)
            xticks = x
            xticklabels = yvals["grid_value_display"].tolist()

        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.plot(x, yvals["spearman_rho"].to_numpy(), marker="o", linestyle="-")
        ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel(scenario.grid_param)
        ax.set_ylabel("Spearman ρ")
        ax.set_title(f"{metric_label} vs. {estimator} (scenario: {scenario.name})")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

        fname = (
            f"corr_{_sanitize_label(metric_label)}_est-{estimator.replace(' ', '_')}"
            ".png"
        )
        fig.savefig(plot_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _load_manifest(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _select_grid_directory(
    scenario_dir: pathlib.Path,
    grid_param: str,
    *,
    focus_index: int | None,
    focus_value: float | None,
) -> pathlib.Path:
    if focus_index is not None:
        candidate = scenario_dir / f"{grid_param}_{focus_index:03d}"
        if not candidate.exists():
            raise FileNotFoundError(
                f"No grid directory {candidate} for index {focus_index}."
            )
        return candidate

    if focus_value is None:
        raise ValueError("Provide either --focus-index or --focus-value for --just-plot.")

    best: pathlib.Path | None = None
    for config_path in scenario_dir.glob(f"{grid_param}_*/config.json"):
        manifest = _load_manifest(config_path)
        gv = manifest.get("grid_value")
        gv_float = _safe_float(gv)
        if gv_float is None:
            continue
        if math.isclose(gv_float, focus_value, rel_tol=1e-9, abs_tol=1e-9):
            best = config_path.parent
            break
    if best is None:
        raise FileNotFoundError(
            f"Could not find a run with {grid_param}≈{focus_value}."
        )
    return best


def run_sweep(args: argparse.Namespace) -> None:
    scenario = SCENARIOS[args.scenario]
    outdir = pathlib.Path(args.outdir)
    scenario_dir = outdir / scenario.name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    est_list = [t.strip().lower() for t in args.estimators.split(",") if t.strip()]
    if not est_list:
        raise ValueError("No estimators specified.")

    grid_values = ensure_grid(scenario, parse_int_list(getattr(args, f"{scenario.grid_param}_grid")))

    all_records: list[dict[str, object]] = []
    for idx, value in enumerate(grid_values):
        records = _run_single_grid_point(
            scenario=scenario,
            grid_index=idx,
            grid_value=value,
            args=args,
            outdir=scenario_dir,
            est_list=est_list,
        )
        all_records.extend(records)

    corr_df = pd.DataFrame.from_records(all_records)
    corr_path = scenario_dir / "correlations_summary.csv"
    corr_df.to_csv(corr_path, index=False)

    _plot_correlation_summary(corr_df, scenario, scenario_dir)

    print("Saved sweep outputs to:")
    print("  ", scenario_dir)
    print("  ", corr_path)


def run_just_plot(args: argparse.Namespace) -> None:
    scenario = SCENARIOS[args.scenario]
    outdir = pathlib.Path(args.outdir)
    scenario_dir = outdir / scenario.name
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario directory {scenario_dir} does not exist.")

    focus_value = None
    if args.focus_value is not None:
        focus_value = float(args.focus_value)

    grid_dir = _select_grid_directory(
        scenario_dir,
        scenario.grid_param,
        focus_index=args.focus_index,
        focus_value=focus_value,
    )

    results_path = grid_dir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Expected results CSV at {results_path}.")

    df = pd.read_csv(results_path)
    if not set(col for col, _ in METRIC_COLUMNS).issubset(df.columns):
        df = add_transforms(df)

    ycols = [c for c in df.columns if c.startswith("err_")]
    if not ycols:
        raise RuntimeError("No estimation error columns found in the stored CSV.")

    plot_dir = grid_dir / "plots_refresh"
    plot_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = grid_dir / "config.json"
    tag = f"justplot_{scenario.name}_{scenario.grid_param}-{args.focus_index if args.focus_index is not None else args.focus_value}"  # noqa: E501
    if manifest_path.exists():
        manifest = _load_manifest(manifest_path)
        tag = f"justplot_{manifest['scenario']}_{manifest['grid_param']}-{manifest['grid_value_repr']}"

    for y in ycols:
        scatter_plots(df, y, plot_dir, tag)
        scatter_plots_zoom(df, y, plot_dir, tag, q_zoom=args.zoom_q)

    print("Refreshed scatter plots in", plot_dir)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", choices=SCENARIOS.keys(), required=True)
    ap.add_argument("--outdir", type=str, default="sim_score_sweep_out",
                    help="Base directory for sweep outputs.")

    # Base simulation parameters (overridden per-scenario as needed).
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--deficiency", type=int, default=1)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--seed-stride", type=int, default=100,
                    help="Offset added to the seed for each grid point (default 100).")
    ap.add_argument("--ensemble", type=str, default="ginibre",
                    choices=["ginibre", "stable", "sparse", "binary"])
    ap.add_argument("--estimators", type=str, default="dmdc,moesp",
                    help="Comma-separated estimator list (subset of sim_scoree options).")

    # Grids (scenario-specific).
    ap.add_argument("--m-grid", type=str, help="Comma-separated grid for m (vary_m scenario).")
    ap.add_argument("--deficiency-grid", type=str,
                    help="Comma-separated grid for deficiency d (vary_def scenario).")
    ap.add_argument("--T-grid", type=str, help="Comma-separated grid for T (vary_T scenario).")

    # Plotting controls.
    ap.add_argument("--zoom-q", type=float, default=0.9,
                    help="Quantile used for zoomed scatter plots (default 0.9).")

    # Just-plot controls.
    ap.add_argument("--just-plot", action="store_true",
                    help="Reuse stored CSVs to regenerate scatter plots for one grid point.")
    ap.add_argument("--focus-index", type=int,
                    help="Grid index to target when using --just-plot (0-based).")
    ap.add_argument("--focus-value", type=float,
                    help="Grid value to target when using --just-plot.")

    return ap


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.just_plot:
        run_just_plot(args)
    else:
        run_sweep(args)


if __name__ == "__main__":  # pragma: no cover
    main()