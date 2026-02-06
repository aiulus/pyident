"""Controllability fraction conditioned on system properties.

This experiment mirrors ``sim_regcomb`` but replaces x0-based score
statistics with the fraction of controllable pairs (A, B), i.e. the
fraction of draws where rank(C_n(A,B)) = n.

Example
-------
```
python -m pyident.experiments.sim_regcomb_ctrb --axes "sparsity, ndim" \
    --sparsity-grid 0.0:0.1:1.0 --ndim-grid 2:2:20 --samples 100 \
    --outdir results/sim3_sparse_state
```
"""
from __future__ import annotations

import argparse
import math
import pathlib
from collections import defaultdict
from typing import Any, Mapping, Sequence, Tuple

try:  # pragma: no cover - import guard for optional dependency
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without matplotlib
    plt = None  # type: ignore[assignment]
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None
import numpy as np
import pandas as pd

from ..ensembles import controllability_rank
from . import sim_regcomb as base


SCORE_NAME = "controllable_fraction"
SCORE_DISPLAY_NAMES = {
    SCORE_NAME: "Controllable fraction",
}


def make_heatmap_title(x_axis: str, y_axis: str, data: pd.DataFrame) -> str:
    """Construct a descriptive heatmap title for controllability summaries."""

    x_label = base.AXIS_TITLE_NAMES.get(x_axis, x_axis.title())
    y_label = base.AXIS_TITLE_NAMES.get(y_axis, y_axis.title())
    base_title = f"{x_label} vs. {y_label}"

    if (x_axis, y_axis) == ("underactuation", "sparsity") and "n" in data:
        n_values = sorted({int(val) for val in data["n"].dropna().unique()})
        if n_values:
            if len(n_values) == 1:
                suffix = f", n = {n_values[0]}"
            else:
                suffix = ", n ∈ {" + ", ".join(str(val) for val in n_values) + "}"
            return f"{base_title} (controllable fraction{suffix})"

    return f"{base_title} (controllable fraction)"


def _legacy_scenarios(args: argparse.Namespace) -> list[tuple[str, float, int, int, dict[str, Any]]]:
    property_name = args.property
    scenarios: list[tuple[str, float, int, int, dict[str, Any]]] = []

    if property_name == "underactuation":
        n_grid_spec = args.n_grid or args.cond_grid
        if n_grid_spec is None:
            raise ValueError("--cond-grid or --n-grid must be provided for property=underactuation")
        if args.m_grid is None:
            raise ValueError("--m-grid must be provided for property=underactuation")
        n_values = base.parse_grid(n_grid_spec)
        m_values = base.parse_grid(args.m_grid)
        for n_val in n_values:
            n_cur = int(round(n_val))
            if n_cur <= 0:
                raise ValueError("state dimensions must be positive integers")
            for m_val in m_values:
                m_cur = int(round(m_val))
                if m_cur <= 0:
                    raise ValueError("input dimensions must be positive integers")
                prop_val = float(n_cur - m_cur)
                info = {
                    "property": property_name,
                    "property_value": prop_val,
                    "n": n_cur,
                    "m": m_cur,
                    "input_fraction": float(m_cur) / float(n_cur),
                }
                scenarios.append((property_name, prop_val, n_cur, m_cur, info))
        return scenarios

    conditioning_values = base.parse_grid(args.cond_grid)
    for prop_value in conditioning_values:
        info = {
            "property": property_name,
            "property_value": float(prop_value),
            "n": args.n,
            "m": args.m,
        }
        scenarios.append((property_name, float(prop_value), args.n, args.m, info))
    return scenarios


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    outdir = base.ensure_output_dir(pathlib.Path(args.outdir))

    axes = base.parse_axes_spec(args.axes)
    if axes:
        if args.property is not None and args.property != "density":
            raise ValueError("--property cannot be combined with --axes")
        scenarios = base.build_axis_scenarios(args, axes)
    else:
        scenarios = _legacy_scenarios(args)

    if not scenarios:
        raise ValueError("no scenarios were generated; please check the grid specifications")

    accum: dict[tuple[str, Tuple[Tuple[str, Any], ...]], tuple[base.Moment, list[float]]] = defaultdict(
        lambda: (base.Moment(), [])
    )
    system_records: list[dict] = []
    A_list: list[np.ndarray] = []
    B_list: list[np.ndarray] = []

    for property_name, prop_value, n_cur, m_cur, prop_info in scenarios:
        prop_key = base.freeze_items(prop_info)

        for _ in range(args.samples):
            A, B, meta = base.generate_system(
                property_name,
                prop_value,
                n_cur,
                m_cur,
                rng,
                sparse_which=args.sparse_which,
                sparse_tol=args.sparse_tol,
                base_density=args.sparse_density,
                deficiency_base=args.deficiency_base,
                deficiency_embed_random=not args.deficiency_no_embed,
            )

            rk, _ = controllability_rank(A, B, order=n_cur, rtol=args.ctrb_rtol)
            controllable = float(rk == n_cur)

            sys_record = {
                "system_index": len(system_records),
                "n": n_cur,
                "m": m_cur,
                "controllability_rank": int(rk),
                "controllable": int(controllable),
            }
            for key, value in prop_info.items():
                if np.isscalar(value):
                    sys_record[key] = value

            meta.setdefault("density_A", base.matrix_density(A, tol=args.sparse_tol))
            meta.setdefault("density_B", base.matrix_density(B, tol=args.sparse_tol))
            meta.setdefault(
                "density_AB", base.matrix_density(np.hstack([A, B]), tol=args.sparse_tol)
            )
            meta.setdefault("underactuation", float(n_cur - m_cur))
            meta.setdefault(
                "input_fraction", float(m_cur) / float(n_cur) if n_cur else float("nan")
            )
            for key, value in meta.items():
                if np.isscalar(value):
                    sys_record[f"meta_{key}"] = value

            system_records.append(sys_record)
            A_list.append(A)
            B_list.append(B)

            key = (SCORE_NAME, prop_key)
            mom, buf = accum[key]
            mom.update(controllable)
            buf.append(controllable)
            accum[key] = (mom, buf)

    summary_rows = []
    for (score_name, prop_key), (moment, buf) in accum.items():
        q05 = np.nan
        q50 = np.nan
        q95 = np.nan
        if buf:
            arr = np.array(buf, dtype=float)
            q05, q50, q95 = np.quantile(arr, [0.05, 0.5, 0.95])
        prop_info = dict(prop_key)
        summary_rows.append(
            {
                "score": score_name,
                **prop_info,
                "count": moment.count,
                "mean": moment.mean,
                "std": moment.std,
                "q05": q05,
                "q50": q50,
                "q95": q95,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        raise RuntimeError("no summary statistics were computed; check the configuration")

    sort_cols = ["score"]
    for col in ["property_value", "n", "m", *base.AXIS_COLUMN.values()]:
        if col in summary_df.columns:
            sort_cols.append(col)
    summary_df = summary_df.sort_values(sort_cols)
    summary_path = outdir / "scores_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    systems_df = pd.DataFrame(system_records)
    systems_df.to_csv(outdir / "systems.csv", index=False)
    if args.save_matrices:
        matrices_path = outdir / "systems_matrices.npz"
        np.savez_compressed(
            matrices_path,
            A=np.array(A_list, dtype=object),
            B=np.array(B_list, dtype=object),
            system_index=systems_df["system_index"].to_numpy(),
            n=systems_df["n"].to_numpy(),
            m=systems_df["m"].to_numpy(),
        )

    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plotting; please install it to run this experiment"
        ) from _MATPLOTLIB_IMPORT_ERROR

    score_label = SCORE_DISPLAY_NAMES.get(SCORE_NAME, SCORE_NAME)
    heat_threshold = float(getattr(args, "heatthr", 1e-12))

    if axes:
        axis_columns = [base.AXIS_COLUMN[a] for a in axes]
        if len(axes) == 2:
            x_axis, y_axis = axis_columns[0], axis_columns[1]
            x_label = base.AXIS_LABEL[axes[0]]
            y_label = base.AXIS_LABEL[axes[1]]

            sub = summary_df[summary_df["score"] == SCORE_NAME]
            if not sub.empty and x_axis in sub.columns and y_axis in sub.columns:
                pivot = sub.pivot(index=y_axis, columns=x_axis, values="mean")
                pivot = pivot.sort_index().sort_index(axis=1)
                if not pivot.empty:
                    data = pivot.to_numpy()
                    n_rows, n_cols = data.shape

                    fig, ax = plt.subplots(figsize=(7.2, 5.2))
                    im = ax.imshow(
                        data,
                        origin="lower",
                        aspect="auto",
                        extent=(-0.5, n_cols - 0.5, -0.5, n_rows - 0.5),
                    )
                    ax.set_xlim(-0.5, n_cols - 0.5)
                    ax.set_ylim(-0.5, n_rows - 0.5)
                    ax.set_xticks(np.arange(n_cols))
                    ax.set_yticks(np.arange(n_rows))
                    ax.set_xticklabels(
                        [base.format_axis_tick(axes[0], value) for value in pivot.columns]
                    )
                    ax.set_yticklabels(
                        [base.format_axis_tick(axes[1], value) for value in pivot.index]
                    )

                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.set_title(make_heatmap_title(axes[0], axes[1], sub))

                    special_state_under = axes[0] == "ndim" and axes[1] == "underactuation"
                    if special_state_under and pivot.size:
                        column_values = list(pivot.columns)
                        row_values = list(pivot.index)
                        diag_coords: list[tuple[float, float]] = []
                        for col_idx, col_val in enumerate(column_values):
                            for row_idx, row_val in enumerate(row_values):
                                if math.isclose(
                                    float(col_val), float(row_val), rel_tol=0.0, abs_tol=1e-9
                                ):
                                    diag_coords.append((float(col_idx), float(row_idx)))
                                    break
                        if diag_coords:
                            xs, ys = zip(*diag_coords)
                            ax.plot(xs, ys, color="red", linewidth=2.0, solid_capstyle="round")
                            x_start, x_end = xs[0], xs[-1]
                            y_start, y_end = ys[0], ys[-1]
                            x_mid = 0.5 * (x_start + x_end)
                            y_mid = 0.5 * (y_start + y_end)
                            if len(xs) > 1:
                                rotation = math.degrees(
                                    math.atan2(y_end - y_start, x_end - x_start)
                                )
                            else:
                                rotation = 45.0
                            ax.text(
                                x_mid,
                                y_mid - 0.35,
                                "n = m",
                                color="red",
                                fontsize=8,
                                rotation=rotation,
                                rotation_mode="anchor",
                                ha="center",
                                va="center",
                            )

                    cbar = fig.colorbar(im, ax=ax, pad=0.02)
                    cbar.set_label(f"{score_label} (mean)")
                    fig.tight_layout()
                    plot_path = (
                        outdir
                        / "plots"
                        / f"{SCORE_NAME}_heatmap_{axes[0]}_{axes[1]}.png"
                    )
                    fig.savefig(plot_path, dpi=200)
                    base_cmap = im.get_cmap()
                    base_norm = im.norm
                    plt.close(fig)

                    thr_fig, thr_ax = plt.subplots(figsize=(7.2, 5.2))
                    thr_im = thr_ax.imshow(
                        data,
                        origin="lower",
                        aspect="auto",
                        extent=(-0.5, n_cols - 0.5, -0.5, n_rows - 0.5),
                        cmap=base_cmap,
                        norm=base_norm,
                    )
                    thr_ax.set_xlim(-0.5, n_cols - 0.5)
                    thr_ax.set_ylim(-0.5, n_rows - 0.5)
                    thr_ax.set_xticks(np.arange(n_cols))
                    thr_ax.set_yticks(np.arange(n_rows))
                    thr_ax.set_xticklabels(
                        [base.format_axis_tick(axes[0], value) for value in pivot.columns]
                    )
                    thr_ax.set_yticklabels(
                        [base.format_axis_tick(axes[1], value) for value in pivot.index]
                    )

                    thr_ax.set_xlabel(x_label)
                    thr_ax.set_ylabel(y_label)
                    thr_ax.set_title(
                        f"{make_heatmap_title(axes[0], axes[1], sub)} (red < {heat_threshold:.1e})"
                    )

                    if special_state_under and pivot.size:
                        column_values = list(pivot.columns)
                        row_values = list(pivot.index)
                        diag_coords = []
                        for col_idx, col_val in enumerate(column_values):
                            for row_idx, row_val in enumerate(row_values):
                                if math.isclose(
                                    float(col_val), float(row_val), rel_tol=0.0, abs_tol=1e-9
                                ):
                                    diag_coords.append((float(col_idx), float(row_idx)))
                                    break
                        if diag_coords:
                            xs, ys = zip(*diag_coords)
                            thr_ax.plot(xs, ys, color="red", linewidth=2.0, solid_capstyle="round")
                            x_start, x_end = xs[0], xs[-1]
                            y_start, y_end = ys[0], ys[-1]
                            x_mid = 0.5 * (x_start + x_end)
                            y_mid = 0.5 * (y_start + y_end)
                            if len(xs) > 1:
                                rotation = math.degrees(
                                    math.atan2(y_end - y_start, x_end - x_start)
                                )
                            else:
                                rotation = 45.0
                            thr_ax.text(
                                x_mid,
                                y_mid - 0.35,
                                "n = m",
                                color="red",
                                fontsize=8,
                                rotation=rotation,
                                rotation_mode="anchor",
                                ha="center",
                                va="center",
                            )

                    if not math.isnan(heat_threshold):
                        mask = data < heat_threshold
                    else:
                        mask = np.zeros_like(data, dtype=bool)
                    if np.any(mask):
                        red_overlay = np.zeros((n_rows, n_cols, 4), dtype=float)
                        red_overlay[mask] = (1.0, 0.0, 0.0, 1.0)
                        thr_ax.imshow(
                            red_overlay,
                            origin="lower",
                            aspect="auto",
                            extent=(-0.5, n_cols - 0.5, -0.5, n_rows - 0.5),
                        )

                    thr_cbar = thr_fig.colorbar(thr_im, ax=thr_ax, pad=0.02)
                    thr_cbar.set_label(
                        f"{score_label} (mean; red < {heat_threshold:.1e})"
                    )
                    thr_fig.tight_layout()
                    thr_path = (
                        outdir
                        / "plots"
                        / f"{SCORE_NAME}_heatmap_{axes[0]}_{axes[1]}_thr.png"
                    )
                    thr_fig.savefig(thr_path, dpi=200)
                    plt.close(thr_fig)

            return

        axis = axes[0]
        axis_col = axis_columns[0]
        axis_label = base.AXIS_LABEL[axis]

        sub = summary_df[summary_df["score"] == SCORE_NAME]
        if sub.empty or axis_col not in sub.columns:
            return

        sub = sub.sort_values(axis_col)
        x = sub[axis_col].to_numpy()
        mean = sub["mean"].to_numpy()
        std = sub["std"].to_numpy()

        plt.figure(figsize=(6.4, 4.0))
        plt.plot(x, mean, marker="o", label=f"{score_label} mean")
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, label="±1 std")
        plt.xlabel(axis_label)
        plt.ylabel(score_label)
        plt.title(f"{score_label} vs {axis}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plot_path = outdir / "plots" / f"{SCORE_NAME}_vs_{axis}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        return

    xlabel = {
        "density": "Density level",
        "deficiency": "Controllability deficiency",
        "state_dimension": "State dimension n",
    }.get(args.property, args.property)

    sub = summary_df[summary_df["score"] == SCORE_NAME]
    if sub.empty:
        return

    sub = sub.sort_values("property_value")
    x = sub["property_value"].to_numpy()
    mean = sub["mean"].to_numpy()
    std = sub["std"].to_numpy()

    plt.figure(figsize=(6.4, 4.0))
    plt.plot(x, mean, marker="o", label=f"{score_label} mean")
    plt.fill_between(x, mean - std, mean + std, alpha=0.2, label="±1 std")
    plt.xlabel(xlabel)
    plt.ylabel(score_label)
    plt.title(f"{score_label} vs {args.property}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plot_path = outdir / "plots" / f"{SCORE_NAME}_vs_{args.property}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="state dimension")
    parser.add_argument(
        "--m",
        type=int,
        default=None,
        help="input dimension (defaults to n when omitted)",
    )
    parser.add_argument(
        "--samples", type=int, default=200, help="number of (A,B) draws per conditioning value"
    )
    parser.add_argument(
        "--axes",
        default=None,
        help="comma-separated list of axes to sweep (subset of: sparsity, ndim, underactuation)",
    )
    parser.add_argument(
        "--property",
        default="density",
        choices=["density", "deficiency", "state_dimension", "underactuation"],
        help="system property to condition on",
    )
    parser.add_argument(
        "--cond-grid",
        default="0:0.05:1",
        help="conditioning grid specification (e.g., '0:0.05:1' or '0,0.5,1')",
    )
    parser.add_argument(
        "--n-grid",
        default=None,
        help="grid specification for state dimension sweeps (overrides --cond-grid for n)",
    )
    parser.add_argument(
        "--m-grid",
        default=None,
        help="grid specification for input dimension sweeps when property=underactuation",
    )
    parser.add_argument(
        "--sparsity-grid",
        default=None,
        help="grid specification for sparsity sweeps when using --axes",
    )
    parser.add_argument(
        "--ndim-grid",
        default=None,
        help="grid specification for state dimension sweeps when using --axes",
    )
    parser.add_argument("--seed", type=int, default=12345, help="base RNG seed")
    parser.add_argument("--outdir", default="results/sim3", help="output directory")

    parser.add_argument(
        "--ctrb-rtol",
        type=float,
        default=None,
        help="relative tolerance for controllability rank (default: numpy SVD tolerance)",
    )

    parser.add_argument(
        "--sparse-which",
        default="both",
        choices=["A", "B", "both"],
        help="which matrices to sparsify when property=density",
    )
    parser.add_argument(
        "--sparse-tol",
        type=float,
        default=1e-12,
        help="tolerance when measuring realised density",
    )
    parser.add_argument(
        "--sparse-density",
        type=float,
        default=0.3,
        help="baseline density used when property is not 'density'",
    )

    parser.add_argument(
        "--heatthr",
        type=float,
        default=1e-12,
        help=(
            "heatmap threshold: cells with mean scores below this value are coloured red"
        ),
    )

    parser.add_argument(
        "--deficiency-base",
        default="ginibre",
        choices=["ginibre", "stable", "binary", "sparse"],
        help="base ensemble for controllable/uncontrollable blocks",
    )
    parser.add_argument(
        "--deficiency-no-embed",
        action="store_true",
        help="disable random basis embedding in draw_with_ctrb_rank",
    )
    parser.add_argument(
        "--save-matrices",
        action="store_true",
        help="save A,B draws to systems_matrices.npz for downstream filtering",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.m is None:
        args.m = int(args.n)
    run(args)


if __name__ == "__main__":
    main()
