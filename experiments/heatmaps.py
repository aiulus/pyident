"""Offline heatmap generation for ``sim_regcomb`` experiment summaries.

This module provides a lightweight plotting entry-point that can be run after
the computationally expensive portion of the ``sim_regcomb`` experiment has
finished.  It reads the ``scores_summary.csv`` file produced by the experiment
and recreates only the thresholded heatmaps where entries that fall below a
specified threshold are highlighted in red.

Example
-------

To regenerate the heatmaps for a finished experiment::

    python -m pyident.offline.sim_regcomb_heatmaps \
        --summary results/sim3_sparse_state/scores_summary.csv \
        --threshold 1e-2

The generated plots are stored alongside the summary file inside a dedicated
``plots_offline`` directory unless a different output directory is provided.
"""

from __future__ import annotations

import argparse
import math
import pathlib
from typing import Iterable

try:  # pragma: no cover - import guard for optional dependency
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without matplotlib
    plt = None  # type: ignore[assignment]
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None

import numpy as np
import pandas as pd

from ..experiments import sim_regcomb

# ``AXIS_COLUMN`` maps logical axis names ("sparsity", "ndim", ...)
# to the column names that appear in the summary CSV.  We require the
# reverse mapping for labelling.
AXIS_FROM_COLUMN = {value: key for key, value in sim_regcomb.AXIS_COLUMN.items()}


def _resolve_axes(columns: Iterable[str]) -> list[str]:
    """Resolve axis columns that are present in the summary table."""

    axes = []
    for column in columns:
        if not column.startswith("axis_"):
            continue
        axes.append(column)
    if len(axes) < 2:
        raise ValueError(
            "The provided summary does not contain two axis columns; "
            "did you run a two-axis ``sim_regcomb`` experiment?"
        )
    return axes[:2]


def _make_plot_name(score: str, axis_columns: tuple[str, str]) -> str:
    axis_names = [AXIS_FROM_COLUMN.get(col, col) for col in axis_columns]
    axis_part = "_".join(axis_names)
    return f"{score}_heatmap_{axis_part}_thr".replace(",", "_")


def _plot_threshold_heatmap(
    data: np.ndarray,
    pivot_columns: Iterable,
    pivot_index: Iterable,
    axis_columns: tuple[str, str],
    score: str,
    subset: pd.DataFrame,
    threshold: float,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a heatmap figure that highlights values below ``threshold``."""

    if plt is None:  # pragma: no cover - handled above, kept for type-checkers
        raise RuntimeError("matplotlib is required for plotting")

    x_axis_col, y_axis_col = axis_columns
    x_axis_name = AXIS_FROM_COLUMN.get(x_axis_col, x_axis_col)
    y_axis_name = AXIS_FROM_COLUMN.get(y_axis_col, y_axis_col)

    x_label = sim_regcomb.AXIS_LABEL.get(x_axis_name, x_axis_name)
    y_label = sim_regcomb.AXIS_LABEL.get(y_axis_name, y_axis_name)

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
        [sim_regcomb.format_axis_tick(x_axis_name, value) for value in pivot_columns]
    )
    ax.set_yticklabels(
        [sim_regcomb.format_axis_tick(y_axis_name, value) for value in pivot_index]
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    threshold_title = sim_regcomb.make_heatmap_title(x_axis_name, y_axis_name, score, subset)
    ax.set_title(f"{threshold_title} (red < {threshold:.1e})")

    special_state_under = x_axis_name == "ndim" and y_axis_name == "underactuation"
    if special_state_under and data.size:
        column_values = list(pivot_columns)
        row_values = list(pivot_index)
        diag_coords: list[tuple[float, float]] = []
        for col_idx, col_val in enumerate(column_values):
            for row_idx, row_val in enumerate(row_values):
                if math.isclose(float(col_val), float(row_val), rel_tol=0.0, abs_tol=1e-9):
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
                rotation = math.degrees(math.atan2(y_end - y_start, x_end - x_start))
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

    if not math.isnan(threshold):
        mask = data < threshold
    else:
        mask = np.zeros_like(data, dtype=bool)
    if np.any(mask):
        red_overlay = np.zeros((n_rows, n_cols, 4), dtype=float)
        red_overlay[mask] = (1.0, 0.0, 0.0, 1.0)
        ax.imshow(
            red_overlay,
            origin="lower",
            aspect="auto",
            extent=(-0.5, n_cols - 0.5, -0.5, n_rows - 0.5),
        )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    score_label = sim_regcomb.SCORE_DISPLAY_NAMES.get(score, score)
    cbar.set_label(f"{score_label} (mean; red < {threshold:.1e})")
    fig.tight_layout()
    return fig, ax


def generate_threshold_plots(
    summary_path: pathlib.Path,
    threshold: float,
    output_dir: pathlib.Path | None = None,
) -> list[pathlib.Path]:
    """Generate thresholded heatmaps for the provided summary CSV."""

    if plt is None:  # pragma: no cover - exercised only when matplotlib is missing
        raise RuntimeError(
            "matplotlib is required for plotting; please install it to run this script"
        ) from _MATPLOTLIB_IMPORT_ERROR

    summary_df = pd.read_csv(summary_path)
    axis_columns = _resolve_axes(summary_df.columns)
    x_axis_col, y_axis_col = axis_columns[:2]

    axis_tuple = (x_axis_col, y_axis_col)
    base_dir = summary_path.parent if output_dir is None else output_dir
    plot_dir = base_dir / "plots_offline"
    plot_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[pathlib.Path] = []
    for score_name in summary_df["score"].unique():
        subset = summary_df[summary_df["score"] == score_name]
        if subset.empty:
            continue
        if x_axis_col not in subset.columns or y_axis_col not in subset.columns:
            continue

        pivot = subset.pivot(index=y_axis_col, columns=x_axis_col, values="mean")
        pivot = pivot.sort_index().sort_index(axis=1)
        if pivot.empty:
            continue

        data = pivot.to_numpy()
        fig, _ = _plot_threshold_heatmap(
            data=data,
            pivot_columns=pivot.columns,
            pivot_index=pivot.index,
            axis_columns=axis_tuple,
            score=score_name,
            subset=subset,
            threshold=threshold,
        )

        plot_name = _make_plot_name(score_name, axis_tuple)
        output_path = plot_dir / f"{plot_name}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        saved_paths.append(output_path)

    if not saved_paths:
        raise RuntimeError(
            "No heatmaps were generated. Ensure the summary contains "
            "two axis columns and at least one score with data."
        )

    return saved_paths


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=pathlib.Path,
        required=True,
        help="Path to the `scores_summary.csv` file produced by `sim_regcomb`.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-12,
        help="Threshold below which cells are highlighted in red.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional directory where the `plots_offline` folder will be created. "
            "Defaults to the parent directory of the summary file."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    summary_path = args.summary
    if summary_path.suffix.lower() != ".csv":
        raise ValueError("--summary must point to a CSV file")
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    saved_paths = generate_threshold_plots(
        summary_path=summary_path,
        threshold=float(args.threshold),
        output_dir=args.output_dir,
    )

    for path in saved_paths:
        print(f"Saved heatmap: {path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()