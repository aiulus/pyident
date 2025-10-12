from __future__ import annotations

import argparse
import pathlib
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from .sim_escon import (
    EstimatorConsistencyConfig,
    VisibilitySweepResult,
    run_visibility_sweep,
)


def _boxplot_with_zero_floor(
    ax: plt.Axes,
    data: Sequence[np.ndarray],
    *,
    zero_thresh: float = 1e-10,
    positions: Optional[Sequence[float]] = None,
    widths: Optional[Union[float, Sequence[float]]] = None,
    **kwargs,
):
    """Draw a boxplot and replace near-zero series with a baseline bar."""

    cleaned: List[np.ndarray] = []
    zero_flags: List[bool] = []
    for arr in data:
        if arr is None:
            cleaned.append(np.asarray([np.nan]))
            zero_flags.append(False)
            continue

        arr = np.asarray(arr, dtype=float)
        finite_vals = arr[np.isfinite(arr)]
        if finite_vals.size == 0:
            cleaned.append(np.asarray([np.nan]))
            zero_flags.append(False)
            continue
        cleaned.append(finite_vals)
        zero_flags.append(float(np.max(finite_vals)) < zero_thresh)

    bp = ax.boxplot(cleaned, positions=positions, widths=widths, **kwargs)

    if not any(zero_flags):
        return bp

    if positions is None:
        pos_arr = np.arange(1, len(cleaned) + 1, dtype=float)
    else:
        pos_arr = np.asarray(positions, dtype=float)

    if widths is None:
        width_arr = np.full(len(cleaned), 0.5, dtype=float)
    elif np.isscalar(widths):
        width_arr = np.full(len(cleaned), float(widths))
    else:
        width_arr = np.asarray(widths, dtype=float)

    for idx, use_bar in enumerate(zero_flags):
        if not use_bar:
            continue

        edge_color = None
        if "boxes" in bp and len(bp["boxes"]) > idx:
            box_artist = bp["boxes"][idx]
            box_artist.set_visible(False)

            if hasattr(box_artist, "get_edgecolor"):
                edge_color = box_artist.get_edgecolor()
            elif hasattr(box_artist, "get_color"):
                edge_color = box_artist.get_color()
            elif hasattr(box_artist, "get_facecolor"):
                edge_color = box_artist.get_facecolor()

        if "medians" in bp:
            bp["medians"][idx].set_visible(False)

        if "whiskers" in bp:
            bp["whiskers"][2 * idx].set_visible(False)
            bp["whiskers"][2 * idx + 1].set_visible(False)

        if "caps" in bp:
            bp["caps"][2 * idx].set_visible(False)
            bp["caps"][2 * idx + 1].set_visible(False)

        if "fliers" in bp and bp["fliers"]:
            bp["fliers"][idx].set_visible(False)

        if isinstance(edge_color, (list, tuple, np.ndarray)):
            edge_color = np.asarray(edge_color)
            if edge_color.ndim > 1:
                edge_color = edge_color[0]
            edge_color = tuple(edge_color)

        if edge_color is None:
            edge_color = "C0"

        half_width = 0.5 * float(width_arr[idx])
        x0 = pos_arr[idx] - half_width
        x1 = pos_arr[idx] + half_width
        ax.hlines(0.0, x0, x1, colors=edge_color, linewidth=3.0, zorder=5)

    return bp


def _compute_ylim(
    *data_groups: Sequence[Sequence[np.ndarray]],
    pad_frac: float = 0.05,
) -> Tuple[float, float]:
    """Return y-axis limits that cover the provided ``boxplot`` data."""

    collected: List[np.ndarray] = []
    for group in data_groups:
        for arr in group:
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float)
            if arr.size == 0:
                continue
            finite_vals = arr[np.isfinite(arr)]
            if finite_vals.size:
                collected.append(finite_vals)

    if not collected:
        return 0.0, 1.0

    max_val = max(float(np.max(vals)) for vals in collected)
    if not np.isfinite(max_val):
        max_val = 1.0
    if max_val <= 0:
        max_val = 1e-6
    pad = max(pad_frac * max_val, 1e-8)
    return 0.0, max_val + pad


def _render_single_mode_figures(
    result: VisibilitySweepResult,
    out_dir: pathlib.Path,
) -> None:
    if not result.dims_with_data:
        raise RuntimeError(
            f"No successful trials recorded for algorithm {result.algorithm}; unable to plot."
        )

    dims_desc = sorted(result.dims_with_data, reverse=True)
    xticks = list(range(1, len(dims_desc) + 1))
    xticklabels = [str(dim) for dim in dims_desc]

    dataA_std = [np.asarray(result.errA_standard[dim], float) for dim in dims_desc]
    dataB_std = [np.asarray(result.errB_standard[dim], float) for dim in dims_desc]
    dataA_vis = [np.asarray(result.errA_visible[dim], float) for dim in dims_desc]
    dataB_vis = [np.asarray(result.errB_visible[dim], float) for dim in dims_desc]
    uni_std = [np.asarray(result.unified_standard[dim], float) for dim in dims_desc]
    uni_vis = [np.asarray(result.unified_visible[dim], float) for dim in dims_desc]

    safe_name = result.algorithm.replace(" ", "_")

    fig_uni_std, ax_uni_std = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    _boxplot_with_zero_floor(ax_uni_std, uni_std, whis=(5, 95), showfliers=False)
    ax_uni_std.set_title(
        f"{result.algorithm}: Unified error (A-B mean) — Standard basis"
    )
    ax_uni_std.set_xlabel("dim $V(x_0)$")
    ax_uni_std.set_ylabel("Relative error (A-B mean)")
    ax_uni_std.set_xticks(xticks, xticklabels)
    ax_uni_std.grid(True, axis="y", linestyle="--", alpha=0.6)
    fig_uni_std.tight_layout()
    fig_uni_std.savefig(out_dir / f"single_{safe_name}_standard.png", dpi=150)
    plt.close(fig_uni_std)

    fig_uni_vis, ax_uni_vis = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    _boxplot_with_zero_floor(ax_uni_vis, uni_vis, whis=(5, 95), showfliers=False)
    ax_uni_vis.set_title(
        f"{result.algorithm}: Unified error (A-B mean) — V(x0)-basis"
    )
    ax_uni_vis.set_xlabel("dim $V(x_0)$")
    ax_uni_vis.set_ylabel("Relative error (A-B mean)")
    ax_uni_vis.set_xticks(xticks, xticklabels)
    ax_uni_vis.grid(True, axis="y", linestyle="--", alpha=0.6)
    fig_uni_vis.tight_layout()
    fig_uni_vis.savefig(out_dir / f"single_{safe_name}_Vx0basis.png", dpi=150)
    plt.close(fig_uni_vis)

    fig_std_AB, axes_std_AB = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)
    _boxplot_with_zero_floor(axes_std_AB[0], dataA_std, whis=(5, 95), showfliers=False)
    axes_std_AB[0].set_title("A — Standard basis")
    axes_std_AB[0].set_ylabel("Relative error")
    axes_std_AB[0].grid(True, axis="y", linestyle="--", alpha=0.6)

    _boxplot_with_zero_floor(axes_std_AB[1], dataB_std, whis=(5, 95), showfliers=False)
    axes_std_AB[1].set_title("B — Standard basis")
    axes_std_AB[1].set_xlabel("dim $V(x_0)$")
    axes_std_AB[1].set_ylabel("Relative error")
    axes_std_AB[1].set_xticks(xticks, xticklabels)
    axes_std_AB[1].grid(True, axis="y", linestyle="--", alpha=0.6)

    ymin_std, ymax_std = _compute_ylim(dataA_std, dataB_std)
    for ax in axes_std_AB:
        ax.set_ylim(ymin_std, ymax_std)

    fig_std_AB.tight_layout()
    fig_std_AB.savefig(out_dir / f"single_{safe_name}_standard_AB.png", dpi=150)
    plt.close(fig_std_AB)

    fig_mixed_AB, axes_mixed_AB = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)
    _boxplot_with_zero_floor(axes_mixed_AB[0], dataA_std, whis=(5, 95), showfliers=False)
    axes_mixed_AB[0].set_title("A — Standard basis")
    axes_mixed_AB[0].set_ylabel("Relative error")
    axes_mixed_AB[0].grid(True, axis="y", linestyle="--", alpha=0.6)

    _boxplot_with_zero_floor(axes_mixed_AB[1], dataB_vis, whis=(5, 95), showfliers=False)
    axes_mixed_AB[1].set_title("B — V(x0)-basis")
    axes_mixed_AB[1].set_xlabel("dim $V(x_0)$")
    axes_mixed_AB[1].set_ylabel("Relative error")
    axes_mixed_AB[1].set_xticks(xticks, xticklabels)
    axes_mixed_AB[1].grid(True, axis="y", linestyle="--", alpha=0.6)

    ymin_mixed, ymax_mixed = _compute_ylim(dataA_std, dataB_vis)
    for ax in axes_mixed_AB:
        ax.set_ylim(ymin_mixed, ymax_mixed)

    fig_mixed_AB.tight_layout()
    fig_mixed_AB.savefig(out_dir / f"single_{safe_name}_Vx0basis_AB.png", dpi=150)
    plt.close(fig_mixed_AB)


def run_single_mode_figures(
    cfg: Optional[EstimatorConsistencyConfig] = None,
    algos: Optional[Sequence[str]] = None,
    ensemble_size: int = 200,
    out_dir: Optional[pathlib.Path] = None,
) -> Dict[str, VisibilitySweepResult]:
    """Run the visibility sweep and save single-mode figures."""

    if cfg is None:
        cfg = EstimatorConsistencyConfig()

    out_path = out_dir or (cfg.save_dir / "vis_sweep")
    out_path.mkdir(parents=True, exist_ok=True)

    results = run_visibility_sweep(
        cfg,
        algos=algos,
        ensemble_size=ensemble_size,
    )

    for result in results.values():
        if not result.dims_with_data:
            print(
                f"[{result.algorithm}] No successful trials collected; skipping plotting.",
                flush=True,
            )
            continue
        _render_single_mode_figures(result, out_path)

    return results


def _parse_algorithms(arg: Optional[str]) -> Optional[List[str]]:
    if arg is None:
        return None
    parsed: List[str] = []
    for chunk in arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parsed.extend(part for part in chunk.split() if part)
    return parsed or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--single",
        action="store_true",
        help="Generate single-layout visibility sweep figures.",
    )
    ap.add_argument(
        "--det",
        action="store_true",
        help="Use deterministic Krylov-based x0 construction (target dim V(x0)).",
    )
    ap.add_argument(
        "--vis-ntrials",
        type=int,
        default=200,
        help="Ensemble size per visible dimension (default: 200).",
    )
    ap.add_argument(
        "--vis-outdir",
        type=str,
        default=None,
        help="Output directory for sweep figures (defaults to cfg.save_dir/vis_sweep).",
    )
    ap.add_argument(
        "--algos",
        type=str,
        default=None,
        help="Comma-separated list of algorithms to include (default: all).",
    )

    args = ap.parse_args()
    if not args.single:
        ap.error("--single must be provided to generate the requested figures.")

    cfg = EstimatorConsistencyConfig()
    cfg.det = bool(args.det)

    out_dir = pathlib.Path(args.vis_outdir) if args.vis_outdir else None
    algos = _parse_algorithms(args.algos)

    run_single_mode_figures(
        cfg,
        algos=algos,
        ensemble_size=int(args.vis_ntrials),
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()