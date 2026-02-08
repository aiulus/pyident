"""Shared matplotlib boxplot styling helpers for sim_unctrb_* scripts."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def set_default_mpl_style() -> None:
    """Lightweight, publication-friendly defaults."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
        }
    )


def _finite_or_empty(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def nice_boxplot(
    ax,
    data,
    labels,
    *,
    colors=None,
    colorize: bool = True,
    title=None,
    ylabel=None,
    whis=(5, 95),
    showfliers=False,
    notch=False,
    widths=0.6,
    jitter_points=True,
    jitter_alpha=0.25,
    point_size=10,
    annotate_n=True,
    median_linewidth: float = 2.0,
    yscale=None,
):
    """A clean default boxplot with optional jitter overlay and n-annotation."""
    clean = [_finite_or_empty(d) for d in data]
    clean = [c if c.size > 0 else np.array([np.nan]) for c in clean]

    bp = ax.boxplot(
        clean,
        labels=labels,
        patch_artist=True,
        widths=widths,
        whis=whis,
        showfliers=showfliers,
        notch=notch,
        medianprops=dict(linewidth=median_linewidth),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.2),
        flierprops=dict(markersize=3, alpha=0.35),
    )

    if colorize:
        if colors is None:
            colors = [f"C{i}" for i in range(len(labels))]
        for box, c in zip(bp["boxes"], colors):
            box.set_facecolor(c)
            box.set_alpha(0.25)
            box.set_edgecolor(c)
    else:
        for box in bp["boxes"]:
            box.set_facecolor("none")
            box.set_alpha(1.0)
            box.set_edgecolor("0.25")

    for k in ("whiskers", "caps", "medians"):
        for line in bp[k]:
            line.set_color("0.25")

    ax.grid(True, axis="y")
    ax.set_axisbelow(True)

    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)

    if yscale is not None:
        ax.set_yscale(yscale)

    if jitter_points:
        rng = np.random.default_rng(0)
        for i, vals in enumerate(clean, start=1):
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            x = i + rng.uniform(-0.12, 0.12, size=vals.size)
            ax.scatter(x, vals, s=point_size, alpha=jitter_alpha, linewidths=0)

    if annotate_n:
        ymin, ymax = ax.get_ylim()
        ytext = ymin + 0.02 * (ymax - ymin)
        for i, vals in enumerate(clean, start=1):
            n = int(np.isfinite(vals).sum())
            ax.text(i, ytext, f"n={n}", ha="center", va="bottom", fontsize=9, alpha=0.8)

    return bp


def nice_grouped_boxplot(
    ax,
    data_a,
    data_b,
    labels,
    *,
    label_a="standard",
    label_b="P-basis",
    color_a="#4C78A8",
    color_b="#F58518",
    whis=(5, 95),
    showfliers=False,
    widths=0.55,
    gap=2.4,
    inner=0.8,
    title=None,
    ylabel=None,
    jitter_points=True,
    yscale=None,
):
    """Side-by-side boxplots per algorithm, consistent legend + styling."""
    A = [_finite_or_empty(d) for d in data_a]
    B = [_finite_or_empty(d) for d in data_b]
    A = [c if c.size > 0 else np.array([np.nan]) for c in A]
    B = [c if c.size > 0 else np.array([np.nan]) for c in B]

    positions, all_data, tickpos = [], [], []
    for i in range(len(labels)):
        base = i * gap
        positions += [base, base + inner]
        all_data += [A[i], B[i]]
        tickpos.append(base + inner / 2)

    bp = ax.boxplot(
        all_data,
        positions=positions,
        widths=widths,
        patch_artist=True,
        whis=whis,
        showfliers=showfliers,
        medianprops=dict(linewidth=2.0),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.2),
        flierprops=dict(markersize=3, alpha=0.35),
    )

    colors = [color_a, color_b]
    for i, box in enumerate(bp["boxes"]):
        c = colors[i % 2]
        box.set_facecolor(c)
        box.set_alpha(0.25)
        box.set_edgecolor(c)

    for k in ("whiskers", "caps", "medians"):
        for line in bp[k]:
            line.set_color("0.25")

    ax.set_xticks(tickpos)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)

    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if yscale is not None:
        ax.set_yscale(yscale)

    if jitter_points:
        rng = np.random.default_rng(0)
        for j, vals in enumerate(all_data):
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            x0 = positions[j]
            x = x0 + rng.uniform(-0.10, 0.10, size=vals.size)
            ax.scatter(x, vals, s=10, alpha=0.22, linewidths=0)

    ax.legend(
        handles=[
            Patch(facecolor=color_a, edgecolor=color_a, alpha=0.25, label=label_a),
            Patch(facecolor=color_b, edgecolor=color_b, alpha=0.25, label=label_b),
        ],
        frameon=False,
        loc="upper right",
    )
    return bp


def nice_violinplot(
    ax,
    data,
    labels,
    *,
    colors=None,
    colorize: bool = True,
    title=None,
    ylabel=None,
    widths=0.8,
    showmeans=False,
    showmedians=True,
    showextrema=False,
    jitter_points=True,
    jitter_alpha=0.25,
    point_size=10,
    annotate_n=True,
    median_linewidth: float = 2.0,
    yscale=None,
):
    """A clean default violin plot with optional jitter overlay and n-annotation."""
    clean = [_finite_or_empty(d) for d in data]
    clean = [c if c.size > 0 else np.array([np.nan]) for c in clean]

    vp = ax.violinplot(
        clean,
        showmeans=showmeans,
        showmedians=showmedians,
        showextrema=showextrema,
        widths=widths,
    )

    if colorize:
        if colors is None:
            colors = [f"C{i}" for i in range(len(labels))]
        for body, c in zip(vp["bodies"], colors):
            body.set_facecolor(c)
            body.set_alpha(0.25)
            body.set_edgecolor(c)
            body.set_linewidth(1.1)
    else:
        for body in vp["bodies"]:
            body.set_facecolor("none")
            body.set_alpha(1.0)
            body.set_edgecolor("0.25")
            body.set_linewidth(1.1)

    for key in ("cmeans", "cmedians", "cmaxes", "cmins", "cbars"):
        if key in vp:
            vp[key].set_color("0.25")
            if key == "cmedians":
                vp[key].set_linewidth(median_linewidth)
            else:
                vp[key].set_linewidth(1.2)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)

    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)

    if yscale is not None:
        ax.set_yscale(yscale)

    if jitter_points:
        rng = np.random.default_rng(0)
        for i, vals in enumerate(clean, start=1):
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            x = i + rng.uniform(-0.12, 0.12, size=vals.size)
            ax.scatter(x, vals, s=point_size, alpha=jitter_alpha, linewidths=0)

    if annotate_n:
        ymin, ymax = ax.get_ylim()
        ytext = ymin + 0.02 * (ymax - ymin)
        for i, vals in enumerate(clean, start=1):
            n = int(np.isfinite(vals).sum())
            ax.text(i, ytext, f"n={n}", ha="center", va="bottom", fontsize=9, alpha=0.8)

    return vp
