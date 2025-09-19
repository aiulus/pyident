from __future__ import annotations
from typing import Optional, Sequence, Dict, Tuple, Iterable

import numpy as np
import matplotlib.pyplot as plt


# ====================== sigma_min(·) contour on C-plane ===================

def plot_sigma_contour(
    alpha: np.ndarray,
    beta: np.ndarray,
    Sigma: np.ndarray,
    out_png: str,
    out_pdf: str,
    title: Optional[str] = r"$\sigma_{\min}$ of $[\lambda I - A,\ K]$",
    eigs: Optional[Sequence[complex]] = None,
) -> None:
    """
    Contour plot for σ_min over λ = alpha + i beta grid.

    Disclaimer:
    - NaNs are masked (white) so they don't produce misleading shading.
    - Color map encodes magnitude only; take care when comparing scenarios
      with different dynamic ranges (normalize if needed).
    """
    if Sigma.shape != (alpha.size, beta.size):
        # Many callers pass already-meshed arrays; accept both contracts.
        if Sigma.shape == (alpha.shape[0], beta.shape[0]):
            pass
        else:
            raise ValueError("Sigma must have shape (len(alpha), len(beta)).")

    Agrid, Bgrid = np.meshgrid(alpha, beta, indexing="ij")
    Z = np.ma.masked_invalid(Sigma)

    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    cs = ax.contourf(Agrid, Bgrid, Z, levels=30)
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label(r"$\sigma_{\min}$")

    ax.set_xlabel(r"$\Re(\lambda)$")
    ax.set_ylabel(r"$\Im(\lambda)$")
    if title:
        ax.set_title(title)

    # Optional: overlay eigenvalues
    if eigs:
        ax.scatter(np.real(eigs), np.imag(eigs), s=10, c="k", marker="x", label="eig(A)")
        ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


# ====================== PGF/TikZ export ===============================

def write_sigma_grid_csv(alpha: np.ndarray, beta: np.ndarray, Sigma: np.ndarray, csv_path: str) -> None:
    """
    Emit CSV with columns alpha,beta,sigma for PGFPlots \addplot3 table[…].
    """
    if Sigma.shape != (alpha.size, beta.size):
        if Sigma.shape == (alpha.shape[0], beta.shape[0]):
            pass
        else:
            raise ValueError("Sigma must have shape (len(alpha), len(beta)).")

    # Flatten in (i,j) order consistent with indexing="ij"
    with open(csv_path, "w") as f:
        f.write("alpha,beta,sigma\n")
        for i, a in enumerate(alpha):
            for j, b in enumerate(beta):
                s = Sigma[i, j]
                s_val = "" if np.isnan(s) else f"{float(s)}"
                f.write(f"{float(a)},{float(b)},{s_val}\n")


def emit_pgfplots_tex(csv_path: str, tex_path: str, xlabel: str = r"\Re(\lambda)", ylabel: str = r"\Im(\lambda)"):
    """
    Emit a minimal PGFPlots surface snippet that reads the CSV produced by write_sigma_grid_csv.
    """
    content = rf"""\begin{tikzpicture}
\begin{axis}[view={{0}}{{90}}, colorbar, xlabel={{{ {xlabel} }}}, ylabel={{{ {ylabel} }}}]
\addplot3[surf,shader=interp] table[x=alpha,y=beta,z=sigma,col sep=comma] {{{csv_path}}};
\end{axis}
\end{tikzpicture}
"""
    with open(tex_path, "w") as f:
        f.write(content)


# ====================== Rank bars & histograms ========================

def plot_rank_bars(labels: Sequence[str], ranks: Sequence[int], out_png: str, out_pdf: str, ylabel: str = "rank") -> None:
    """Simple bar plot for ranks (e.g., rank(K), dim(V), rank(Hankel))."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(range(len(labels)), ranks)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(ranks) + 1 if ranks else 1)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_histogram(values: np.ndarray, out_png: str, out_pdf: str, bins: int = 30, xlabel: str = "", ylabel: str = "count") -> None:
    """Histogram helper (for σ_min, margins, errors…)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.hist(values[~np.isnan(values)], bins=bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


# -------------------------- small internal helper ----------------------------
def _save_or_return(fig, axes, out_png: Optional[str], out_pdf: Optional[str]):
    if out_png or out_pdf:
        if out_png: fig.savefig(out_png, dpi=200, bbox_inches="tight")
        if out_pdf: fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig, axes

# ------------------------------ 1) Scree plot --------------------------------
def plot_scree(svals: Sequence[float],
               title: Optional[str] = None,
               out_png: Optional[str] = None, out_pdf: Optional[str] = None):
    """Scree (singular values descending), log-y."""
    svals = np.asarray(svals, dtype=float)
    fig, ax = plt.subplots()
    ax.semilogy(np.arange(1, len(svals)+1), np.sort(svals)[::-1], marker="o")
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular value")
    if title: ax.set_title(title)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    return _save_or_return(fig, ax, out_png, out_pdf)

# --------------------- 2) PE ladder: rank & condition ------------------------
def plot_pe_ladder(depths: Sequence[int],
                   ranks: Sequence[int],
                   conds: Optional[Sequence[float]] = None,
                   rowdim: Optional[int] = None,
                   out_png: Optional[str] = None, out_pdf: Optional[str] = None):
    """
    Plot Hankel rank vs depth (left y) and optionally condition number (right y, log).
    - depths: list of s
    - ranks:  rank(H_s)
    - conds:  cond(H_s) if available
    - rowdim: expected full row rank (e.g. s*m) to show dashed reference
    """
    d = np.asarray(depths)
    r = np.asarray(ranks)
    fig, ax1 = plt.subplots()
    ax1.plot(d, r, marker="o", label="rank(H_s)")
    ax1.set_xlabel("Hankel depth s")
    ax1.set_ylabel("rank(H_s)")
    ax1.grid(True, ls="--", alpha=0.4)
    if rowdim is not None:
        ax1.axhline(rowdim, color="k", ls="--", lw=1, alpha=0.6, label="full row rank")

    if conds is not None:
        c = np.asarray(conds, dtype=float)
        ax2 = ax1.twinx()
        ax2.semilogy(d, c, color="tab:red", marker="s", label="cond(H_s)")
        ax2.set_ylabel("cond(H_s)")
        # build a joint legend
        lines, labels = [], []
        for ax in (ax1, ax2):
            L = ax.get_legend_handles_labels()
            lines += L[0]; labels += L[1]
        ax1.legend(lines, labels, loc="best")
        axes = (ax1, ax2)
    else:
        ax1.legend(loc="best")
        axes = ax1
    return _save_or_return(fig, axes, out_png, out_pdf)

# ----------- 3) Scatter: projected error vs. metric (+binned bands) ----------
def scatter_error_vs_metric(
    x: Sequence[float],
    y_by_alg: Dict[str, Sequence[float]],
    xlabel: str = r"$\delta_{\mathrm{PBH}}$",
    ylabel: str = r"$\|P_V(\hat A-A)P_V\|_F$",
    bins: int = 20,
    quantiles: Tuple[float,float,float] = (0.25, 0.5, 0.75),
    out_png: Optional[str] = None, out_pdf: Optional[str] = None
):
    """
    Scatter points per algorithm; also overlays binned median & IQR bands.
    Returns (fig, ax) and the dict of binned summaries if not saving.
    """
    x = np.asarray(x, float)
    fig, ax = plt.subplots()
    summaries = {}

    # global bin edges
    finite_mask = np.isfinite(x)
    x_f = x[finite_mask]
    if len(x_f) == 0:  # guard
        x_edges = np.linspace(0.0, 1.0, bins+1)
    else:
        x_edges = np.linspace(x_f.min(), x_f.max(), bins+1)

    for alg, y in y_by_alg.items():
        y = np.asarray(y, float)
        m = finite_mask & np.isfinite(y)
        ax.scatter(x[m], y[m], s=12, alpha=0.35, label=f"{alg} (pts)")

        # binned stats
        q_lo, q_md, q_hi = [], [], []
        x_centers = []
        for i in range(bins):
            left, right = x_edges[i], x_edges[i+1]
            sel = (x >= left) & (x < right) & m
            if not np.any(sel):
                q_lo.append(np.nan); q_md.append(np.nan); q_hi.append(np.nan)
            else:
                vals = y[sel]
                q_lo.append(np.nanquantile(vals, quantiles[0]))
                q_md.append(np.nanquantile(vals, quantiles[1]))
                q_hi.append(np.nanquantile(vals, quantiles[2]))
            x_centers.append(0.5*(left+right))

        x_centers = np.array(x_centers)
        q_lo, q_md, q_hi = np.array(q_lo), np.array(q_md), np.array(q_hi)
        ax.plot(x_centers, q_md, lw=2, label=f"{alg} (median)")
        ax.fill_between(x_centers, q_lo, q_hi, alpha=0.15)

        summaries[alg] = {"x": x_centers, "q25": q_lo, "q50": q_md, "q75": q_hi}

    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(loc="best")
    return _save_or_return(fig, ax, out_png, out_pdf) or summaries

# ---------------------------- 4) Violin of errors ----------------------------
def violin_errors(groups: Dict[str, Sequence[float]],
                  ylabel: str = "error",
                  out_png: Optional[str] = None, out_pdf: Optional[str] = None):
    """Matplotlib violin plot for error distributions per group (e.g., estimator)."""
    labels = list(groups.keys())
    data = [np.asarray(groups[k], float) for k in labels]
    fig, ax = plt.subplots()
    parts = ax.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
    ax.set_xticks(np.arange(1, len(labels)+1))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel(ylabel)
    ax.grid(True, ls="--", alpha=0.3)
    return _save_or_return(fig, ax, out_png, out_pdf)

# ------------------------ 5) Subspace angles (degrees) -----------------------
def plot_subspace_angles(angles_rad: Sequence[float],
                         out_png: Optional[str] = None, out_pdf: Optional[str] = None):
    """Stem plot of principal angles (in degrees) between subspaces."""
    ang_deg = np.degrees(np.asarray(angles_rad, float))
    fig, ax = plt.subplots()
    markerline, stemlines, baseline = ax.stem(np.arange(1, len(ang_deg)+1), ang_deg, use_line_collection=True)
    plt.setp(stemlines, 'linewidth', 1.5)
    ax.set_xlabel("Angle index")
    ax.set_ylabel("Principal angle (deg)")
    ax.grid(True, ls="--", alpha=0.3)
    return _save_or_return(fig, ax, out_png, out_pdf)

# ----------------------- 6) Visible-dimension heatmap -----------------------
def heatmap_visible_dim(x_ticks: Sequence,
                        y_ticks: Sequence,
                        Z: np.ndarray,
                        xlabel: str, ylabel: str,
                        cbar_label: str = r"$\dim V$",
                        out_png: Optional[str] = None, out_pdf: Optional[str] = None):
    """
    Heatmap for visible-space dimension. Provide Z with shape (len(y_ticks), len(x_ticks)).
    """
    Z = np.asarray(Z)
    fig, ax = plt.subplots()
    im = ax.imshow(Z, origin="lower", aspect="auto",
                   extent=[0, len(x_ticks), 0, len(y_ticks)])
    cbar = fig.colorbar(im); cbar.set_label(cbar_label)
    ax.set_xticks(np.arange(len(x_ticks)) + 0.5); ax.set_xticklabels(x_ticks, rotation=30)
    ax.set_yticks(np.arange(len(y_ticks)) + 0.5); ax.set_yticklabels(y_ticks)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(False)
    return _save_or_return(fig, ax, out_png, out_pdf)

# ------------------- 7) Hankel full-row-rank “safety” view -------------------
def hankel_rank_condition(depths: Sequence[int], rowdim: Sequence[int],
                          ranks: Sequence[int], conds: Sequence[float],
                          out_png: Optional[str] = None, out_pdf: Optional[str] = None):
    """
    Two aligned panels: (top) rank vs expected rowdim; (bottom) log cond(H_s).
    """
    d = np.asarray(depths); rdim = np.asarray(rowdim); r = np.asarray(ranks); c = np.asarray(conds)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax1.plot(d, r, marker="o", label="rank(H_s)")
    ax1.plot(d, rdim, ls="--", color="k", label="s·m (rowdim)")
    ax1.set_ylabel("rank")
    ax1.legend(); ax1.grid(True, ls="--", alpha=0.3)

    ax2.semilogy(d, c, marker="s", color="tab:red")
    ax2.set_xlabel("Hankel depth s"); ax2.set_ylabel("cond(H_s)")
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    return _save_or_return(fig, (ax1, ax2), out_png, out_pdf)

# ---------------- 8) Eigenvalue overlay (true vs estimated) ------------------
def eig_overlay(true_eigs: Sequence[complex],
                est_eigs_by_alg: Dict[str, Sequence[complex]],
                unit_circle: bool = False,
                out_png: Optional[str] = None, out_pdf: Optional[str] = None):
    """
    Scatter of eigenvalues in the complex plane (discrete-time recommended).
    """
    fig, ax = plt.subplots()
    tr = np.asarray(true_eigs, dtype=complex)
    ax.scatter(tr.real, tr.imag, c="k", s=30, label="true")
    for name, vals in est_eigs_by_alg.items():
        v = np.asarray(vals, dtype=complex)
        ax.scatter(v.real, v.imag, s=22, alpha=0.7, label=name)
    if unit_circle:
        th = np.linspace(0, 2*np.pi, 400)
        ax.plot(np.cos(th), np.sin(th), "k:", lw=1)
    ax.set_xlabel("Re"); ax.set_ylabel("Im"); ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls="--", alpha=0.3); ax.legend()
    return _save_or_return(fig, ax, out_png, out_pdf)

# ------------------------ 9) Branch proportions (bar/pie) --------------------
def branch_proportions(counts: Dict[str, int],
                       kind: str = "bar",
                       out_png: Optional[str] = None, out_pdf: Optional[str] = None):
    """Visualize branch sizes (e.g., in_X0 vs not, in_R0 vs not)."""
    labels = list(counts.keys()); vals = [int(counts[k]) for k in labels]
    fig, ax = plt.subplots()
    if kind == "pie":
        ax.pie(vals, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
    else:
        ax.bar(labels, vals)
        ax.set_ylabel("count")
    return _save_or_return(fig, ax, out_png, out_pdf)

# -------------------- 10) Success-rate vs metric bin curve -------------------
def success_rate_curve(metric: Sequence[float],
                       error: Sequence[float],
                       thresh: float,
                       bins: int = 15,
                       xlabel: str = r"$\delta_{\mathrm{PBH}}$",
                       out_png: Optional[str] = None, out_pdf: Optional[str] = None):
    """
    For a scalar metric (e.g., PBH margin) and an error series, plot the fraction with error<=thresh per bin.
    """
    x = np.asarray(metric, float); y = np.asarray(error, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) == 0:
        x_edges = np.linspace(0.0, 1.0, bins+1)
    else:
        x_edges = np.linspace(x.min(), x.max(), bins+1)
    rates, centers = [], []
    for i in range(bins):
        L, R = x_edges[i], x_edges[i+1]
        sel = (x >= L) & (x < R)
        if not np.any(sel):
            rates.append(np.nan)
        else:
            rates.append(float(np.mean(y[sel] <= thresh)))
        centers.append(0.5*(L+R))
    fig, ax = plt.subplots()
    ax.plot(centers, rates, marker="o")
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel); ax.set_ylabel(f"success rate (err ≤ {thresh:g})")
    ax.grid(True, ls="--", alpha=0.3)
    return _save_or_return(fig, ax, out_png, out_pdf)

# ------------------ 11) Output overlay & residual autocorr -------------------
def model_output_overlay(t: Sequence[float],
                         y_true: np.ndarray,                 # (T,) or (T,p)
                         yhat_by_alg: Dict[str, np.ndarray], # each (T,) or (T,p)
                         dim: int = 0,
                         out_png: Optional[str] = None, out_pdf: Optional[str] = None):
    """Overlay 1D output trajectories (pick dimension with `dim` if multi-output)."""
    y_true = np.asarray(y_true)
    T = y_true.shape[0]
    if y_true.ndim == 2:
        y_true_1d = y_true[:, dim]
    else:
        y_true_1d = y_true
    fig, ax = plt.subplots()
    ax.plot(t, y_true_1d, lw=2, label="true")
    for name, y in yhat_by_alg.items():
        y = np.asarray(y)
        y1 = y[:, dim] if y.ndim == 2 else y
        ax.plot(t, y1, lw=1.5, alpha=0.8, label=name)
    ax.set_xlabel("t"); ax.set_ylabel("y")
    ax.grid(True, ls="--", alpha=0.3); ax.legend()
    return _save_or_return(fig, ax, out_png, out_pdf)

def residual_autocorr(e: Sequence[float], max_lag: int = 60,
                      out_png: Optional[str] = None, out_pdf: Optional[str] = None):
    """
    Simple residual ACF with +/- 1.96/sqrt(N) bounds (white-noise check).
    """
    e = np.asarray(e, float)
    e = e - np.nanmean(e)
    N = len(e)
    acf = [1.0]
    for k in range(1, max_lag+1):
        if k >= N:
            acf.append(np.nan)
            continue
        num = np.nansum(e[:-k]*e[k:])
        den = np.nansum(e*e)
        acf.append(num/den if den > 0 else np.nan)
    acf = np.array(acf)
    fig, ax = plt.subplots()
    ax.stem(range(len(acf)), acf, use_line_collection=True)
    ci = 1.96/np.sqrt(max(1, N))
    ax.axhspan(-ci, ci, color="k", alpha=0.1, lw=0)
    ax.set_xlabel("lag"); ax.set_ylabel("ACF")
    ax.grid(True, ls="--", alpha=0.3)
    return _save_or_return(fig, ax, out_png, out_pdf)


def annotate_ledger_footer(fig, ledger: dict | None):
    if not ledger:
        return
    env = ledger.get("env", {})
    tol = ledger.get("tolerances", {})
    approx = ledger.get("approximations", [])
    footer = []
    if env:
        footer.append(f"backend={env.get('accelerator')}, jax_x64={env.get('jax_x64')}")
    if tol:
        footer.append(f"svd_rtol={tol.get('svd_rtol')}, svd_atol={tol.get('svd_atol')}, pbh_cluster={tol.get('pbh_cluster_tol')}")
    if approx:
        kinds = ",".join(sorted({a.get('kind','') for a in approx}))
        footer.append(f"approximations={kinds}")
    txt = " | ".join(footer)
    if txt:
        fig.text(0.01, 0.01, txt, fontsize=8, ha="left", va="bottom")

def plot_with_band(x: np.ndarray, y_mean: np.ndarray, y_lo: np.ndarray, y_hi: np.ndarray,
                   xlabel: str, ylabel: str, title: str, out_png: str, out_pdf: str,
                   ledger: dict | None = None):
    fig, ax = plt.subplots(figsize=(5,3.2))
    ax.plot(x, y_mean, lw=2)
    ax.fill_between(x, y_lo, y_hi, alpha=0.25)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, alpha=0.3)
    annotate_ledger_footer(fig, ledger)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200); fig.savefig(out_pdf)
    plt.close(fig)

