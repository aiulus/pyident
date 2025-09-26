from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# If we need metrics for visible subspace
sys.path.append("/mnt/data")
try:
    import metrics as M
except Exception:
    M = None  # Heatmap-from-CSV modes won't need this; pred-viz will.

# ---------- CSV loading & pivoting ----------

def _detect_long(df: pd.DataFrame) -> bool:
    base_cols = {"n", "density"}
    val_cols = {"pct_any", "pct_pbh", "pct_mu"}
    return base_cols.issubset(df.columns) and bool(val_cols & set(df.columns))

def _load_wide(csv_path: str) -> tuple[np.ndarray, list, list]:
    """Wide: rows = density, cols = n; values = % unidentifiable."""
    df = pd.read_csv(csv_path)
    if "density" in df.columns:
        df = df.set_index("density")
    keep = []
    for c in df.columns:
        try:
            float(c); keep.append(c)
        except Exception:
            pass
    if keep:
        df = df[keep]
    df.index = pd.to_numeric(df.index, errors="coerce")
    df = df.sort_index()
    df.columns = pd.to_numeric(df.columns, errors="coerce")
    df = df.reindex(sorted(df.columns), axis=1)
    Mx = df.values
    y_vals = df.index.to_list()    # densities
    x_vals = df.columns.to_list()  # n
    return Mx, x_vals, y_vals

def _load_long(csv_path: str,
               value: str,
               index: str = "density",
               columns: str = "n") -> tuple[np.ndarray, list, list, pd.DataFrame]:
    """Long: pivot (index, columns) → values=value."""
    df = pd.read_csv(csv_path)
    if value not in df.columns:
        raise ValueError(f"Column '{value}' not in CSV. Available: {sorted(df.columns)}")
    for c in [index, columns, value]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    pv = df.pivot_table(index=index, columns=columns, values=value, aggfunc="mean")
    pv = pv.sort_index().sort_index(axis=1)
    Mx = pv.values
    y_vals = pv.index.to_list()    # densities
    x_vals = pv.columns.to_list()  # n
    return Mx, x_vals, y_vals, df

def load_matrix(csv_path: str,
                value: str = "pct_any",
                index: str | None = None,
                columns: str | None = None) -> tuple[np.ndarray, list, list, pd.DataFrame | None, bool]:
    head = pd.read_csv(csv_path, nrows=3)
    force_long = (index is not None) or (columns is not None)
    if force_long or _detect_long(head):
        idx = index or "density"
        cols = columns or "n"
        Mx, x_vals, y_vals, df_long = _load_long(csv_path, value=value, index=idx, columns=cols)
        return Mx, x_vals, y_vals, df_long, True
    else:
        Mx, x_vals, y_vals = _load_wide(csv_path)
        return Mx, x_vals, y_vals, None, False

# ---------- Matplotlib helpers ----------

def _compute_extent(x_vals: list, y_vals: list) -> tuple[list[float], float, float]:
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    dx = np.median(np.diff(x_arr)) if x_arr.size > 1 else 1.0
    dy = np.median(np.diff(y_arr)) if y_arr.size > 1 else 1.0
    xmin = float(x_arr.min() - 0.5 * dx)
    xmax = float(x_arr.max() + 0.5 * dx)
    ymin = float(y_arr.min() - 0.5 * dy)
    ymax = float(y_arr.max() + 0.5 * dy)
    return [xmin, xmax, ymin, ymax], dx, dy

def _apply_titles_ticks(x_vals, y_vals, df_long, title, title_extra,
                        xlabel, ylabel, is_density_on_y=True,
                        xtick_step=None, ytick_step=None):
    lines = [title]
    if title_extra:
        lines.append(title_extra)
    else:
        if df_long is not None and "N" in df_long.columns:
            Ns = sorted(set(df_long["N"].dropna().astype(int)))
            if len(Ns) == 1:
                lines.append(f"N={Ns[0]}/cell")
    plt.title("\n".join(lines))

    x_min, x_max = float(min(x_vals)), float(max(x_vals))
    if xtick_step is None:
        span = x_max - x_min
        approx = span / 10 if span > 0 else 1.0
        if is_density_on_y is False:
            xtick_step = max(0.05, round(approx / 0.05) * 0.05)
        else:
            xtick_step = max(1.0, round(approx))
    x_ticks = np.arange(x_min, x_max + 1e-9, xtick_step)
    x_ticks = np.round(x_ticks, 3)
    plt.xticks(x_ticks)

    y_min, y_max = float(min(y_vals)), float(max(y_vals))
    if ytick_step is None:
        span = y_max - y_min
        approx = span / 10 if span > 0 else 1.0
        if is_density_on_y:
            ytick_step = max(0.05, round(approx / 0.05) * 0.05)
        else:
            ytick_step = max(1.0, round(approx))
    y_ticks = np.arange(y_min, y_max + 1e-9, ytick_step)
    y_ticks = np.round(y_ticks, 3)
    plt.yticks(y_ticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def _matrix_heatmap(M: np.ndarray, title: str, out: Path,
                    cmap: str = "coolwarm", dpi: int = 150, center: bool = True):
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.6, 4.8))
    if center:
        vmax = float(np.max(np.abs(M))) if M.size else 1.0
        vmin = -vmax
    else:
        vmin, vmax = None, None
    im = plt.imshow(M, origin="upper", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=dpi)
    plt.close()

# ---------- Plotters (heatmaps from csv) ----------

def plot_heatmap_from_csv(csv: str,
                          out: str,
                          value: str = "pct_any",
                          index: str | None = None,
                          columns: str | None = None,
                          title: str = "Unidentifiable fraction (union criterion)",
                          title_extra: str = "",
                          xlabel: str | None = None,
                          ylabel: str | None = None,
                          cmap: str = "viridis",
                          dpi: int = 150,
                          vmin: float | None = None,
                          vmax: float | None = None,
                          xtick_step: float | None = None,
                          ytick_step: float | None = None) -> None:
    Mx, x_vals, y_vals, df_long, _is_long = load_matrix(csv, value=value, index=index, columns=columns)
    extent, dx, dy = _compute_extent(x_vals, y_vals)
    if xlabel is None: xlabel = "State dimension n"
    if ylabel is None: ylabel = "Density p"
    plt.figure(figsize=(10.0, 6.2))
    im = plt.imshow(Mx, origin="lower", aspect="auto", extent=extent,
                    cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im); cbar.set_label("% unidentifiable (PBH==0 OR μ_min==0)")
    _apply_titles_ticks(
        x_vals, y_vals, df_long, title, title_extra,
        xlabel, ylabel, is_density_on_y=True,
        xtick_step=xtick_step, ytick_step=ytick_step
    )
    outp = Path(out); outp.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(outp, dpi=dpi); plt.close()
    print(f"[offline_plots] saved heatmap → {outp}")

def plot_heatmap_cbk_from_csv(csv: str,
                              out: str,
                              value: str = "pct_any",
                              index: str | None = None,
                              columns: str | None = None,
                              title: str = "Unidentifiable fraction (union criterion)",
                              title_extra: str = "",
                              xlabel: str | None = None,
                              ylabel: str | None = None,
                              cmap: str = "viridis",
                              dpi: int = 150,
                              vmin: float | None = None,
                              vmax: float | None = None,
                              xtick_step: float | None = None,
                              ytick_step: float | None = None) -> None:
    Mx, x_vals, y_vals, df_long, _is_long = load_matrix(csv, value=value, index=index, columns=columns)
    y_vals = np.asarray(y_vals, dtype=float)   # densities
    x_vals = np.asarray(x_vals, dtype=float)   # n
    x_cbk = 1.0 - y_vals         # sparsity
    y_cbk = x_vals               # n
    ord_x = np.argsort(x_cbk)
    ord_y = np.argsort(y_cbk)
    M_cbk = Mx.T
    M_cbk = M_cbk[ord_y, :][:, ord_x]
    x_cbk = x_cbk[ord_x]
    y_cbk = y_cbk[ord_y]
    extent, _, _ = _compute_extent(x_cbk, y_cbk)
    if xlabel is None: xlabel = "Sparsity (1 − density p)"
    if ylabel is None: ylabel = "State dimension n"
    plt.figure(figsize=(10.0, 6.2))
    im = plt.imshow(M_cbk, origin="lower", aspect="auto", extent=extent,
                    cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im); cbar.set_label("% unidentifiable (PBH==0 OR μ_min==0)")
    _apply_titles_ticks(
        x_cbk, y_cbk, df_long, title, title_extra,
        xlabel, ylabel, is_density_on_y=False,
        xtick_step=xtick_step, ytick_step=ytick_step
    )
    outp = Path(out); outp.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(outp, dpi=dpi); plt.close()
    print(f"[offline_plots] saved CBK heatmap → {outp}")

# ---------- Visible subspace & (V, W) transform helpers ----------

def _visible_basis(A: np.ndarray, B: np.ndarray, x0: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Orthonormal basis P for V = span{ K(A, [x0 B]) } using unified_generator + SVD rank cut.
    """
    if M is None or not hasattr(M, "unified_generator"):
        raise RuntimeError("metrics.unified_generator not available; cannot compute visible subspace.")
    K = M.unified_generator(A, B, x0, mode="unrestricted")
    if K.size == 0:
        return np.zeros((A.shape[0], 0))
    U, s, _ = np.linalg.svd(K, full_matrices=False)
    r = int((s > tol).sum())
    return U[:, :r] if r > 0 else np.zeros((A.shape[0], 0))

def _input_basis_W(B_true: np.ndarray) -> np.ndarray:
    """
    Orthonormal basis W in input space R^m from right singular vectors of the TRUE B.
    Returns W (m x m).  (If B is rank-deficient that's fine; basis still spans R^m.)
    """
    # B_true is n x m; Vt has shape (m x m)
    _, _, Vt = np.linalg.svd(B_true, full_matrices=True)
    return Vt.T

# ---------- NEW: visualize random predictions + (V, W) expressions ----------

def visualize_random_predictions(pred_dir: str,
                                 outdir: str,
                                 samples: int = 3,
                                 seed: int = 7,
                                 tol: float = 1e-12,
                                 cmap_raw: str = "coolwarm",
                                 cmap_vw: str = "coolwarm",
                                 dpi: int = 150):
    """
    (1) Load NPZ predictions from `pred_dir` (expects trial_*.npz with Ahat, Bhat, A, B, x0).
    (2) Choose `samples` trials uniformly at random (seeded).
    (3) Plot raw Â and B̂ heat maps.
    (4) Compute V via visible hull K(A,[x0 B]) and W via right-singular-vectors of TRUE B.
    (5) Plot A|_V = Pᵀ Â P and B_{V,W} = Pᵀ B̂ W heat maps.
    (6) Also plot TRUE A, B and their (V,W) expressions (per chosen trial/x0).
    """
    pred_path = Path(pred_dir)
    out_root = Path(outdir)
    out_raw = out_root / "raw"
    out_vw  = out_root / "vw"
    out_true = out_root / "true"
    out_raw.mkdir(parents=True, exist_ok=True)
    out_vw.mkdir(parents=True, exist_ok=True)
    out_true.mkdir(parents=True, exist_ok=True)

    files = sorted(pred_path.glob("trial_*.npz"))
    if len(files) == 0:
        raise FileNotFoundError(f"No prediction files found in {pred_path}")

    rng = np.random.default_rng(seed)
    pick_idx = rng.choice(len(files), size=min(samples, len(files)), replace=False)
    picked = [files[i] for i in pick_idx]

    # We'll read TRUE A,B once (they should be identical across trials)
    A_true = B_true = None
    for f in files:
        with np.load(f) as z:
            if "A" in z and "B" in z:
                A_true = z["A"]; B_true = z["B"]
                break
    if A_true is None or B_true is None:
        # Some runs might have only Ad,Bd; still plot estimates vs (unknown) continuous-time
        with np.load(files[0]) as z:
            A_true = z.get("A", None); B_true = z.get("B", None))

    for f in picked:
        with np.load(f) as z:
            Ahat = z["Ahat"]; Bhat = z["Bhat"]
            x0   = z["x0"]
            if A_true is None or B_true is None:
                # Fall back to DT true if needed
                A_true = z.get("Ad", Ahat*0)
                B_true = z.get("Bd", Bhat*0)
            n, m = Ahat.shape[0], Bhat.shape[1]

            trial_id = f.stem  # e.g., "trial_0007"

            # (3) raw heatmaps
            _matrix_heatmap(Ahat, title=f"Â — {trial_id}", out=out_raw / f"{trial_id}_Ahat.png",
                            cmap=cmap_raw, dpi=dpi)
            _matrix_heatmap(Bhat, title=f"B̂ — {trial_id}", out=out_raw / f"{trial_id}_Bhat.png",
                            cmap=cmap_raw, dpi=dpi)

            # (4) (V, W) expressions
            P = _visible_basis(A_true, B_true, x0, tol=tol)  # n x k
            W = _input_basis_W(B_true)                       # m x m
            k = P.shape[1]

            # Blocks (top-left = restriction to V)
            Ahat_V = P.T @ Ahat @ P             # k x k
            Bhat_VW = P.T @ Bhat @ W            # k x m
            Atru_V  = P.T @ A_true @ P          # k x k
            Btru_VW = P.T @ B_true @ W          # k x m

            _matrix_heatmap(Ahat_V, title=f"Â on V (k={k}) — {trial_id}", out=out_vw / f"{trial_id}_Ahat_onV.png",
                            cmap=cmap_vw, dpi=dpi)
            _matrix_heatmap(Bhat_VW, title=f"B̂ on (V,W) (k×m={k}×{m}) — {trial_id}", out=out_vw / f"{trial_id}_Bhat_onVW.png",
                            cmap=cmap_vw, dpi=dpi)

            # (6) true matrices too (per-trial x0 → P changes, so keep per-trial outputs)
            _matrix_heatmap(A_true, title=f"A (true) — {trial_id}", out=out_true / f"{trial_id}_A_true.png",
                            cmap=cmap_raw, dpi=dpi)
            _matrix_heatmap(B_true, title=f"B (true) — {trial_id}", out=out_true / f"{trial_id}_B_true.png",
                            cmap=cmap_raw, dpi=dpi)
            _matrix_heatmap(Atru_V, title=f"A|_V (true, k={k}) — {trial_id}", out=out_true / f"{trial_id}_A_true_onV.png",
                            cmap=cmap_vw, dpi=dpi)
            _matrix_heatmap(Btru_VW, title=f"B on (V,W) (true, k×m={k}×{m}) — {trial_id}", out=out_true / f"{trial_id}_B_true_onVW.png",
                            cmap=cmap_vw, dpi=dpi)

    print(f"[offline_plots] saved prediction visualizations under: {out_root}")

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Offline plotter for unidentifiability heatmaps and prediction visualizations.")
    # Heatmap-from-CSV options
    ap.add_argument("--csv", default=None, help="Path to summary_wide_*.csv or summary_long.csv")
    ap.add_argument("--out", default=None, help="Output PNG path (for CSV heatmap modes)")
    ap.add_argument("--cbk", action="store_true", help="Use CBK layout (x = sparsity = 1 - density, y = n)")
    ap.add_argument("--value", default="pct_any", help="Value (long CSV): pct_any|pct_pbh|pct_mu")
    ap.add_argument("--index", default=None, help="Pivot index (long CSV). Default: density")
    ap.add_argument("--columns", default=None, help="Pivot columns (long CSV). Default: n")
    ap.add_argument("--title", default="Unidentifiable fraction (union criterion)", help="Main title")
    ap.add_argument("--title-extra", default="", help="Extra title line (e.g., 'N=200/cell, tol=1e-12')")
    ap.add_argument("--xlabel", default=None, help="X-axis label override")
    ap.add_argument("--ylabel", default=None, help="Y-axis label override")
    ap.add_argument("--cmap", default="viridis", help="Matplotlib colormap")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--xtick-step", type=float, default=None, help="Tick step on x-axis")
    ap.add_argument("--ytick-step", type=float, default=None, help="Tick step on y-axis")

    # NEW: predictions visualization options
    ap.add_argument("--pred-dir", default=None, help="Directory containing trial_*.npz predictions (earlier run).")
    ap.add_argument("--pred-outdir", default=None, help="Output directory for prediction heatmaps (many PNGs).")
    ap.add_argument("--pred-samples", type=int, default=3, help="How many random trials to visualize.")
    ap.add_argument("--pred-seed", type=int, default=7, help="RNG seed for picking trials.")
    ap.add_argument("--pred-tol", type=float, default=1e-12, help="Tolerance for visible-subspace rank cut.")
    ap.add_argument("--pred-cmap-raw", default="coolwarm", help="Colormap for raw Â, B̂, A, B heatmaps.")
    ap.add_argument("--pred-cmap-vw", default="coolwarm", help="Colormap for (V,W) heatmaps.")

    args = ap.parse_args()

    # 1) If we were given a predictions directory, run the new visualization
    if args.pred_dir is not None:
        if M is None:
            raise RuntimeError("This mode needs the 'metrics' module (for visible subspace).")
        outdir = args.pred_outdir or (Path(args.pred_dir).parent / "pred_viz")
        visualize_random_predictions(
            pred_dir=args.pred_dir,
            outdir=str(outdir),
            samples=args.pred_samples,
            seed=args.pred_seed,
            tol=args.pred_tol,
            cmap_raw=args.pred_cmap_raw,
            cmap_vw=args.pred_cmap_vw,
            dpi=args.dpi,
        )

    # 2) CSV heatmap modes (optional in the same run)
    if args.csv is not None and args.out is not None:
        if args.cbk:
            plot_heatmap_cbk_from_csv(
                csv=args.csv,
                out=args.out,
                value=args.value,
                index=args.index,
                columns=args.columns,
                title=args.title,
                title_extra=args.title_extra,
                xlabel=args.xlabel,
                ylabel=args.ylabel,
                cmap=args.cmap,
                dpi=args.dpi,
                vmin=args.vmin,
                vmax=args.vmax,
                xtick_step=args.xtick_step,
                ytick_step=args.ytick_step,
            )
        else:
            plot_heatmap_from_csv(
                csv=args.csv,
                out=args.out,
                value=args.value,
                index=args.index,
                columns=args.columns,
                title=args.title,
                title_extra=args.title_extra,
                xlabel=args.xlabel,
                ylabel=args.ylabel,
                cmap=args.cmap,
                dpi=args.dpi,
                vmin=args.vmin,
                vmax=args.vmax,
                xtick_step=args.xtick_step,
                ytick_step=args.ytick_step,
            )

if __name__ == "__main__":
    main()
