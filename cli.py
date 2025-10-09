from __future__ import annotations
import argparse
import json
import numpy as np
from .config import ExpConfig, SolverOpts

import os, csv
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ------------------------
# small parsing utilities
# ------------------------
def _parse_str_list(s: str) -> list[str]:
    # Accepts "a,b,c" or "a b c"
    return [x.strip() for x in s.replace(",", " ").split() if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    """
    Accepts:
      - comma list: "1,2,3"
      - range: "0:10" (0..9), or "0:10:2" (step=2)
    """
    s = s.strip()
    if ":" in s:
        parts = [int(x) for x in s.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = 1
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError(f"Bad range '{s}'. Use 'start:stop[:step]'.")
        return list(range(start, stop, step))
    return [int(x) for x in s.split(",") if x.strip()]

def _parse_float_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _add_common_single_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--T", type=int, default=400)
    p.add_argument("--dt", type=float, default=0.05)

    p.add_argument("--ensemble", type=str, default="ginibre",
                   choices=["ginibre", "sparse", "stable", "binary"])
    p.add_argument("--p_density", type=float, default=0.8,
                   help="Nonzero fraction for A (and B if sparse_which='both' and p_density_B unset).")
    p.add_argument("--sparse-which", "--sparse_which",
               dest="sparse_which",
               type=str, default="both",
               choices=["A", "B", "both"])
    p.add_argument("--p_density_B", type=float, default=None,
                   help="Optional B density (if sparse_which includes B).")

    p.add_argument("--x0_mode", type=str, default="gaussian",
                   choices=["gaussian", "rademacher", "ones", "zero"])

    p.add_argument("--signal", type=str, default="prbs",
                   choices=["prbs", "multisine"])
    p.add_argument("--sigPE", type=int, default=12,
                   help="Targeted richness for signal design.")

    # Pointwise admissibility: restrict to span(W) with q directions
    p.add_argument("--U_restr_dim", type=int, default=None,
                   help="If set, build W = I_m[:, :q] and project inputs onto span(W).")

    # Nonlocal admissibility: moment-PE order r for analysis (not for generation)
    p.add_argument("--PE_r", type=int, default=None,
                   help="If set, analyze identifiability with moment-PE order r.")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--algs", type=str, default="dmdc,moesp",
                   help="Comma-separated list, e.g. 'dmdc,moesp,dmdc_tls,dmdc_iv'.")
    
    # JAX toggles
    p.add_argument("--use-jax", action="store_true",
                   help="Use JAX accelerator for simulation/metrics/LS.")
    p.add_argument("--jax-x64", action="store_true",
                   help="Enable float64 in JAX (recommended).")
    p.add_argument("--jax-platform", type=str, choices=["cpu","metal","auto"],
                default="auto",
                help="Force JAX backend (cpu or metal). 'auto' keeps JAX defaults.")
    
    # ---- Gradient DMDc options ----
    p.add_argument("--gd-steps", type=int, default=None,
                   help="Number of GD iterations (override SolverOpts default).")
    p.add_argument("--gd-lr", type=float, default=None,
                   help="Learning rate; if omitted, set from Lipschitz estimate.")
    p.add_argument("--gd-opt", type=str, choices=["adam", "sgd"], default=None,
                   help="Optimizer for GD-DMDc.")
    p.add_argument("--gd-ridge", type=float, default=None,
                   help="Ridge λ (if omitted, auto-scaled via rcond).")
    p.add_argument("--gd-project", type=str, choices=["ct", "dt"], default=None,
                   help="Project A each step to a stable set: 'ct' (shift-left) or 'dt' (radius scale).")
    p.add_argument("--gd-proj-params", type=str, default=None,
                   help='JSON dict for projection params, e.g. {"ct_margin":1e-3,"dt_rho":0.98}.')
    p.add_argument("--rcond", type=float, default=1e-10,
                   help="LS pseudoinverse rcond (also sets auto ridge scale).")
    
    # Output / light mode
    p.add_argument("--json_out", type=str, default=None,
                   help="If set, write the single-run JSON to this path instead of printing.")
    p.add_argument("--light", action="store_true",
                   help="Light result mode: omit heavy arrays to save disk.")
    p.add_argument("--out-json", "--out_json",
                    dest="out_json",
                    type=str, default=None,
                    help="Write the result JSON to this path (directories auto-created).")




def parse_args():
    p = argparse.ArgumentParser(description="pyident experiments")
    sub = p.add_subparsers(dest="cmd")

    # -------- single --------
    ps = sub.add_parser("single", help="Run a single experiment (default).")
    _add_common_single_args(ps)
    ps.add_argument("--outdir", type=str, default=None,
                help="If set, write outputs under this directory.")
    ps.add_argument("--prefix", type=str, default="single",
                    help="Filename prefix for outputs (e.g., single_…)")

    # optional plotting toggles (no-ops unless you wire them up)
    ps.add_argument("--plots", action="store_true",
                    help="If set, generate basic plots for the single run.")
    ps.add_argument("--plot-format", type=str, choices=["png","pdf","both"], default="png",
                    help="Plot format(s) if --plots is used.")


    # -------- sweep-underactuation --------
    pu = sub.add_parser("sweep-underactuation", help="Sweep over m (underactuation study).")
    pu.add_argument("--n", type=int, default=10)
    pu.add_argument("--m-values", type=str, default="1,2,3,4,5,6,8,10",
                    help="List or range, e.g. '1,2,3' or '1:11'.")
    pu.add_argument("--T", type=int, default=400)
    pu.add_argument("--dt", type=float, default=0.05)
    pu.add_argument("--ensemble", type=str, default="ginibre",
                    choices=["ginibre", "sparse", "stable", "binary"])
    pu.add_argument("--signal", type=str, default="prbs", choices=["prbs", "multisine"])
    pu.add_argument("--sigPE", type=int, default=12)
    pu.add_argument("--seeds", type=str, default="0:50", help="e.g. '0:50' or '0,1,2'")
    pu.add_argument("--algs", type=str, default="dmdc,moesp,dmdc_tls,dmdc_iv")
    pu.add_argument("--x0-mode", type=str, default=None,
                choices=["gaussian","rademacher","ones","zero"],
                help="Initial state policy (use 'zero' to remove homogeneous contribution).")

    pu.add_argument("--out-csv", type=str, default="results_underactuation.csv")

    # Optional admissibility for the whole sweep
    pu.add_argument("--U_restr_dim", type=int, default=None)
    pu.add_argument("--PE_r", type=int, default=None)

    pu.add_argument("--use-jax", action="store_true")
    pu.add_argument("--jax-x64", action="store_true")
    pu.add_argument("--jax-platform", choices=["cpu","metal","auto"], default="auto")

    pu.add_argument("--sparse-which", "--sparse_which", dest="sparse_which",
                type=str, default="B", choices=["A","B","both"])
    pu.add_argument("--p_density_B", type=float, default=0.1,
                    help="If ensemble=sparse and sparse_which includes B, set B's density.")



    # -------- sweep-sparsity --------
    psr = sub.add_parser("sweep-sparsity", help="Sweep over sparsity levels.")
    psr.add_argument("--n", type=int, default=10)
    psr.add_argument("--m", type=int, default=3)
    psr.add_argument("--p-values", type=str, default="1.0,0.9,0.8,0.6,0.4,0.2",
                     help="Comma list of densities (0..1).")
    psr.add_argument("--T", type=int, default=400)
    psr.add_argument("--dt", type=float, default=0.05)
    psr.add_argument("--sparse-which", "--sparse_which",
                     dest="sparse_which",
                     type=str, default="both",
                     choices=["A", "B", "both"])
    psr.add_argument("--signal", type=str, default="prbs", choices=["prbs", "multisine"])
    psr.add_argument("--sigPE", type=int, default=12)
    psr.add_argument("--seeds", type=str, default="0:50", help="e.g. '0:50' or '0,1,2'")
    psr.add_argument("--algs", type=str, default="dmdc,moesp,dmdc_tls,dmdc_iv")
    psr.add_argument("--out-csv", type=str, default="results_sparsity.csv")

    psr.add_argument("--use-jax", action="store_true")
    psr.add_argument("--jax-x64", action="store_true")
    psr.add_argument("--jax-platform", choices=["cpu","metal","auto"], default="auto")

    psr.add_argument("--x0-mode", type=str, default=None,
                 choices=["gaussian","rademacher","ones","zero"])
    psr.add_argument("--U_restr_dim", type=int, default=None,
                    help="If set, restrict inputs to span of first q basis vectors.")

        # -------- underactuation-plus --------
    pup = sub.add_parser("underactuation-plus",
                         help="Underactuation sweep with complementary metrics (α,β,η0, growth, CIs).")
    pup.add_argument("--n", type=int, required=True)
    pup.add_argument("--m", type=int, nargs="+", required=True, dest="m_values")
    pup.add_argument("--T", type=int, default=200)
    pup.add_argument("--dt", type=float, default=0.05)
    pup.add_argument("--trials", type=int, default=100)
    pup.add_argument("--sigPE", type=int, default=31)
    pup.add_argument("--out", type=str, required=True)
    # optional JAX toggles (kept symmetric with other cmds)
    pup.add_argument("--use-jax", action="store_true")
    pup.add_argument("--jax-x64", action="store_true")
    pup.add_argument("--jax-platform", choices=["cpu","metal","auto"], default="auto")

    # -------- sparsity-plus --------
    psp = sub.add_parser("sparsity-plus",
                         help="Sparsity sweep with complementary metrics (α,β,η0, growth, CIs).")
    psp.add_argument("--n", type=int, required=True)
    psp.add_argument("--m", type=int, required=True)
    psp.add_argument("--p", type=float, nargs="+", required=True, dest="p_values",
                     help="List of densities (0..1), e.g. --p 0.2 0.4 0.8")
    psp.add_argument("--T", type=int, default=200)
    psp.add_argument("--dt", type=float, default=0.05)
    psp.add_argument("--trials", type=int, default=100)
    psp.add_argument("--sigPE", type=int, default=31)
    psp.add_argument("--which", type=str, default="both", dest="sparse_which",
                     choices=["A","B","both"])
    psp.add_argument("--out", type=str, required=True)
    # optional JAX toggles
    psp.add_argument("--use-jax", action="store_true")
    psp.add_argument("--jax-x64", action="store_true")
    psp.add_argument("--jax-platform", choices=["cpu","metal","auto"], default="auto")

    # top-level (common) options
    for sp in (pu, psr):
        sp.add_argument("--plots", action="store_true",
                        help="If set, generate quick-look plots alongside CSVs.")
        sp.add_argument("--plot-dir", type=str, default=None,
                        help="Directory to save plots (defaults to outdir).")
        sp.add_argument("--plot-format", type=str, default="png",
                        choices=("png","pdf","both"),
                        help="Image format for plots (default: png).")


    # If no subcommand, fall back to single
    p.set_defaults(cmd="single")
    return p.parse_args()


def _build_U_restr(m: int, q: int | None) -> np.ndarray | None:
    if q is None:
        return None
    if q < 1 or q > m:
        raise ValueError(f"--U_restr_dim must be in [1, m]={m}, got {q}.")
    return np.eye(m)[:, :q]


def main():
    a = parse_args()
    import os
    if getattr(a, "use_jax", False) and getattr(a, "jax_platform", None) in ("cpu","metal"):
        os.environ["JAX_PLATFORM_NAME"] = a.jax_platform

    if a.cmd == "single":
        # Build U_restr if requested (canonical subspace spanned by first q basis vectors)
        U_restr = _build_U_restr(a.m, a.U_restr_dim)

        cfg = ExpConfig(
            n=a.n,
            m=a.m,
            T=a.T,
            dt=a.dt,
            ensemble=a.ensemble,
            p_density=a.p_density,
            sparse_which=a.sparse_which,
            p_density_A=None,
            p_density_B=a.p_density_B,
            x0_mode=a.x0_mode,
            signal=a.signal,
            sigPE=a.sigPE,
            U_restr=U_restr,
            PE_r=a.PE_r,
            algs=tuple(s.strip() for s in a.algs.split(",") if s.strip()),
            light=True,
        )
        sopts = SolverOpts(
            rcond=a.rcond,
            gd_steps=(a.gd_steps if a.gd_steps is not None else SolverOpts().gd_steps),
            gd_lr=a.gd_lr,
            gd_opt=(a.gd_opt if a.gd_opt is not None else SolverOpts().gd_opt),
            gd_ridge=a.gd_ridge,
            gd_project=a.gd_project,
            gd_proj_params=(json.loads(a.gd_proj_params) if a.gd_proj_params else None),
        )

        if a.use_jax:
            try:
                from . import jax_accel as jxa
                jxa.enable_x64(bool(a.jax_x64))
            except Exception:
                raise RuntimeError("JAX requested via --use-jax but not available. ...")

        from .run_single import run_single

        out = run_single(
            cfg,
            seed=a.seed,
            sopts=sopts,
            algs=cfg.algs,
            use_jax=a.use_jax,
            jax_x64=a.jax_x64,
            light=a.light,
        )
        print(f"[pyident] JAX backend: {out['env'].get('accelerator')}  x64={out['env'].get('jax_x64')}")
        if a.json_out:
            from .io_utils import save_json
            import sys
            os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
            with open(a.out_json, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Wrote {a.out_json}", file=sys.stderr)
            if a.plots:
                plot_dir = a.plot_dir or (os.path.dirname(a.out_csv) or ".")
                _quick_plots_from_csv(a.out_csv, plot_dir, fmt=a.plot_format)
        else:
            print(json.dumps(out, indent=2))
            if a.out_json:
                import os, json as _json
                os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
                with open(a.out_json, "w") as f:
                    _json.dump(out, f, indent=2)
        return


    if a.cmd == "sweep-underactuation":
        if a.use_jax:
            try:
                from . import jax_accel as jxa
                jxa.enable_x64(bool(a.jax_x64))
            except Exception:
                raise RuntimeError("JAX requested via --use-jax but not available.")

        from .discarded.underactuation import sweep_underactuation

        m_values = _parse_int_list(a.m_values)
        seeds = _parse_int_list(a.seeds)
        algs = tuple(s.strip() for s in a.algs.split(",") if s.strip())

        # Ensure output dir exists
        os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)

        sweep_underactuation(
            n=a.n,
            m_values=m_values,
            T=a.T,
            dt=a.dt,
            ensemble=a.ensemble,
            signal=a.signal,
            sigPE=a.sigPE,
            seeds=seeds,
            algs=algs,
            out_csv=a.out_csv,
            use_jax=a.use_jax,
            jax_x64=a.jax_x64,
            x0_mode=a.x0_mode,
            sparse_which=a.sparse_which,
            p_density_B=a.p_density_B
        )

        print(f"Wrote {a.out_csv}")

        if a.plots:
            plot_dir = a.plot_dir or (os.path.dirname(a.out_csv) or ".")
            _quick_plots_from_csv(a.out_csv, plot_dir, fmt=a.plot_format)
        return

    if a.cmd == "sweep-sparsity":
        if a.use_jax:
            try:
                from . import jax_accel as jxa
                jxa.enable_x64(bool(a.jax_x64))
            except Exception:
                raise RuntimeError("JAX requested via --use-jax but not available.")

        from .discarded.sparsity import sweep_sparsity
            
        p_values = _parse_float_list(a.p_values)
        seeds = _parse_int_list(a.seeds)
        algs = tuple(s.strip() for s in a.algs.split(",") if s.strip())

        os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)

        sweep_sparsity(
            n=a.n,
            m=a.m,
            p_values=p_values,
            T=a.T,
            dt=a.dt,
            sparse_which=a.sparse_which,
            signal=a.signal,
            sigPE=a.sigPE,
            seeds=seeds,
            algs=algs,
            out_csv=a.out_csv,
            use_jax=a.use_jax,
            jax_x64=a.jax_x64,
            x0_mode=a.x0_mode,
            U_restr_dim=a.U_restr_dim,
        )

        print(f"Wrote {a.out_csv}")
        if a.plots:
            plot_dir = a.plot_dir or (os.path.dirname(a.out_csv) or ".")
            _quick_plots_from_csv(a.out_csv, plot_dir, fmt=a.plot_format)
        return

    if a.cmd == "underactuation-plus":
        if getattr(a, "use_jax", False):
            try:
                from . import jax_accel as jxa
                jxa.enable_x64(bool(a.jax_x64))
            except Exception:
                raise RuntimeError("JAX requested via --use-jax but not available.")
        from .discarded.underactuation import sweep_underactuation_plus
        from .io_utils import save_json
        res = sweep_underactuation_plus(
            n=a.n,
            m_values=a.m_values,
            T=a.T,
            dt=a.dt,
            trials=a.trials,
            sigPE=a.sigPE,
            use_jax=getattr(a, "use_jax", False),
        )
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        save_json(res, a.out)
        print(f"Wrote {a.out}")
        if a.plots:
            plot_dir = a.plot_dir or (os.path.dirname(a.out_csv) or ".")
            _quick_plots_from_csv(a.out_csv, plot_dir, fmt=a.plot_format)
        return

    if a.cmd == "sparsity-plus":
        if getattr(a, "use_jax", False):
            try:
                from . import jax_accel as jxa
                jxa.enable_x64(bool(a.jax_x64))
            except Exception:
                raise RuntimeError("JAX requested via --use-jax but not available.")
        from .discarded.sparsity import sweep_sparsity_plus
        from .io_utils import save_json
        res = sweep_sparsity_plus(
            n=a.n,
            m=a.m,
            p_values=a.p_values,
            T=a.T,
            dt=a.dt,
            trials=a.trials,
            sigPE=a.sigPE,
            sparse_which=a.sparse_which,
            use_jax=getattr(a, "use_jax", False),
        )
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        save_json(res, a.out)
        print(f"Wrote {a.out}")
        if a.plots:
            plot_dir = a.plot_dir or (os.path.dirname(a.out_csv) or ".")
            _quick_plots_from_csv(a.out_csv, plot_dir, fmt=a.plot_format)
        return
    
# ============= Helpers =============

def _parse_seeds(s: str):
    s = s.strip()
    if ":" in s:
        a, b = s.split(":")
        return list(range(int(a), int(b)))
    return [int(x) for x in s.split(",") if x.strip() != ""]

def _ensure_dir(path: str):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def _save_fig_both(fig, outbase: str, fmt: str):
    import matplotlib.pyplot as plt
    fmt = fmt.lower()
    if fmt in ("png", "both"):
        fig.savefig(outbase + ".png", dpi=150, bbox_inches="tight")
    if fmt in ("pdf", "both"):
        fig.savefig(outbase + ".pdf", bbox_inches="tight")
    plt.close(fig)

def _quick_plots_from_csv(csv_path: str, plot_dir: str, fmt: str = "png"):
    """
    Minimal, schema-agnostic plot generator:
      - histograms for K_rank (if present) and delta_pbh (if present)
      - median trend vs a detected group key among ['m','p_density','sigPE','r','q'] if present
    Uses matplotlib directly for the trends; uses plots.py histogram when available.
    """
    from . import plots as P
    import matplotlib.pyplot as plt

    _ensure_dir(plot_dir)
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    # Collect numeric columns as needed
    def get_col(name, cast=float):
        vals = []
        for r in rows:
            if name in r and r[name] not in ("", None):
                try:
                    vals.append(cast(r[name]))
                except Exception:
                    pass
        return vals

    k_rank = get_col("K_rank", int)
    delta  = get_col("delta_pbh", float)

    base = os.path.splitext(os.path.basename(csv_path))[0]
    outbase = os.path.join(plot_dir, base)

    # Histograms (robust, small)
    if k_rank:
        bins = min(30, max(5, len(k_rank)//4))
        if fmt in ("png", "both"):
            P.plot_histogram(k_rank, bins=bins,
                            out_png=outbase + "_Krank.png",
                            title="K_rank distribution", xlabel="K_rank")
        if fmt in ("pdf", "both"):
            P.plot_histogram(k_rank, bins=bins,
                            out_pdf=outbase + "_Krank.pdf",
                            title="K_rank distribution", xlabel="K_rank")

    if delta:
        bins = min(30, max(5, len(delta)//4))
        if fmt in ("png", "both"):
            P.plot_histogram(delta, bins=bins,
                            out_png=outbase + "_PBH.png",
                            title="PBH margin distribution", xlabel="δ_PBH")
        if fmt in ("pdf", "both"):
            P.plot_histogram(delta, bins=bins,
                            out_pdf=outbase + "_PBH.pdf",
                            title="PBH margin distribution", xlabel="δ_PBH")
    

    # Grouped medians for common sweep keys
    for gkey in ("m", "p_density", "sigPE", "r", "q"):
        gvals = get_col(gkey, float)
        if not gvals:
            continue
        # Build per-group medians
        from collections import defaultdict
        by = defaultdict(lambda: {"K_rank": [], "delta_pbh": []})
        for r in rows:
            try:
                g = float(r[gkey])
            except Exception:
                continue
            if "K_rank" in r:
                try: by[g]["K_rank"].append(int(r["K_rank"]))
                except Exception: pass
            if "delta_pbh" in r:
                try: by[g]["delta_pbh"].append(float(r["delta_pbh"]))
                except Exception: pass

        groups = sorted(by.keys())
        if not groups:
            continue

        # K_rank median trend
        y1 = []
        for g in groups:
            arr = by[g]["K_rank"]
            y1.append(float(np.median(arr)) if arr else np.nan)
        if any(np.isfinite(y1)):
            fig, ax = plt.subplots(figsize=(5.6, 3.2))
            ax.plot(groups, y1, marker="o")
            ax.set_title(f"Median K_rank vs {gkey}")
            ax.set_xlabel(gkey); ax.set_ylabel("median K_rank")
            _save_fig_both(fig, outbase + f"_Krank_vs_{gkey}", fmt)

        # δ_PBH median trend
        y2 = []
        for g in groups:
            arr = by[g]["delta_pbh"]
            y2.append(float(np.median(arr)) if arr else np.nan)
        if any(np.isfinite(y2)):
            fig, ax = plt.subplots(figsize=(5.6, 3.2))
            ax.plot(groups, y2, marker="o")
            ax.set_title(f"Median δ_PBH vs {gkey}")
            ax.set_xlabel(gkey); ax.set_ylabel("median δ_PBH")
            _save_fig_both(fig, outbase + f"_PBH_vs_{gkey}", fmt)

        break  # plot against the first grouping key we find


if __name__ == "__main__":
    main()
