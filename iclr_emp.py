"""End-to-end experiment runner (ICLR empirical V variant).

This script mirrors ``iclrii.py`` but uses an empirical visible subspace
estimate (from simulated trajectories) when computing subsystem errors.
"""
from __future__ import annotations

import argparse
import math
import pathlib
from typing import Sequence

from pyident.experiments import sim_regcomb_ctrb
from pyident.experiments import sim_unctrb_x0_boxplot
from pyident.experiments import sim_unctrb_pbh_estimators
from pyident.experiments import sim_unctrb_pbh_error_boxplots


def _mask_ps_from_scheme(args: argparse.Namespace) -> list[float]:
    if args.sparse_p is not None and args.sphere:
        raise ValueError("Use either --sphere or --sparse-p, not both.")
    if args.sparse_p is not None:
        p = float(args.sparse_p)
        if not (0.0 < p <= 1.0):
            raise ValueError("--sparse-p must be in (0, 1].")
        return [p]
    # default to sphere
    return [1.0]


def run(args: argparse.Namespace) -> None:
    base_out = pathlib.Path(args.base_outdir)
    ensemble_out = base_out / args.ensemble_dir
    x0_out = base_out / args.x0_dir
    pbh_out = base_out / args.pbh_dir
    boxplot_out = pbh_out / "boxplots"

    # 1) Sweep + controllability fraction, save matrices
    regcomb_args = sim_regcomb_ctrb.build_parser().parse_args(
        [
            "--axes",
            "sparsity,ndim",
            "--sparsity-grid",
            args.sparsity_grid,
            "--ndim-grid",
            args.ndim_grid,
            *([] if args.m is None else ["--m", str(args.m)]),
            "--samples",
            str(args.samples),
            "--outdir",
            str(ensemble_out),
            "--save-matrices",
        ]
    )
    if regcomb_args.m is None:
        regcomb_args.m = int(regcomb_args.n)
    sim_regcomb_ctrb.run(regcomb_args)

    # Use the full (A,B) dataset as-is.
    systems_csv = ensemble_out / "systems.csv"
    systems_npz = ensemble_out / "systems_matrices.npz"

    mask_ps = _mask_ps_from_scheme(args)

    # 2) Identifiability score boxplots (single x0 scheme)
    x0_args = sim_unctrb_x0_boxplot.build_parser().parse_args(
        [
            "--dataset-csv",
            str(systems_csv),
            "--dataset-npz",
            str(systems_npz),
            "--x0-samples",
            str(args.x0_samples),
            "--mask-ps",
            *[str(p) for p in mask_ps],
            "--mask-renorm",
            "--x0-min-norm",
            str(args.x0_min_norm),
            "--x0-min-support",
            str(args.x0_min_support),
            "--x0-max-attempts",
            str(args.x0_max_attempts),
            "--outdir",
            str(x0_out),
            "--outlier-trim",
            str(args.outlier_trim),
        ]
    )
    sim_unctrb_x0_boxplot.run(x0_args)

    # 3) Estimation on x0 (PBH selection optional)
    pbh_threshold = float(args.pbh_threshold)
    if args.selected_suffix is not None:
        suffix = args.selected_suffix
    else:
        suffix = "all_x0" if math.isinf(pbh_threshold) else None
    pbh_args = sim_unctrb_pbh_estimators.build_parser().parse_args(
        [
            "--dataset-csv",
            str(systems_csv),
            "--dataset-npz",
            str(systems_npz),
            "--outdir",
            str(pbh_out),
            "--seed",
            str(args.seed),
            "--x0-samples",
            str(args.x0_samples),
            "--mask-ps",
            *[str(p) for p in mask_ps],
            "--mask-renorm",
            "--x0-min-norm",
            str(args.x0_min_norm),
            "--x0-min-support",
            str(args.x0_min_support),
            "--x0-max-attempts",
            str(args.x0_max_attempts),
            "--pbh-threshold",
            str(pbh_threshold),
            *([] if suffix is None else ["--suffix", suffix]),
            "--min-visible-dim",
            str(args.min_visible_dim),
            *([] if not args.require_partial else ["--require-partial"]),
            "--visible-basis",
            args.visible_basis,
            "--T",
            str(args.T),
            "--dt",
            str(args.dt),
            "--u-scale",
            str(args.u_scale),
            "--dwell",
            str(args.dwell),
            "--input-family",
            args.input_family,
            "--pe-method",
            args.pe_method,
            "--pe-tol",
            str(args.pe_tol),
            "--pe-max-tries",
            str(args.pe_max_tries),
            "--algos",
            args.algos,
            "--dmdc-z-cond-max",
            str(args.dmdc_z_cond_max),
        ]
        + (["--ridge", "--ridge-lam", str(args.ridge_lam)] if args.ridge else [])
    )
    sim_unctrb_pbh_estimators.run(pbh_args)

    # 4) Error boxplots
    selected_suffix = suffix or f"pbh_lt_{pbh_threshold:g}"
    selected_csv = pbh_out / f"selected_{selected_suffix}.csv"
    selected_npz = pbh_out / f"selected_{selected_suffix}.npz"

    box_args = sim_unctrb_pbh_error_boxplots.build_parser().parse_args(
        [
            "--selected-npz",
            str(selected_npz),
            "--selected-csv",
            str(selected_csv),
            "--outdir",
            str(boxplot_out),
            "--seed",
            str(args.seed),
            "--min-visible-dim",
            str(args.min_visible_dim),
            *([] if not args.require_partial else ["--require-partial"]),
            "--visible-basis",
            args.visible_basis,
            "--T",
            str(args.T),
            "--dt",
            str(args.dt),
            "--u-scale",
            str(args.u_scale),
            "--dwell",
            str(args.dwell),
            "--input-family",
            args.input_family,
            "--pe-method",
            args.pe_method,
            "--pe-tol",
            str(args.pe_tol),
            "--pe-max-tries",
            str(args.pe_max_tries),
            "--algos",
            args.algos,
            "--dmdc-z-cond-max",
            str(args.dmdc_z_cond_max),
        ]
        + (["--ridge", "--ridge-lam", str(args.ridge_lam)] if args.ridge else [])
    )
    sim_unctrb_pbh_error_boxplots.run(box_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-outdir", default="pyident_results", help="root output directory")
    parser.add_argument("--ensemble-dir", default="fresh_ensemble", help="subdir for ensemble outputs")
    parser.add_argument("--x0-dir", default="fresh_ABx0", help="subdir for x0 score plots")
    parser.add_argument("--pbh-dir", default="fresh_nonidentifiable_ABx0", help="subdir for estimator outputs")

    parser.add_argument("--sparsity-grid", default="0.0:0.1:1.0")
    parser.add_argument("--ndim-grid", default="2:1:10")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--m", type=int, default=None, help="input dimension for the ensemble")

    parser.add_argument("--x0-samples", type=int, default=10)
    parser.add_argument("--outlier-trim", type=float, default=0.05)

    parser.add_argument(
        "--sphere",
        action="store_true",
        help="use unit-sphere x0 sampling (default)",
    )
    parser.add_argument(
        "--sparse-p",
        type=float,
        default=None,
        help="mask keep probability when --x0-scheme sparse is used",
    )

    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--u-scale", type=float, default=3.0)
    parser.add_argument("--dwell", type=int, default=1)
    parser.add_argument("--algos", type=str, default="DMDc")

    parser.add_argument(
        "--input-family",
        choices=["prbs", "multisine"],
        default="prbs",
        help="input family used for PE generation",
    )
    parser.add_argument(
        "--pe-method",
        choices=["block", "moment"],
        default="block",
        help="PE verification method",
    )
    parser.add_argument("--pe-tol", type=float, default=1e-8)
    parser.add_argument("--pe-max-tries", type=int, default=128)
    parser.add_argument("--dmdc-z-cond-max", type=float, default=1e8)

    parser.add_argument("--x0-min-norm", type=float, default=1e-6)
    parser.add_argument("--x0-min-support", type=int, default=1)
    parser.add_argument("--x0-max-attempts", type=int, default=128)
    parser.add_argument("--min-visible-dim", type=int, default=0)
    parser.add_argument(
        "--require-partial",
        action="store_true",
        help="require 0 < dim V(x0) < n when selecting triples",
    )
    parser.add_argument(
        "--pbh-threshold",
        type=float,
        default=float("inf"),
        help="PBH cutoff for selection (default: no cutoff)",
    )

    parser.add_argument(
        "--visible-basis",
        choices=["oracle", "empirical"],
        default="empirical",
        help="basis for V(E): oracle uses A,B,x0; empirical uses trajectory span",
    )

    parser.add_argument(
        "--selected-suffix",
        default=None,
        help="override selected dataset suffix (default: all_x0)",
    )

    parser.add_argument(
        "--ridge",
        action="store_true",
        help="use ridge-regularized DMDc instead of TLS",
    )
    parser.add_argument(
        "--ridge-lam",
        type=float,
        default=1e-6,
        help="ridge parameter for DMDc when --ridge is enabled",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
