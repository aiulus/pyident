"""End-to-end experiment runner for the uncontrollable identifiability pipeline.

This script reproduces the multi-step workflow:
1) sim_regcomb_ctrb (grid sweep + save matrices)
2) filter_unctrb_dataset (density + uncontrollable filter)
3) sim_unctrb_x0_boxplot (identifiability score boxplots)
4) sim_unctrb_pbh_estimators (select low-PBH triples + estimation errors)
5) sim_unctrb_pbh_error_boxplots (error boxplots)

All steps are executed in-process without shelling out.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Sequence

from pyident.experiments import sim_regcomb_ctrb
from pyident.experiments import filter_unctrb_dataset
from pyident.experiments import sim_unctrb_x0_boxplot
from pyident.experiments import sim_unctrb_pbh_estimators
from pyident.experiments import sim_unctrb_pbh_error_boxplots


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
            "--samples",
            str(args.samples),
            "--outdir",
            str(ensemble_out),
            "--save-matrices",
        ]
    )
    # default m := n if omitted
    if regcomb_args.m is None:
        regcomb_args.m = int(regcomb_args.n)
    sim_regcomb_ctrb.run(regcomb_args)

    # 2) Filter uncontrollable by density
    filter_args = filter_unctrb_dataset.build_parser().parse_args(
        [
            "--outdir",
            str(ensemble_out),
            "--density-min",
            str(args.density_min),
            "--density-max",
            str(args.density_max),
            "--density-source",
            args.density_source,
            "--max-spectral-radius",
            "6.0",
            "--min-b-norm",
            "1e-6",
        ]
    )
    filter_unctrb_dataset.run(filter_args)

    filtered_csv = ensemble_out / f"systems_unctrb_d{args.density_min:g}_{args.density_max:g}.csv"
    filtered_npz = ensemble_out / f"systems_unctrb_d{args.density_min:g}_{args.density_max:g}.npz"

    # 3) Identifiability score boxplots
    x0_args = sim_unctrb_x0_boxplot.build_parser().parse_args(
        [
            "--dataset-csv",
            str(filtered_csv),
            "--dataset-npz",
            str(filtered_npz),
            "--x0-samples",
            str(args.x0_samples),
            "--mask-ps",
            *[str(p) for p in args.mask_ps],
            "--mask-renorm",
            "--x0-min-norm",
            "1e-6",
            "--x0-min-support",
            "1",
            "--x0-max-attempts",
            "128",
            "--outdir",
            str(x0_out),
            "--outlier-trim",
            str(args.outlier_trim),
        ]
    )
    sim_unctrb_x0_boxplot.run(x0_args)

    # 4) Low-PBH selection + estimation
    pbh_args = sim_unctrb_pbh_estimators.build_parser().parse_args(
        [
            "--dataset-csv",
            str(filtered_csv),
            "--dataset-npz",
            str(filtered_npz),
            "--outdir",
            str(pbh_out),
            "--seed",
            str(args.seed),
            "--x0-samples",
            str(args.x0_samples),
            "--mask-ps",
            *[str(p) for p in args.mask_ps_pbh],
            "--mask-renorm",
            "--x0-min-norm",
            "1e-6",
            "--x0-min-support",
            "1",
            "--x0-max-attempts",
            "128",
            "--pbh-threshold",
            str(args.pbh_threshold),
            "--min-visible-dim",
            "1",
            "--max-spectral-radius",
            "6.0",
            "--min-b-norm",
            "1e-6",
            "--T",
            str(args.T),
            "--dt",
            str(args.dt),
            "--u-scale",
            str(args.u_scale),
            "--dwell",
            str(args.dwell),
            "--algos",
            args.algos,
            "--dmdc-z-cond-max",
            "1e8",
        ]
    )
    sim_unctrb_pbh_estimators.run(pbh_args)

    # 5) Error boxplots
    selected_suffix = args.selected_suffix or f"pbh_lt_{args.pbh_threshold:g}"
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
            "1",
            "--T",
            str(args.T),
            "--dt",
            str(args.dt),
            "--u-scale",
            str(args.u_scale),
            "--dwell",
            str(args.dwell),
            "--algos",
            args.algos,
            "--dmdc-z-cond-max",
            "1e8",
        ]
    )
    sim_unctrb_pbh_error_boxplots.run(box_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-outdir", default="pyident_results", help="root output directory")
    parser.add_argument("--ensemble-dir", default="fresh_ensemble", help="subdir for ensemble outputs")
    parser.add_argument("--x0-dir", default="fresh_ABx0", help="subdir for x0 score plots")
    parser.add_argument("--pbh-dir", default="fresh_nonidentifiable_ABx0", help="subdir for PBH selection + errors")

    parser.add_argument("--sparsity-grid", default="0.0:0.1:1.0")
    parser.add_argument("--ndim-grid", default="2:1:10")
    parser.add_argument("--samples", type=int, default=10000)

    parser.add_argument("--density-min", type=float, default=0.3)
    parser.add_argument("--density-max", type=float, default=0.7)
    parser.add_argument("--density-source", choices=["A", "B", "AB"], default="AB")

    parser.add_argument("--x0-samples", type=int, default=10)
    parser.add_argument("--mask-ps", nargs="*", default=[0.25, 0.5, 0.75])
    parser.add_argument("--outlier-trim", type=float, default=0.05)

    parser.add_argument("--mask-ps-pbh", nargs="*", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--pbh-threshold", type=float, default=1e-6)

    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--u-scale", type=float, default=3.0)
    parser.add_argument("--dwell", type=int, default=1)
    parser.add_argument("--algos", type=str, default="DMDc")

    parser.add_argument(
        "--selected-suffix",
        default=None,
        help="override selected dataset suffix (default: pbh_lt_<threshold>)",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
