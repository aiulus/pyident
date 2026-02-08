"""Visibility-sweep experiment runner (ICLR main).

This script targets visible-subspace dimensions explicitly and produces
axis-aligned plots of standard-basis vs V(x0)-basis errors, mirroring the
scheme described in the manuscript.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Sequence

from pyident.experiments.sim_equi import EstimatorConsistencyConfig, run_visibility_sweep_plots


def _parse_algos(spec: str) -> list[str]:
    if not spec:
        return []
    algos: list[str] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        algos.extend(part for part in chunk.split() if part)
    return algos


def run(args: argparse.Namespace) -> None:
    cfg = EstimatorConsistencyConfig()

    cfg.n = int(args.n)
    cfg.m = int(args.m)
    cfg.T = int(args.T)
    cfg.dt = float(args.dt)
    cfg.u_scale = float(args.u_scale)
    cfg.noise_std = float(args.noise_std)
    cfg.seed = int(args.seed)

    cfg.ensemble = args.ensemble
    cfg.det = bool(args.det)
    cfg.max_system_draws = int(args.max_system_draws)
    cfg.max_x0_draws = int(args.max_x0_draws)
    cfg.pe_order_max = int(args.pe_order_max)
    cfg.prbs_dwell_scale = float(args.prbs_dwell_scale)

    out_dir = pathlib.Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_dir = out_dir

    algos = _parse_algos(args.algos)
    if not algos:
        raise ValueError("--algos must specify at least one algorithm.")

    run_visibility_sweep_plots(
        cfg,
        algos=algos,
        ensemble_size=int(args.vis_ntrials),
        out_dir=out_dir,
        single_mode=bool(args.single),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="pyident_results/iclr_main", help="output directory")

    parser.add_argument("--n", type=int, default=15, help="state dimension")
    parser.add_argument("--m", type=int, default=5, help="input dimension")
    parser.add_argument("--T", type=int, default=200, help="trajectory horizon")
    parser.add_argument("--dt", type=float, default=0.1, help="discretization time step")
    parser.add_argument("--u-scale", type=float, default=5.0, help="PRBS amplitude")
    parser.add_argument("--noise-std", type=float, default=0.0, help="process noise std")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--ensemble",
        default="ginibre",
        choices=["ginibre", "stable", "binary", "sparse", "A_stbl_B_ctrb"],
        help="base ensemble used for block construction",
    )

    parser.add_argument(
        "--vis-ntrials",
        type=int,
        default=200,
        help="ensemble size per visible dimension",
    )
    parser.add_argument(
        "--algos",
        type=str,
        default="MOESP",
        help="comma-separated list of algorithms (e.g., MOESP,DMDc,SINDy,NODE)",
    )

    parser.add_argument(
        "--det",
        dest="det",
        action="store_true",
        help="use deterministic Krylov-based x0 construction",
    )
    parser.add_argument(
        "--no-det",
        dest="det",
        action="store_false",
        help="use rejection sampling for x0 construction",
    )
    parser.set_defaults(det=True)

    parser.add_argument(
        "--single",
        action="store_true",
        help="use the single-figure layout (if available)",
    )

    parser.add_argument("--max-system-draws", type=int, default=128)
    parser.add_argument("--max-x0-draws", type=int, default=256)
    parser.add_argument("--pe-order-max", type=int, default=16)
    parser.add_argument("--prbs-dwell-scale", type=float, default=8.0)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
