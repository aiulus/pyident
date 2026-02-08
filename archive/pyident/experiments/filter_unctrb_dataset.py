"""Filter uncontrollable systems by density from sim_regcomb_ctrb outputs.

Inputs: systems.csv + systems_matrices.npz (saved with --save-matrices).
Outputs: filtered systems CSV + matrices NPZ in same directory.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from .unctrb_utils import b_norms, spectral_radius

DENSITY_COL = {
    "A": "meta_density_A",
    "B": "meta_density_B",
    "AB": "meta_density_AB",
}


def run(args: argparse.Namespace) -> None:
    outdir = pathlib.Path(args.outdir)
    systems_path = outdir / args.systems_csv
    matrices_path = outdir / args.matrices_npz

    systems_df = pd.read_csv(systems_path)
    if "controllable" not in systems_df.columns:
        raise ValueError("systems.csv must include a 'controllable' column")

    density_col = DENSITY_COL[args.density_source]
    if density_col not in systems_df.columns:
        raise ValueError(f"systems.csv missing '{density_col}' column")

    density_min = float(args.density_min)
    density_max = float(args.density_max)
    if density_min > density_max:
        raise ValueError("density-min must be <= density-max")

    matrices = np.load(matrices_path, allow_pickle=True)
    A_list = matrices["A"]
    B_list = matrices["B"]
    sys_idx = matrices["system_index"]

    # Compute stability and B-norm metadata for all systems.
    rho_map: dict[int, float] = {}
    bnorm_map: dict[int, float] = {}
    brow_map: dict[int, float] = {}
    bcol_map: dict[int, float] = {}
    for pos, sys_id in enumerate(sys_idx):
        A = A_list[pos]
        B = B_list[pos]
        rho_map[int(sys_id)] = spectral_radius(A)
        b_fro, b_row, b_col = b_norms(B)
        bnorm_map[int(sys_id)] = b_fro
        brow_map[int(sys_id)] = b_row
        bcol_map[int(sys_id)] = b_col

    systems_df = systems_df.copy()
    systems_df["meta_spectral_radius"] = systems_df["system_index"].map(rho_map)
    systems_df["meta_b_norm_fro"] = systems_df["system_index"].map(bnorm_map)
    systems_df["meta_b_min_row_norm"] = systems_df["system_index"].map(brow_map)
    systems_df["meta_b_min_col_norm"] = systems_df["system_index"].map(bcol_map)

    mask = (
        (systems_df["controllable"] == 0)
        & (systems_df[density_col] >= density_min)
        & (systems_df[density_col] <= density_max)
    )
    if args.max_spectral_radius is not None:
        mask &= systems_df["meta_spectral_radius"] <= float(args.max_spectral_radius)
    if args.min_b_norm is not None:
        mask &= systems_df["meta_b_norm_fro"] >= float(args.min_b_norm)
    if args.min_b_row_norm is not None:
        mask &= systems_df["meta_b_min_row_norm"] >= float(args.min_b_row_norm)
    if args.min_b_col_norm is not None:
        mask &= systems_df["meta_b_min_col_norm"] >= float(args.min_b_col_norm)

    filtered_df = systems_df.loc[mask].copy()
    if filtered_df.empty:
        raise RuntimeError("no systems matched the uncontrollable + density filter")

    keep_mask = np.isin(sys_idx, filtered_df["system_index"].to_numpy())
    A_keep = A_list[keep_mask]
    B_keep = B_list[keep_mask]
    idx_keep = sys_idx[keep_mask]

    if len(A_keep) != len(filtered_df):
        raise RuntimeError("matrix count does not match filtered systems.csv")

    suffix = args.suffix or f"unctrb_d{density_min:g}_{density_max:g}"
    out_csv = outdir / f"systems_{suffix}.csv"
    out_npz = outdir / f"systems_{suffix}.npz"

    filtered_df.to_csv(out_csv, index=False)
    np.savez_compressed(
        out_npz,
        A=np.array(A_keep, dtype=object),
        B=np.array(B_keep, dtype=object),
        system_index=idx_keep,
        n=filtered_df["n"].to_numpy(),
        m=filtered_df["m"].to_numpy(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, help="directory with sim_regcomb_ctrb outputs")
    parser.add_argument(
        "--systems-csv",
        default="systems.csv",
        help="systems.csv file name in outdir",
    )
    parser.add_argument(
        "--matrices-npz",
        default="systems_matrices.npz",
        help="systems_matrices.npz file name in outdir",
    )
    parser.add_argument(
        "--density-min",
        type=float,
        default=0.3,
        help="minimum density filter",
    )
    parser.add_argument(
        "--density-max",
        type=float,
        default=0.7,
        help="maximum density filter",
    )
    parser.add_argument(
        "--density-source",
        choices=["A", "B", "AB"],
        default="AB",
        help="which density column to use",
    )
    parser.add_argument(
        "--max-spectral-radius",
        type=float,
        default=None,
        help="maximum allowed spectral radius of A (discrete-time stability filter)",
    )
    parser.add_argument(
        "--min-b-norm",
        type=float,
        default=None,
        help="minimum allowed Frobenius norm of B",
    )
    parser.add_argument(
        "--min-b-row-norm",
        type=float,
        default=None,
        help="minimum allowed row 2-norm of B",
    )
    parser.add_argument(
        "--min-b-col-norm",
        type=float,
        default=None,
        help="minimum allowed column 2-norm of B",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="suffix for output files (defaults to unctrb_dmin_dmax)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
