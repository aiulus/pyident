"""Estimate (A,B) errors on low-PBH (A,B,x0) triples.

Pipeline
--------
1) Load (A,B) dataset saved by filter_unctrb_dataset.py.
2) Re-generate x0 samples (same procedure as sim_unctrb_x0_boxplot).
3) Select (A,B,x0) with PBH score below a threshold.
4) Simulate trajectories under PRBS excitation.
5) Estimate (A,B) via chosen estimators and compute relative errors
   in the standard basis and the V(x0)-basis.
"""
from __future__ import annotations

import argparse
import math
import pathlib
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from ..metrics import pbh_margin_structured
from ..metrics import left_eigvec_overlap
from ..projectors import build_visible_basis_dt
from ..simulation import prbs, simulate_dt
from ..estimators import dmdc_tls, moesp_fit_old, sindy_fit, node_fit
from ..experiments.sim_unctrb_x0_boxplot import (
    sample_unit_sphere,
    sample_masked_sphere,
    format_mode_label,
)


def x0_sampler(mode: str, p_keep: float, renorm: bool):
    if mode == "sphere":
        return lambda n, rng: sample_unit_sphere(n, rng)
    if mode == "mask":
        return lambda n, rng: sample_masked_sphere(n, rng, p_keep, renorm=renorm)
    raise ValueError(f"unknown x0 sampling mode '{mode}'")


def compute_pbh(A: np.ndarray, B: np.ndarray, x0: np.ndarray) -> float:
    return float(pbh_margin_structured(A, B, x0))


def compute_mu_min(A: np.ndarray, B: np.ndarray, x0: np.ndarray) -> float:
    Xaug = np.concatenate([x0.reshape(-1, 1), B], axis=1)
    mu_vals = left_eigvec_overlap(A, Xaug)
    return float(np.min(mu_vals)) if mu_vals.size else 0.0


def relative_errors(Ahat: np.ndarray, Bhat: np.ndarray, Atrue: np.ndarray, Btrue: np.ndarray) -> Dict[str, float]:
    errA = float(np.linalg.norm(Ahat - Atrue, ord="fro"))
    errB = float(np.linalg.norm(Bhat - Btrue, ord="fro"))
    nrmA = float(np.linalg.norm(Atrue, ord="fro") + 1e-15)
    nrmB = float(np.linalg.norm(Btrue, ord="fro") + 1e-15)
    relA = errA / nrmA
    relB = errB / nrmB
    return {
        "errA_rel": relA,
        "errB_rel": relB,
        "err_mean_rel": 0.5 * (relA + relB),
    }


def _visible_basis(A: np.ndarray, B: np.ndarray, x0: np.ndarray, tol: float) -> np.ndarray:
    return build_visible_basis_dt(A, B, x0, tol=tol)


def _moesp_wrapper(X0: np.ndarray, X1: np.ndarray, U_cm: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    n = X0.shape[0]
    return moesp_fit_old(X0, X1, U_cm, n=n)


def _resolve_estimators(names: Sequence[str]):
    registry = {
        "SINDy": lambda X0, X1, U_cm, dt: sindy_fit(X0, X1, U_cm, dt),
        "DMDc": lambda X0, X1, U_cm, dt: dmdc_tls(X0, X1, U_cm),
        "MOESP": _moesp_wrapper,
        "NODE": lambda X0, X1, U_cm, dt: node_fit(
            X0, X1, U_cm, dt, epochs=200, lr=1e-2, early_stopping=True, verbose=False
        ),
    }
    lookup = {k.lower(): k for k in registry.keys()}
    resolved = []
    for name in names:
        key = name.strip().lower()
        if not key:
            continue
        if key not in lookup:
            raise ValueError(f"Unknown estimator '{name}'. Choose from {list(registry)}.")
        resolved.append(lookup[key])
    if not resolved:
        raise ValueError("No estimators selected.")
    return {name: registry[name] for name in resolved}


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    systems_df = pd.read_csv(args.dataset_csv)
    matrices = np.load(args.dataset_npz, allow_pickle=True)
    A_list = matrices["A"]
    B_list = matrices["B"]
    sys_idx = matrices["system_index"]
    index_map = {int(idx): pos for pos, idx in enumerate(sys_idx)}

    mode_configs: list[tuple[str, float]] = []
    p_list = [float(p) for p in args.mask_ps]
    if 1.0 not in p_list:
        p_list.append(1.0)
    p_list = sorted(set(p_list))
    for p in p_list:
        if math.isclose(p, 1.0, rel_tol=0.0, abs_tol=1e-12):
            mode_configs.append(("sphere", 1.0))
        else:
            mode_configs.append(("mask", p))

    estimators = _resolve_estimators(args.algos.split(",") if args.algos else ["SINDy", "DMDc", "MOESP", "NODE"])

    selected_rows: list[dict[str, Any]] = []
    selected_A: list[np.ndarray] = []
    selected_B: list[np.ndarray] = []
    selected_x0: list[np.ndarray] = []
    selected_sys_idx: list[int] = []
    selected_x0_idx: list[int] = []

    results_rows: list[dict[str, Any]] = []

    for _, row in systems_df.iterrows():
        sys_id = int(row["system_index"])
        pos = index_map.get(sys_id)
        if pos is None:
            raise RuntimeError(f"system_index {sys_id} missing from matrices file")
        A = A_list[pos]
        B = B_list[pos]
        n = int(row["n"])
        m = int(row["m"])

        for mode, p_keep in mode_configs:
            sampler = x0_sampler(mode, p_keep, renorm=args.mask_renorm)
            for x0_idx in range(args.x0_samples):
                x0 = sampler(n, rng)
                pbh = compute_pbh(A, B, x0)
                if pbh >= args.pbh_threshold:
                    continue

                # Store selected triple
                sel = {
                    "system_index": sys_id,
                    "x0_index": x0_idx,
                    "mode": format_mode_label(mode, p_keep),
                    "p_keep": p_keep,
                    "pbh": float(pbh),
                    "mu_min": compute_mu_min(A, B, x0),
                }
                for col, val in row.items():
                    if np.isscalar(val):
                        sel[col] = val
                selected_rows.append(sel)
                selected_A.append(A)
                selected_B.append(B)
                selected_x0.append(x0)
                selected_sys_idx.append(sys_id)
                selected_x0_idx.append(x0_idx)

                # Simulate trajectory
                U = prbs(args.T, m, scale=args.u_scale, dwell=args.dwell, rng=rng)
                X = simulate_dt(x0, A, B, U)
                X0 = X[:, :-1]
                X1 = X[:, 1:]
                U_cm = U.T

                # Visible basis and P
                Vbasis = _visible_basis(A, B, x0, tol=args.visible_tol)
                A_V = Vbasis.T @ A @ Vbasis if Vbasis.size else np.zeros((0, 0))
                B_V = Vbasis.T @ B if Vbasis.size else np.zeros((0, B.shape[1]))

                for algo_name, algo_fn in estimators.items():
                    try:
                        Ahat, Bhat = algo_fn(X0, X1, U_cm, args.dt)
                        Ahat_V = Vbasis.T @ Ahat @ Vbasis if Vbasis.size else np.zeros((0, 0))
                        Bhat_V = Vbasis.T @ Bhat if Vbasis.size else np.zeros((0, Bhat.shape[1]))
                        err_std = relative_errors(Ahat, Bhat, A, B)
                        err_P = relative_errors(Ahat_V, Bhat_V, A_V, B_V)
                        err_msg = ""
                    except Exception as exc:
                        err_std = {"errA_rel": np.nan, "errB_rel": np.nan, "err_mean_rel": np.nan}
                        err_P = {"errA_rel": np.nan, "errB_rel": np.nan, "err_mean_rel": np.nan}
                        err_msg = str(exc)

                    results_rows.append(
                        {
                            **sel,
                            "algo": algo_name,
                            "dim_visible": int(Vbasis.shape[1]),
                            "errA_rel": err_std["errA_rel"],
                            "errB_rel": err_std["errB_rel"],
                            "err_mean_rel": err_std["err_mean_rel"],
                            "errA_rel_P": err_P["errA_rel"],
                            "errB_rel_P": err_P["errB_rel"],
                            "err_mean_rel_P": err_P["err_mean_rel"],
                            "estimator_error": err_msg,
                        }
                    )

                if args.max_selected is not None and len(selected_rows) >= args.max_selected:
                    break
            if args.max_selected is not None and len(selected_rows) >= args.max_selected:
                break
        if args.max_selected is not None and len(selected_rows) >= args.max_selected:
            break

    if not selected_rows:
        raise RuntimeError("No (A,B,x0) pairs satisfied the PBH threshold.")

    selected_df = pd.DataFrame(selected_rows)
    selected_suffix = args.suffix or f"pbh_lt_{args.pbh_threshold:g}"
    selected_csv = outdir / f"selected_{selected_suffix}.csv"
    selected_npz = outdir / f"selected_{selected_suffix}.npz"
    selected_df.to_csv(selected_csv, index=False)
    np.savez_compressed(
        selected_npz,
        A=np.array(selected_A, dtype=object),
        B=np.array(selected_B, dtype=object),
        x0=np.array(selected_x0, dtype=object),
        system_index=np.array(selected_sys_idx, dtype=int),
        x0_index=np.array(selected_x0_idx, dtype=int),
    )

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(outdir / "estimation_errors.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", required=True, help="filtered systems CSV")
    parser.add_argument("--dataset-npz", required=True, help="filtered systems NPZ")
    parser.add_argument("--outdir", required=True, help="output directory")

    parser.add_argument("--seed", type=int, default=12345, help="RNG seed")
    parser.add_argument("--x0-samples", type=int, default=10, help="x0 samples per mode per system")
    parser.add_argument("--mask-ps", nargs="*", default=[0.25, 0.5, 0.75, 1.0], help="Bernoulli keep probs")
    parser.add_argument("--mask-renorm", action="store_true", help="renormalize masked x0")

    parser.add_argument("--pbh-threshold", type=float, required=True, help="PBH cutoff for selection")
    parser.add_argument("--max-selected", type=int, default=None, help="cap on selected triples")
    parser.add_argument("--suffix", default=None, help="suffix for selected dataset files")

    parser.add_argument("--T", type=int, default=100, help="trajectory horizon")
    parser.add_argument("--dt", type=float, default=1.0, help="sampling interval for estimators")
    parser.add_argument("--u-scale", type=float, default=3.0, help="PRBS amplitude")
    parser.add_argument("--dwell", type=int, default=1, help="PRBS dwell time")
    parser.add_argument("--visible-tol", type=float, default=1e-12, help="tolerance for V(x0) basis")

    parser.add_argument(
        "--algos",
        type=str,
        default=None,
        help="Comma-separated list of estimators (default: SINDy,DMDc,MOESP,NODE)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
