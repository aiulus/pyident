"""Box plots of estimation errors for selected (A,B,x0) triples.

Loads the selected dataset produced by sim_unctrb_pbh_estimators and
re-runs estimators to compute error distributions in:
  - standard basis
  - visible-subspace restriction (A|_V, B|_V with V=V(x0))
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from ..projectors import build_visible_basis_dt
from ..simulation import prbs, simulate_dt
from ..estimators import dmdc_tls, moesp_fit_old, sindy_fit, node_fit
from .boxplot_style import set_default_mpl_style, nice_boxplot, nice_grouped_boxplot


ALG_ORDER = ["DMDc", "MOESP", "SINDy", "NODE"]


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

    if args.selected_csv:
        selected_df = pd.read_csv(args.selected_csv)
    else:
        selected_df = None

    sel = np.load(args.selected_npz, allow_pickle=True)
    A_list = sel["A"]
    B_list = sel["B"]
    x0_list = sel["x0"]

    estimators = _resolve_estimators(args.algos.split(",") if args.algos else ALG_ORDER)

    rows: list[dict[str, Any]] = []

    for idx in range(len(A_list)):
        A = A_list[idx]
        B = B_list[idx]
        x0 = x0_list[idx]
        n = A.shape[0]
        m = B.shape[1]

        U = prbs(args.T, m, scale=args.u_scale, dwell=args.dwell, rng=rng)
        X = simulate_dt(x0, A, B, U)
        X0 = X[:, :-1]
        X1 = X[:, 1:]
        U_cm = U.T

        Vbasis = _visible_basis(A, B, x0, tol=args.visible_tol)
        # Restrict to the visible subspace V(x0), matching the theory:
        # A|_V = V^T A V, B|_V = V^T B
        if Vbasis.size:
            A_V = Vbasis.T @ A @ Vbasis
            B_V = Vbasis.T @ B
        else:
            A_V = np.zeros((0, 0), dtype=float)
            B_V = np.zeros((0, B.shape[1]), dtype=float)

        base_info = {
            "sample_index": idx,
            "dim_visible": int(Vbasis.shape[1]),
        }
        if selected_df is not None:
            for col, val in selected_df.iloc[idx].items():
                if np.isscalar(val):
                    base_info[col] = val

        for algo_name, algo_fn in estimators.items():
            try:
                Ahat, Bhat = algo_fn(X0, X1, U_cm, args.dt)
                if Vbasis.size:
                    Ahat_V = Vbasis.T @ Ahat @ Vbasis
                    Bhat_V = Vbasis.T @ Bhat
                else:
                    Ahat_V = np.zeros((0, 0), dtype=float)
                    Bhat_V = np.zeros((0, Bhat.shape[1]), dtype=float)
                err_std = relative_errors(Ahat, Bhat, A, B)
                err_P = relative_errors(Ahat_V, Bhat_V, A_V, B_V)
                err_msg = ""
            except Exception as exc:
                err_std = {"errA_rel": np.nan, "errB_rel": np.nan, "err_mean_rel": np.nan}
                err_P = {"errA_rel": np.nan, "errB_rel": np.nan, "err_mean_rel": np.nan}
                err_msg = str(exc)

            rows.append(
                {
                    **base_info,
                    "algo": algo_name,
                    "err_mean_rel": err_std["err_mean_rel"],
                    "err_mean_rel_P": err_P["err_mean_rel"],
                    "estimator_error": err_msg,
                }
            )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(outdir / "estimation_errors_boxplot.csv", index=False)

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from exc

    set_default_mpl_style()
    order = [name for name in ALG_ORDER if name in results_df["algo"].unique()]
    std_data = [results_df[results_df["algo"] == name]["err_mean_rel"].to_numpy() for name in order]
    P_data = [results_df[results_df["algo"] == name]["err_mean_rel_P"].to_numpy() for name in order]
    yscale = None if args.yscale == "none" else args.yscale

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    nice_boxplot(
        ax,
        std_data,
        order,
        title="Standard-basis estimation error",
        ylabel="mean relative error (standard)",
        yscale=yscale,
    )
    fig.tight_layout()
    fig.savefig(outdir / "boxplot_err_standard.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    nice_boxplot(
        ax,
        P_data,
        order,
        title="Visible-subspace estimation error",
        ylabel="mean relative error (V-basis)",
        yscale=yscale,
    )
    fig.tight_layout()
    fig.savefig(outdir / "boxplot_err_Pbasis.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    nice_grouped_boxplot(
        ax,
        std_data,
        P_data,
        order,
        title="Standard vs visible-subspace estimation error",
        ylabel="mean relative error",
        yscale=yscale,
    )
    fig.tight_layout()
    fig.savefig(outdir / "boxplot_err_standard_vs_Pbasis.png", bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-npz", required=True, help="selected dataset NPZ from sim_unctrb_pbh_estimators")
    parser.add_argument("--selected-csv", default=None, help="selected dataset CSV (optional, for metadata)")
    parser.add_argument("--outdir", required=True, help="output directory")

    parser.add_argument("--seed", type=int, default=12345, help="RNG seed")
    parser.add_argument("--T", type=int, default=100, help="trajectory horizon")
    parser.add_argument("--dt", type=float, default=1.0, help="sampling interval for estimators")
    parser.add_argument("--u-scale", type=float, default=3.0, help="PRBS amplitude")
    parser.add_argument("--dwell", type=int, default=1, help="PRBS dwell time")
    parser.add_argument("--visible-tol", type=float, default=1e-12, help="tolerance for V(x0) basis")

    parser.add_argument(
        "--algos",
        type=str,
        default=None,
        help="Comma-separated list of estimators (default: DMDc,MOESP,SINDy,NODE)",
    )
    parser.add_argument(
        "--yscale",
        choices=["none", "log"],
        default="none",
        help="y-axis scale for boxplots (default: none)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
