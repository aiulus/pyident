"""Box plots of identifiability scores for uncontrollable systems.

Workflow
--------
1) Draw (A,B) from the same ensembles as ``sim_regcomb`` *or* load a filtered dataset.
2) Keep only uncontrollable systems with density in [density-min, density-max] (generation mode).
3) Sample x0 via different modes (unit sphere or masked sphere).
4) Compute identifiability scores (PBH margin, left-eigenvector score).
5) Create box plots per score showing distributions per x0 mode.

Example
-------
python -m pyident.experiments.sim_unctrb_x0_boxplot --axes "sparsity, ndim" \
    --sparsity-grid 0.0:0.1:1.0 --ndim-grid 2:2:20 --samples 100 \
    --x0-samples 100 --outdir results/unctrb_x0
"""
from __future__ import annotations

import argparse
import math
import pathlib
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

try:  # pragma: no cover - import guard for optional dependency
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without matplotlib
    plt = None  # type: ignore[assignment]
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None

import numpy as np
import pandas as pd

from ..ensembles import controllability_rank
from ..metrics import pbh_margin_structured, left_eigvec_overlap
from . import sim_regcomb as base


SCORES = ("pbh", "mu")


def sample_unit_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(n)
    nrm = float(np.linalg.norm(v))
    return v / (nrm if nrm > 0.0 else 1.0)


def sample_masked_sphere(
    n: int,
    rng: np.random.Generator,
    p_keep: float,
    *,
    renorm: bool,
) -> np.ndarray:
    x0 = sample_unit_sphere(n, rng)
    mask = rng.random(n) < p_keep
    x0 = x0 * mask
    if renorm:
        nrm = float(np.linalg.norm(x0))
        if nrm > 0.0:
            x0 = x0 / nrm
    return x0


def x0_sampler(mode: str, p_keep: float, renorm: bool):
    if mode == "sphere":
        return lambda n, rng: sample_unit_sphere(n, rng)
    if mode == "mask":
        return lambda n, rng: sample_masked_sphere(n, rng, p_keep, renorm=renorm)
    raise ValueError(f"unknown x0 sampling mode '{mode}'")


def format_mode_label(mode: str, p_keep: float) -> str:
    if mode == "sphere":
        return "sphere"
    return f"mask_p={p_keep:g}"


def _legacy_scenarios(args: argparse.Namespace) -> list[tuple[str, float, int, int, dict[str, Any]]]:
    property_name = args.property
    scenarios: list[tuple[str, float, int, int, dict[str, Any]]] = []

    if property_name == "underactuation":
        n_grid_spec = args.n_grid or args.cond_grid
        if n_grid_spec is None:
            raise ValueError("--cond-grid or --n-grid must be provided for property=underactuation")
        if args.m_grid is None:
            raise ValueError("--m-grid must be provided for property=underactuation")
        n_values = base.parse_grid(n_grid_spec)
        m_values = base.parse_grid(args.m_grid)
        for n_val in n_values:
            n_cur = int(round(n_val))
            if n_cur <= 0:
                raise ValueError("state dimensions must be positive integers")
            for m_val in m_values:
                m_cur = int(round(m_val))
                if m_cur <= 0:
                    raise ValueError("input dimensions must be positive integers")
                prop_val = float(n_cur - m_cur)
                info = {
                    "property": property_name,
                    "property_value": prop_val,
                    "n": n_cur,
                    "m": m_cur,
                    "input_fraction": float(m_cur) / float(n_cur),
                }
                scenarios.append((property_name, prop_val, n_cur, m_cur, info))
        return scenarios

    conditioning_values = base.parse_grid(args.cond_grid)
    for prop_value in conditioning_values:
        info = {
            "property": property_name,
            "property_value": float(prop_value),
            "n": args.n,
            "m": args.m,
        }
        scenarios.append((property_name, float(prop_value), args.n, args.m, info))
    return scenarios


def _density_value(meta: Mapping[str, Any], source: str) -> float:
    if source == "A":
        return float(meta.get("density_A", np.nan))
    if source == "B":
        return float(meta.get("density_B", np.nan))
    return float(meta.get("density_AB", np.nan))


def _score_values(A: np.ndarray, B: np.ndarray, x0: np.ndarray) -> dict[str, float]:
    scores: dict[str, float] = {}
    scores["pbh"] = float(pbh_margin_structured(A, B, x0))
    Xaug = np.concatenate([x0.reshape(-1, 1), B], axis=1)
    mu_vals = left_eigvec_overlap(A, Xaug)
    scores["mu"] = float(np.min(mu_vals)) if mu_vals.size else 0.0
    return scores


def _scenario_label(info: Mapping[str, Any], axes: Sequence[str]) -> str:
    if not axes:
        return "all"
    parts = []
    for axis in axes:
        col = base.AXIS_COLUMN[axis]
        val = info.get(col, info.get(axis, None))
        parts.append(f"{axis}={val}")
    return ",".join(parts) if parts else "all"


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    outdir = base.ensure_output_dir(pathlib.Path(args.outdir))

    if args.dataset_csv and not args.dataset_npz:
        raise ValueError("--dataset-npz must be provided with --dataset-csv")

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

    records: list[dict[str, Any]] = []

    if args.dataset_csv:
        systems_df = pd.read_csv(args.dataset_csv)
        matrices = np.load(args.dataset_npz, allow_pickle=True)
        A_list = matrices["A"]
        B_list = matrices["B"]
        sys_idx = matrices["system_index"]
        index_map = {int(idx): pos for pos, idx in enumerate(sys_idx)}
        density_col = {
            "A": "meta_density_A",
            "B": "meta_density_B",
            "AB": "meta_density_AB",
        }.get(args.density_source, "meta_density_AB")

        for _, row in systems_df.iterrows():
            sys_id = int(row["system_index"])
            pos = index_map.get(sys_id)
            if pos is None:
                raise RuntimeError(f"system_index {sys_id} missing from matrices file")
            A = A_list[pos]
            B = B_list[pos]
            n_cur = int(row["n"])
            m_cur = int(row["m"])
            dval = float(row.get(density_col, np.nan))
            rk = int(row.get("controllability_rank", -1))

            for mode, p_keep in mode_configs:
                sampler = x0_sampler(mode, p_keep, renorm=args.mask_renorm)
                for _ in range(args.x0_samples):
                    x0 = sampler(n_cur, rng)
                    scores = _score_values(A, B, x0)
                    for score_name, value in scores.items():
                        rec = {
                            "score": score_name,
                            "value": float(value),
                            "mode": format_mode_label(mode, p_keep),
                            "p_keep": p_keep,
                            "n": n_cur,
                            "m": m_cur,
                            "density": dval,
                            "controllability_rank": rk,
                        }
                        for col, val in row.items():
                            if np.isscalar(val):
                                rec[col] = val
                        records.append(rec)
    else:
        axes = base.parse_axes_spec(args.axes)
        if axes:
            if args.property is not None and args.property != "density":
                raise ValueError("--property cannot be combined with --axes")
            scenarios = base.build_axis_scenarios(args, axes)
        else:
            scenarios = _legacy_scenarios(args)

        if not scenarios:
            raise ValueError("no scenarios were generated; please check the grid specifications")

        density_min = float(args.density_min)
        density_max = float(args.density_max)
        if density_min > density_max:
            raise ValueError("density-min must be <= density-max")

        for property_name, prop_value, n_cur, m_cur, prop_info in scenarios:
            accepted = 0
            draws = 0
            max_draws = int(args.max_draws)
            while accepted < args.samples and draws < max_draws:
                draws += 1
                A, B, meta = base.generate_system(
                    property_name,
                    prop_value,
                    n_cur,
                    m_cur,
                    rng,
                    sparse_which=args.sparse_which,
                    sparse_tol=args.sparse_tol,
                    base_density=args.sparse_density,
                    deficiency_base=args.deficiency_base,
                    deficiency_embed_random=not args.deficiency_no_embed,
                )

                rk, _ = controllability_rank(A, B, order=n_cur, rtol=args.ctrb_rtol)
                if rk >= n_cur:
                    continue

                meta.setdefault("density_A", base.matrix_density(A, tol=args.sparse_tol))
                meta.setdefault("density_B", base.matrix_density(B, tol=args.sparse_tol))
                meta.setdefault(
                    "density_AB", base.matrix_density(np.hstack([A, B]), tol=args.sparse_tol)
                )
                dval = _density_value(meta, args.density_source)
                if not (density_min <= dval <= density_max):
                    continue

                accepted += 1

                for mode, p_keep in mode_configs:
                    sampler = x0_sampler(mode, p_keep, renorm=args.mask_renorm)
                    for _ in range(args.x0_samples):
                        x0 = sampler(n_cur, rng)
                        scores = _score_values(A, B, x0)
                        for score_name, value in scores.items():
                            rec = {
                                "score": score_name,
                                "value": float(value),
                                "mode": format_mode_label(mode, p_keep),
                                "p_keep": p_keep,
                                "n": n_cur,
                                "m": m_cur,
                                "density": dval,
                                "controllability_rank": int(rk),
                            }
                            for key, val in prop_info.items():
                                if np.isscalar(val):
                                    rec[key] = val
                            records.append(rec)

            if accepted < args.samples:
                raise RuntimeError(
                    f"only accepted {accepted} systems (target {args.samples}) for scenario {prop_info}"
                )

    scores_df = pd.DataFrame(records)
    scores_path = outdir / "identifiability_scores.csv"
    scores_df.to_csv(scores_path, index=False)

    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plotting; please install it to run this experiment"
        ) from _MATPLOTLIB_IMPORT_ERROR

    axes = base.parse_axes_spec(args.axes)
    if axes:
        scenario_groups: list[tuple[str, pd.DataFrame]] = []
        group_cols = [base.AXIS_COLUMN[a] for a in axes]
        missing = [col for col in group_cols if col not in scores_df.columns]
        if missing:
            raise ValueError(f"axis columns missing from dataset: {missing}")
        for group_vals, group_df in scores_df.groupby(group_cols, dropna=False):
            if not isinstance(group_vals, tuple):
                group_vals = (group_vals,)
            info = {col: val for col, val in zip(group_cols, group_vals)}
            label = _scenario_label(info, axes)
            scenario_groups.append((label, group_df))
    else:
        scenario_groups = [("all", scores_df)]

    for scen_label, scen_df in scenario_groups:
        for score_name in SCORES:
            sub = scen_df[scen_df["score"] == score_name]
            if sub.empty:
                continue
            p_vals = sorted({float(p) for p in sub["p_keep"]})
            order = []
            if 1.0 in p_vals:
                order.append(format_mode_label("sphere", 1.0))
            order += [format_mode_label("mask", p) for p in p_vals if p < 1.0]
            data = [sub[sub["mode"] == mode]["value"].to_numpy() for mode in order]

            plt.figure(figsize=(7.2, 4.8))
            plt.boxplot(data, labels=order, showfliers=True)
            plt.ylabel(score_name)
            title = f"{score_name} score distribution"
            if scen_label != "all":
                title = f"{title} ({scen_label})"
            plt.title(title)
            plt.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            safe_label = scen_label.replace(" ", "_").replace(",", "_")
            plot_name = f"{score_name}_boxplot_{safe_label}.png"
            plt.savefig(outdir / "plots" / plot_name, dpi=200)
            plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-csv",
        default=None,
        help="filtered systems CSV (from filter_unctrb_dataset.py)",
    )
    parser.add_argument(
        "--dataset-npz",
        default=None,
        help="filtered systems matrices NPZ (from filter_unctrb_dataset.py)",
    )
    parser.add_argument("--n", type=int, default=10, help="state dimension")
    parser.add_argument("--m", type=int, default=2, help="input dimension")
    parser.add_argument(
        "--samples", type=int, default=50, help="accepted uncontrollable systems per scenario"
    )
    parser.add_argument(
        "--x0-samples", type=int, default=100, help="number of x0 draws per mode per system"
    )
    parser.add_argument(
        "--axes",
        default=None,
        help="comma-separated list of axes to sweep (subset of: sparsity, ndim, underactuation)",
    )
    parser.add_argument(
        "--property",
        default="density",
        choices=["density", "deficiency", "state_dimension", "underactuation"],
        help="system property to condition on",
    )
    parser.add_argument(
        "--cond-grid",
        default="0:0.05:1",
        help="conditioning grid specification (e.g., '0:0.05:1' or '0,0.5,1')",
    )
    parser.add_argument(
        "--n-grid",
        default=None,
        help="grid specification for state dimension sweeps (overrides --cond-grid for n)",
    )
    parser.add_argument(
        "--m-grid",
        default=None,
        help="grid specification for input dimension sweeps when property=underactuation",
    )
    parser.add_argument(
        "--sparsity-grid",
        default=None,
        help="grid specification for sparsity sweeps when using --axes",
    )
    parser.add_argument(
        "--ndim-grid",
        default=None,
        help="grid specification for state dimension sweeps when using --axes",
    )
    parser.add_argument("--seed", type=int, default=12345, help="base RNG seed")
    parser.add_argument("--outdir", default="results/unctrb_x0", help="output directory")

    parser.add_argument(
        "--ctrb-rtol",
        type=float,
        default=None,
        help="relative tolerance for controllability rank (default: numpy SVD tolerance)",
    )

    parser.add_argument(
        "--density-min",
        type=float,
        default=0.3,
        help="minimum density for filtering uncontrollable systems",
    )
    parser.add_argument(
        "--density-max",
        type=float,
        default=0.7,
        help="maximum density for filtering uncontrollable systems",
    )
    parser.add_argument(
        "--density-source",
        choices=["A", "B", "AB"],
        default="AB",
        help="which density to filter on: A, B, or AB",
    )
    parser.add_argument(
        "--max-draws",
        type=int,
        default=20000,
        help="maximum raw draws per scenario to reach the accepted sample count",
    )

    parser.add_argument(
        "--mask-ps",
        nargs="*",
        default=[0.25, 0.5, 0.75, 1.0],
        help="Bernoulli keep probabilities for masked x0 sampling",
    )
    parser.add_argument(
        "--mask-renorm",
        action="store_true",
        help="renormalize masked x0 to unit length when nonzero",
    )

    parser.add_argument(
        "--sparse-which",
        default="both",
        choices=["A", "B", "both"],
        help="which matrices to sparsify when property=density",
    )
    parser.add_argument(
        "--sparse-tol",
        type=float,
        default=1e-12,
        help="tolerance when measuring realised density",
    )
    parser.add_argument(
        "--sparse-density",
        type=float,
        default=0.3,
        help="baseline density used when property is not 'density'",
    )

    parser.add_argument(
        "--deficiency-base",
        default="ginibre",
        choices=["ginibre", "stable", "binary", "sparse"],
        help="base ensemble for controllable/uncontrollable blocks",
    )
    parser.add_argument(
        "--deficiency-no-embed",
        action="store_true",
        help="disable random basis embedding in draw_with_ctrb_rank",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
