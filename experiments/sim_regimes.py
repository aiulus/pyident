"""Empirical score distributions conditioned on system properties.

This experiment samples sparse (A, B) ensembles together with random
initial conditions x0 that are drawn uniformly from the unit sphere. For a
chosen conditioning variable—by default the target sparsity level—we compute
several x0-based empirical scores and summarise their distributions via the
mean and standard deviation conditioned on the property value.

The pipeline is intentionally modular: both the score family and the
conditioning property can be extended.  For example, running with
``--property density`` reproduces the behaviour described in the user
specification, whereas ``--property deficiency`` conditions on the
controllability deficiency obtained via :func:`draw_with_ctrb_rank`.

Outputs
-------
* ``scores_summary.csv`` – aggregated moments (count, mean, std) per score.
* ``systems.csv`` – bookkeeping for each sampled system (e.g. realised
  density, controllability rank).
* One plot per score showing mean ± standard deviation against the
  conditioning variable.

Example
-------
```
python -m experiments.sim3 --n 6 --m 2 --samples 200 --x0-samples 1000 \
    --property density --cond-grid 0:0.05:1 --outdir results/sim3_density
```
"""

from __future__ import annotations

import argparse
import math
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

try:  # pragma: no cover - import guard for optional dependency
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without matplotlib
    plt = None  # type: ignore[assignment]
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None
import numpy as np
import pandas as pd

from ..ensembles import (
    sparse_continuous,
    draw_with_ctrb_rank,
    controllability_rank,
)
from ..metrics import (
    pbh_margin_structured,
    unified_generator,
    left_eigvec_overlap,
)


EPS = 1e-12
DEFAULT_SCORES: tuple[str, ...] = ("pbh", "krylov", "mu")


def sample_unit_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform sample on the unit sphere S^{n-1}."""

    v = rng.standard_normal(n)
    nrm = float(np.linalg.norm(v))
    return v / (nrm if nrm > 0.0 else 1.0)


def matrix_density(M: np.ndarray, tol: float = 0.0) -> float:
    """Fraction of entries whose magnitude exceeds ``tol``."""

    return float(np.mean(np.abs(M) > tol))


def compute_scores(
    A: np.ndarray,
    B: np.ndarray,
    x0: np.ndarray,
    score_names: Sequence[str],
) -> Mapping[str, float]:
    """Evaluate selected x0-based empirical scores."""

    out: dict[str, float] = {}
    if not score_names:
        return out

    if "pbh" in score_names:
        #margin = float(pbh_margin_structured(A, B, x0))
        #out["pbh"] = max(margin, EPS)
        out["pbh"] = float(pbh_margin_structured(A, B, x0))

    if "krylov" in score_names:
        K = unified_generator(A, B, x0, mode="unrestricted")
        svals = np.linalg.svd(K, compute_uv=False)
        smin = float(svals.min()) if svals.size else 0.0
        out["krylov"] = max(smin, EPS)

    if "mu" in score_names:
        Xaug = np.concatenate([x0.reshape(-1, 1), B], axis=1)
        mu = left_eigvec_overlap(A, Xaug)
        mu_min = float(np.min(mu)) if mu.size else 0.0
        out["mu"] = max(mu_min, EPS)

    return out


@dataclass
class Moment:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.total_sq += value * value

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")

    @property
    def std(self) -> float:
        if self.count == 0:
            return float("nan")
        mu = self.mean
        var = max(self.total_sq / self.count - mu * mu, 0.0)
        return math.sqrt(var)


def parse_grid(spec: str) -> np.ndarray:
    """Parse grid specifications like ``0:0.05:1`` or comma-separated values."""

    spec = spec.strip()
    if not spec:
        raise ValueError("conditioning grid specification cannot be empty")
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError("range grid must have the form start:step:stop")
        start, step, stop = map(float, parts)
        if step <= 0:
            raise ValueError("grid step must be positive")
        count = int(math.floor((stop - start) / step + 0.5)) + 1
        return start + step * np.arange(count)
    tokens = spec.replace(",", " ").split()
    values = [float(tok) for tok in tokens if tok]
    if not values:
        raise ValueError("no values parsed from grid specification")
    return np.asarray(values, dtype=float)


def ensure_output_dir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "plots").mkdir(exist_ok=True)
    return path


def generate_system(
    property_name: str,
    property_value: float,
    n: int,
    m: int,
    rng: np.random.Generator,
    *,
    sparse_which: str,
    sparse_tol: float,
    deficiency_base: str,
    deficiency_embed_random: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Draw a system according to the chosen conditioning property."""

    if property_name == "density":
        A, B = sparse_continuous(
            n=n,
            m=m,
            rng=rng,
            which=sparse_which,
            p_density=float(property_value),
        )
        meta = {
            "target_density": float(property_value),
            "density_A": matrix_density(A, tol=sparse_tol),
            "density_B": matrix_density(B, tol=sparse_tol),
            "density_AB": matrix_density(np.hstack([A, B]), tol=sparse_tol),
        }
        return A, B, meta

    if property_name == "deficiency":
        deficiency = int(round(property_value))
        rank_target = max(n - deficiency, 0)
        A, B, meta = draw_with_ctrb_rank(
            n=n,
            m=m,
            r=rank_target,
            rng=rng,
            ensemble_type=deficiency_base,
            base_u=deficiency_base,
            embed_random_basis=deficiency_embed_random,
        )
        rk, _ = controllability_rank(A, B)
        meta_info = {
            "target_deficiency": deficiency,
            "achieved_rank": int(rk),
            "achieved_deficiency": int(n - rk),
        }
        meta_info.update(meta)
        return A, B, meta_info

    raise ValueError(f"unsupported property '{property_name}'")


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    outdir = ensure_output_dir(pathlib.Path(args.outdir))

    conditioning_values = parse_grid(args.cond_grid)
    if args.scores is None:
        score_names = DEFAULT_SCORES
    else:
        if not args.scores:
            raise ValueError("--scores was provided but no score names were given")
        score_names = tuple(dict.fromkeys(args.scores))

    # Pre-sample x0's to reuse across systems for lower variance estimates.
    x0_samples = [sample_unit_sphere(args.n, rng) for _ in range(args.x0_samples)]

    # Moment accumulators: {(score, property_value) -> Moment}
    #accum: dict[tuple[str, float], Moment] = defaultdict(Moment)
    accum: dict[tuple[str, float], tuple[Moment, list[float]]] = defaultdict(lambda: (Moment(), []))
    system_records: list[dict] = []

    valid_scores = set(DEFAULT_SCORES)
    invalid = set(score_names) - valid_scores
    if invalid:
        raise ValueError(f"unknown scores requested: {sorted(invalid)}")

    for prop_value in conditioning_values:
        for system_idx in range(args.samples):
            A, B, meta = generate_system(
                args.property,
                prop_value,
                args.n,
                args.m,
                rng,
                sparse_which=args.sparse_which,
                sparse_tol=args.sparse_tol,
                deficiency_base=args.deficiency_base,
                deficiency_embed_random=not args.deficiency_no_embed,
            )

            sys_record = {
                "property": args.property,
                "property_value": float(prop_value),
                "system_index": len(system_records),
            }
            for key, value in meta.items():
                if np.isscalar(value):
                    sys_record[f"meta_{key}"] = value
            system_records.append(sys_record)

            for x0_idx, x0 in enumerate(x0_samples):
                scores = compute_scores(A, B, x0, score_names)
                for score_name, value in scores.items():
                    key = (score_name, float(prop_value))
                    mom, buf = accum[key]
                    mom.update(float(value))
                    buf.append(float(value))
                    accum[key] = (mom, buf)

    # Summaries -------------------------------------------------------------
    summary_rows = []
    for (score_name, prop_value), (moment, buf) in accum.items():
        q05 = np.nan
        q50 = np.nan
        q95 = np.nan
        if buf:
            arr = np.array(buf, dtype=float)
            q05, q50, q95 = np.quantile(arr, [0.05, 0.5, 0.95])
        summary_rows.append(
            {
                "score": score_name,
                "property": args.property,
                "property_value": prop_value,
                "count": moment.count,
                "mean": moment.mean,
                "std": moment.std,
                "q05": q05,
                "q50": q50,
                "q95": q95,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["score", "property_value"])
    summary_path = outdir / "scores_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    systems_df = pd.DataFrame(system_records)
    systems_df.to_csv(outdir / "systems.csv", index=False)

    # Plots ----------------------------------------------------------------
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plotting; please install it to run this experiment"
        ) from _MATPLOTLIB_IMPORT_ERROR

    xlabel = {
        "density": "Target density level",
        "deficiency": "Controllability deficiency",
    }.get(args.property, args.property)

    for score_name in summary_df["score"].unique():
        sub = summary_df[summary_df["score"] == score_name]
        if sub.empty:
            continue
        sub = sub.sort_values("property_value")

        x = sub["property_value"].to_numpy()
        mean = sub["mean"].to_numpy()
        std = sub["std"].to_numpy()

        plt.figure(figsize=(6.4, 4.0))
        plt.plot(x, mean, marker="o", label=f"{score_name} mean")
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, label="±1 std")
        plt.xlabel(xlabel)
        plt.ylabel(f"{score_name} score")
        plt.title(f"{score_name} vs {args.property}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plot_path = outdir / "plots" / f"{score_name}_vs_{args.property}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=6, help="state dimension")
    parser.add_argument("--m", type=int, default=2, help="input dimension")
    parser.add_argument(
        "--samples", type=int, default=200, help="number of (A,B) draws per conditioning value"
    )
    parser.add_argument(
        "--x0-samples", type=int, default=1000, help="number of x0 draws reused for every system"
    )
    parser.add_argument(
        "--scores",
        nargs="*",
        choices=DEFAULT_SCORES,
        default=None,
        metavar="SCORE",
        help="score names to compute (subset of: %s)"
        % ", ".join(DEFAULT_SCORES),
    )
    parser.add_argument(
        "--property",
        default="density",
        choices=["density", "deficiency"],
        help="system property to condition on",
    )
    parser.add_argument(
        "--cond-grid",
        default="0:0.05:1",
        help="conditioning grid specification (e.g., '0:0.05:1' or '0,0.5,1')",
    )
    parser.add_argument("--seed", type=int, default=12345, help="base RNG seed")
    parser.add_argument("--outdir", default="results/sim3", help="output directory")

    # Sparse ensemble controls (for density property)
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

    # Deficiency property controls
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
