"""Empirical score distributions conditioned on system properties.

```
# Single-axis sparsity sweep (legacy mode)
python -m pyident.experiments.sim_regcomb --n 6 --m 2 --samples 200 --x0-samples 1000 \
    --property density --cond-grid 0:0.05:1 --outdir results/sim3_density

# Full sweep (One axis)
# S1: Sparsity
python -m pyident.experiments.sim_regcomb --axes "sparsity" \
    --sparsity-grid 0.0:0.1:1.0 --samples 100 \
    --x0-samples 100 --outdir results/sim3_sparse

# S2: State dimension
python -m pyident.experiments.sim_regcomb --axes "ndim" \
    --ndim-grid 2:2:20 --samples 100 \
    --x0-samples 100 --outdir results/sim3_state

# S3: Underactuation
python -m pyident.experiments.sim_regcomb --axes "underactuation" \
     --samples 100 --x0-samples 100 --outdir results/sim3_underactuation

# Full sweeps (Two axes)
# E1: Sparsity vs. state dimension
python -m pyident.experiments.sim_regcomb --axes "sparsity, ndim" \
    --sparsity-grid 0.0:0.1:1.0 --ndim-grid 2:2:20 --samples 100 \
    --x0-samples 100 --outdir results/sim3_sparse_state

# E2: State dimension vs. underactuation
python -m pyident.experiments.sim_regcomb --axes "ndim, underactuation" \
    --ndim-grid 2:2:20 --samples 100 \
    --x0-samples 100 --outdir results/sim3_state_underactuation

# E3: Underaction vs. sparsity
python -m pyident.experiments.sim_regcomb --axes "underactuation, sparsity" \
        --sparsity-grid 0.0:0.1:1.0 --samples 100 \
        --x0-samples 100 --outdir results/sim3_underactuation_sparsity

# Smaller runs for faster testing
# E1: Sparsity vs. state dimension
python -m pyident.experiments.sim_regcomb --axes "sparsity, ndim" \
    --sparsity-grid 0.0:0.2:1.0 --ndim-grid 2:4:20 --samples 10 \
    --x0-samples 10 --outdir results/sim3_sparse_state

# E2: State dimension vs. underactuation
python -m pyident.experiments.sim_regcomb --axes "ndim, underactuation" \
    --ndim-grid 2:4:20 --samples 10 \
    --x0-samples 10 --outdir results/sim3_state_underactuation

# E3: Underaction vs. sparsity
python -m pyident.experiments.sim_regcomb --axes "underactuation, sparsity" \
        --sparsity-grid 0.0:0.2:1.0 --samples 10 \
        --x0-samples 10 --outdir results/sim3_underactuation_sparsity
```
"""
from __future__ import annotations

import argparse
import math
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Any, Mapping, Sequence, Tuple

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

SCORE_DISPLAY_NAMES = {
    "pbh": "PBH score",
    "krylov": "Krylov score",
    "mu": "Left eigenvalue score",
}

AXIS_TITLE_NAMES = {
    "sparsity": "Sparsity",
    "ndim": "State Dimension",
    "underactuation": "Underactuation",
}

AXIS_ALIASES = {
    "sparsity": "sparsity",
    "sparse": "sparsity",
    "density": "sparsity",
    "ndim": "ndim",
    "xdim": "ndim",
    "state_dimension": "ndim",
    "underactuation": "underactuation",
    "undera": "underactuation",
}

AXIS_COLUMN = {
    "sparsity": "axis_sparsity",
    "ndim": "axis_ndim",
    "underactuation": "axis_underactuation",
}

AXIS_LABEL = {
    "sparsity": "Density level",
    "ndim": "State dimension n",
    "underactuation": "Input dimension m",
}

DEFAULT_STATE_INPUT_VALUES = tuple(range(2, 21, 2))

def format_axis_tick(axis_name: str, value: Any) -> str:
    """Format axis tick labels for heatmaps."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    if axis_name == "sparsity":
        return f"{numeric:.1f}"
    if axis_name in {"ndim", "underactuation"} and math.isclose(
        numeric, round(numeric), rel_tol=0.0, abs_tol=1e-9
    ):
        return f"{int(round(numeric))}"
    return f"{numeric:g}"


def make_heatmap_title(
    x_axis: str, y_axis: str, score_name: str, data: pd.DataFrame
) -> str:
    """Construct a descriptive heatmap title for score summaries."""

    score_label = SCORE_DISPLAY_NAMES.get(score_name, score_name)
    x_label = AXIS_TITLE_NAMES.get(x_axis, x_axis.title())
    y_label = AXIS_TITLE_NAMES.get(y_axis, y_axis.title())
    base_title = f"{x_label} vs. {y_label}"

    if (x_axis, y_axis) == ("underactuation", "sparsity") and "n" in data:
        n_values = sorted({int(val) for val in data["n"].dropna().unique()})
        if n_values:
            if len(n_values) == 1:
                suffix = f", n = {n_values[0]}"
            else:
                suffix = ", n ∈ {" + ", ".join(str(val) for val in n_values) + "}"
            return f"{base_title} ({score_label}{suffix})"

    return f"{base_title} ({score_label})"

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

    if not score_names:
        return {}
    
    score_set = set(score_names)
    scores: dict[str, float] = {}

    if "pbh" in score_set:
        scores["pbh"] = float(pbh_margin_structured(A, B, x0))

    if "krylov" in score_set:
        K = unified_generator(A, B, x0, mode="unrestricted")
        svals = np.linalg.svd(K, compute_uv=False)
        smin = float(svals.min()) if svals.size else 0.0
        scores["krylov"] = max(smin, EPS)

    if "mu" in score_set:
        Xaug = np.concatenate([x0.reshape(-1, 1), B], axis=1)
        mu = left_eigvec_overlap(A, Xaug)
        mu_min = float(np.min(mu)) if mu.size else 0.0
        scores["mu"] = max(mu_min, EPS)

    return scores


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
        if stop < start:
            raise ValueError("grid stop must not be smaller than the start value")

        # ``np.arange`` can accumulate floating point error and produce values that
        # overshoot the requested ``stop`` bound (for example 0.1:0.2:1.0 yields
        # 1.1).  This is problematic when the grid is used to parameterise
        # probabilities such as sparsity densities.  Build the sequence manually
        # and clamp it to the stop value instead.
        values: list[float] = []
        current = start
        # Allow a small tolerance when comparing against ``stop`` so that values
        # very close to the end-point (e.g. 0.6 + 0.2 ≈ 0.7999999999999999) are
        # still included.
        tol = 1e-12 * max(1.0, abs(stop))
        while current <= stop + tol:
            values.append(float(current))
            current += step
        return np.asarray(values, dtype=float)
    
    tokens = spec.replace(",", " ").split()
    values = [float(tok) for tok in tokens if tok]
    if not values:
        raise ValueError("no values parsed from grid specification")
    return np.asarray(values, dtype=float)


def ensure_output_dir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "plots").mkdir(exist_ok=True)
    return path

def parse_axes_spec(spec: str | None) -> tuple[str, ...]:
    if not spec:
        return ()
    tokens = [tok.strip().lower() for tok in spec.replace(",", " ").split() if tok.strip()]
    if not tokens:
        raise ValueError("--axes was provided but no axis names were found")
    axes: list[str] = []
    for tok in tokens:
        if tok not in AXIS_ALIASES:
            raise ValueError(f"unknown axis '{tok}'")
        canon = AXIS_ALIASES[tok]
        if canon in axes:
            raise ValueError(f"duplicate axis '{tok}' in specification")
        axes.append(canon)
    if len(axes) > 2:
        raise ValueError("at most two axes can be combined")
    return tuple(axes)


def underactuation_grid(n_dim: int) -> list[int]:
    if n_dim <= 0:
        raise ValueError("state dimension must be positive when using the underactuation axis")
    step = max(1, n_dim // 8)
    values = list(range(1, n_dim + 1, step))
    if values[-1] != n_dim:
        values.append(n_dim)
    return sorted(set(values))


def freeze_items(mapping: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """Create a hashable representation of a dictionary."""

    return tuple(sorted(mapping.items()))


def generate_system(
    property_name: str,
    property_value: float,
    n: int,
    m: int,
    rng: np.random.Generator,
    *,
    sparse_which: str,
    sparse_tol: float,
    base_density: float,
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

    if property_name == "state_dimension":
        A, B = sparse_continuous(
            n=n,
            m=m,
            rng=rng,
            which=sparse_which,
            p_density=float(base_density),
        )
        meta = {
            "density_A": matrix_density(A, tol=sparse_tol),
            "density_B": matrix_density(B, tol=sparse_tol),
            "density_AB": matrix_density(np.hstack([A, B]), tol=sparse_tol),
        }
        return A, B, meta

    if property_name == "underactuation":
        A, B = sparse_continuous(
            n=n,
            m=m,
            rng=rng,
            which=sparse_which,
            p_density=float(base_density),
        )
        meta = {
            "density_A": matrix_density(A, tol=sparse_tol),
            "density_B": matrix_density(B, tol=sparse_tol),
            "density_AB": matrix_density(np.hstack([A, B]), tol=sparse_tol),
            "underactuation": float(n - m),
            "input_fraction": float(m) / float(n) if n else float("nan"),
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


def build_axis_scenarios(
    args: argparse.Namespace, axes: Iterable[str]
) -> list[tuple[str, float, int, int, dict[str, Any]]]:
    axes = tuple(axes)
    results: list[tuple[str, float, int, int, dict[str, Any]]] = []
    base_n = int(args.n)
    base_m = int(args.m)

    def append(
        property_name: str,
        property_value: float,
        n_cur: int,
        m_cur: int,
        axis_values: Mapping[str, Any],
    ) -> None:
        info: dict[str, Any] = {
            "property": property_name,
            "property_value": float(property_value),
            "axes": ",".join(axes),
            "n": n_cur,
            "m": m_cur,
        }
        for axis_name, axis_value in axis_values.items():
            info[AXIS_COLUMN[axis_name]] = axis_value
        if "underactuation" in axes:
            info.setdefault("underactuation_level", float(n_cur - m_cur))
            info.setdefault(
                "input_fraction", float(m_cur) / float(n_cur) if n_cur else float("nan")
            )
        if "ndim" in axes:
            info.setdefault("state_dimension", int(n_cur))
        if "sparsity" in axes:
            info.setdefault("target_density", float(axis_values.get("sparsity", property_value)))
        results.append((property_name, float(property_value), n_cur, m_cur, info))

    axis_set = set(axes)
    if len(axes) == 1:
        axis = axes[0]
        if axis == "sparsity":
            grid_spec = args.sparsity_grid or args.cond_grid
            if grid_spec is None:
                raise ValueError("a sparsity grid must be provided for the sparsity axis")
            densities = parse_grid(grid_spec)
            for dens in densities:
                append(
                    "density",
                    float(dens),
                    base_n,
                    base_m,
                    {"sparsity": float(dens)},
                )
        elif axis == "ndim":
            grid_spec = args.ndim_grid or args.n_grid or args.cond_grid
            if grid_spec is None:
                raise ValueError("an n-dimension grid must be provided for the ndim axis")
            n_values = parse_grid(grid_spec)
            for n_val in n_values:
                n_cur = int(round(n_val))
                if n_cur <= 0:
                    raise ValueError("state dimensions must be positive integers")
                m_cur = min(base_m, n_cur)
                append("state_dimension", float(n_cur), n_cur, m_cur, {"ndim": n_cur})
        elif axis == "underactuation":
            m_values = underactuation_grid(base_n)
            for m_cur in m_values:
                append(
                    "underactuation",
                    float(base_n - m_cur),
                    base_n,
                    m_cur,
                    {"underactuation": m_cur},
                )
        else:
            raise ValueError(f"unsupported axis '{axis}'")
        return results

    if axis_set == {"sparsity", "ndim"}:
        dens_spec = args.sparsity_grid or args.cond_grid
        nd_spec = args.ndim_grid or args.n_grid
        if dens_spec is None:
            raise ValueError("a sparsity grid must be provided when combining axes")
        if nd_spec is None:
            raise ValueError("an n-dimension grid must be provided when combining axes")
        densities = parse_grid(dens_spec)
        n_values = parse_grid(nd_spec)
        for dens in densities:
            for n_val in n_values:
                n_cur = int(round(n_val))
                if n_cur <= 0:
                    raise ValueError("state dimensions must be positive integers")
                m_cur = min(base_m, n_cur)
                axis_vals = {"sparsity": float(dens), "ndim": n_cur}
                append("density", float(dens), n_cur, m_cur, axis_vals)
        return results

    if axis_set == {"sparsity", "underactuation"}:
        dens_spec = args.sparsity_grid or args.cond_grid
        if dens_spec is None:
            raise ValueError("a sparsity grid must be provided when combining axes")
        densities = parse_grid(dens_spec)
        m_values = underactuation_grid(base_n)
        for dens in densities:
            for m_cur in m_values:
                axis_vals = {"sparsity": float(dens), "underactuation": m_cur}
                append("density", float(dens), base_n, m_cur, axis_vals)
        return results

    if axis_set == {"ndim", "underactuation"}:
        nd_spec = args.ndim_grid or args.n_grid or args.cond_grid
        if nd_spec is None:
            n_values = np.array(DEFAULT_STATE_INPUT_VALUES, dtype=float)
        else:
            n_values = parse_grid(nd_spec)
        default_m_values = sorted(
            {int(val) for val in DEFAULT_STATE_INPUT_VALUES} | {int(base_m)}
        )
        for n_val in n_values:
            n_cur = int(round(n_val))
            if n_cur <= 0:
                raise ValueError("state dimensions must be positive integers")
            m_values = default_m_values
            for m_cur in m_values:
                axis_vals = {"ndim": n_cur, "underactuation": m_cur}
                append("underactuation", float(n_cur - m_cur), n_cur, m_cur, axis_vals)
        return results

    raise ValueError(f"unsupported axis combination: {axes}")



def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    outdir = ensure_output_dir(pathlib.Path(args.outdir))

    if args.scores is None:
        score_names = DEFAULT_SCORES
    else:
        if not args.scores:
            raise ValueError("--scores was provided but no score names were given")
        score_names = tuple(dict.fromkeys(args.scores))


    x0_cache: dict[int, list[np.ndarray]] = {}

    # Moment accumulators: {(score, property_value) -> Moment}
    def get_x0_samples(n_dim: int) -> list[np.ndarray]:
        samples = x0_cache.get(n_dim)
        if samples is None:
            samples = [sample_unit_sphere(n_dim, rng) for _ in range(args.x0_samples)]
            x0_cache[n_dim] = samples
        return samples

    # Moment accumulators keyed by (score, frozen_property_info)
    accum: dict[tuple[str, Tuple[Tuple[str, Any], ...]], tuple[Moment, list[float]]] = defaultdict(
        lambda: (Moment(), [])
    )
    system_records: list[dict] = []

    valid_scores = set(DEFAULT_SCORES)
    invalid = set(score_names) - valid_scores
    if invalid:
        raise ValueError(f"unknown scores requested: {sorted(invalid)}")
    axes = parse_axes_spec(args.axes)
    if axes:
        if args.property is not None and args.property != "density":
            # ``--property`` is kept for legacy workflows; avoid confusion when
            # the new multi-axis mode is used.
            raise ValueError("--property cannot be combined with --axes")
        scenarios = build_axis_scenarios(args, axes)
    else:
        property_name = args.property
        scenarios: list[tuple[str, float, int, int, dict[str, Any]]] = []
        if property_name == "underactuation":
            n_grid_spec = args.n_grid or args.cond_grid
            if n_grid_spec is None:
                raise ValueError("--cond-grid or --n-grid must be provided for property=underactuation")
            if args.m_grid is None:
                raise ValueError("--m-grid must be provided for property=underactuation")
            n_values = parse_grid(n_grid_spec)
            m_values = parse_grid(args.m_grid)
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
                        "m": args.m,
                        "input_fraction": float(args.m) / float(n_cur),
                    }
                scenarios.append((property_name, prop_val, n_cur, args.m, info))
        else:
            conditioning_values = parse_grid(args.cond_grid)
            for prop_value in conditioning_values:
                info = {
                    "property": property_name,
                    "property_value": float(prop_value),
                    "n": args.n,
                    "m": args.m,
                }
                scenarios.append((property_name, float(prop_value), args.n, args.m, info))

    if not scenarios:
        raise ValueError("no scenarios were generated; please check the grid specifications")

    for property_name, prop_value, n_cur, m_cur, prop_info in scenarios:
        prop_key = freeze_items(prop_info)
        x0_samples = get_x0_samples(n_cur)

        for _ in range(args.samples):
            A, B, meta = generate_system(
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

            sys_record = {
                "system_index": len(system_records),
                "n": n_cur,
                "m": m_cur,
            }
            for key, value in prop_info.items():
                if np.isscalar(value):
                    sys_record[key] = value

            # Ensure meta captures realised densities and actuation metrics.
            meta.setdefault("density_A", matrix_density(A, tol=args.sparse_tol))
            meta.setdefault("density_B", matrix_density(B, tol=args.sparse_tol))
            meta.setdefault("density_AB", matrix_density(np.hstack([A, B]), tol=args.sparse_tol))
            meta.setdefault("underactuation", float(n_cur - m_cur))
            meta.setdefault(
                "input_fraction", float(m_cur) / float(n_cur) if n_cur else float("nan")
            )
            for key, value in meta.items():
                if np.isscalar(value):
                    sys_record[f"meta_{key}"] = value

            system_records.append(sys_record)

            for x0 in x0_samples:
                scores = compute_scores(A, B, x0, score_names)
                for score_name, value in scores.items():
                    key = (score_name, prop_key)
                    mom, buf = accum[key]
                    mom.update(float(value))
                    buf.append(float(value))
                    accum[key] = (mom, buf)

    # Summaries -------------------------------------------------------------
    summary_rows = []
    for (score_name, prop_key), (moment, buf) in accum.items():
        q05 = np.nan
        q50 = np.nan
        q95 = np.nan
        if buf:
            arr = np.array(buf, dtype=float)
            q05, q50, q95 = np.quantile(arr, [0.05, 0.5, 0.95])
        prop_info = dict(prop_key)
        summary_rows.append(
            {
                "score": score_name,
                **prop_info,
                "count": moment.count,
                "mean": moment.mean,
                "std": moment.std,
                "q05": q05,
                "q50": q50,
                "q95": q95,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        raise RuntimeError("no summary statistics were computed; check the score configuration")

    sort_cols = ["score"]
    for col in ["property_value", "n", "m", *AXIS_COLUMN.values()]:
        if col in summary_df.columns:
            sort_cols.append(col)
    summary_df = summary_df.sort_values(sort_cols)
    summary_path = outdir / "scores_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    systems_df = pd.DataFrame(system_records)
    systems_df.to_csv(outdir / "systems.csv", index=False)

    # Plots ----------------------------------------------------------------
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plotting; please install it to run this experiment"
        ) from _MATPLOTLIB_IMPORT_ERROR

    if axes:
        axis_columns = [AXIS_COLUMN[a] for a in axes]
        if len(axes) == 2:
            x_axis, y_axis = axis_columns[0], axis_columns[1]
            x_label = AXIS_LABEL[axes[0]]
            y_label = AXIS_LABEL[axes[1]]
            heat_threshold = float(getattr(args, "heatthr", 1e-12))
            for score_name in summary_df["score"].unique():
                sub = summary_df[summary_df["score"] == score_name]
                if sub.empty or x_axis not in sub.columns or y_axis not in sub.columns:
                    continue
                pivot = sub.pivot(index=y_axis, columns=x_axis, values="mean")
                pivot = pivot.sort_index().sort_index(axis=1)
                if pivot.empty:
                    continue

                data = pivot.to_numpy()
                n_rows, n_cols = data.shape
                fig, ax = plt.subplots(figsize=(7.2, 5.2))
                im = ax.imshow(
                    data,
                    origin="lower",
                    aspect="auto",
                    extent=(-0.5, n_cols - 0.5, -0.5, n_rows - 0.5),
                )
                ax.set_xlim(-0.5, n_cols - 0.5)
                ax.set_ylim(-0.5, n_rows - 0.5)
                ax.set_xticks(np.arange(n_cols))
                ax.set_yticks(np.arange(n_rows))
                ax.set_xticklabels(
                    [format_axis_tick(axes[0], value) for value in pivot.columns]
                )
                ax.set_yticklabels(
                    [format_axis_tick(axes[1], value) for value in pivot.index]
                )

                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)

                ax.set_title(make_heatmap_title(axes[0], axes[1], score_name, sub))
                special_state_under = axes[0] == "ndim" and axes[1] == "underactuation"
                if special_state_under and pivot.size:
                    column_values = list(pivot.columns)
                    row_values = list(pivot.index)
                    diag_coords: list[tuple[float, float]] = []
                    for col_idx, col_val in enumerate(column_values):
                        for row_idx, row_val in enumerate(row_values):
                            if math.isclose(float(col_val), float(row_val), rel_tol=0.0, abs_tol=1e-9):
                                diag_coords.append((float(col_idx), float(row_idx)))
                                break
                    if diag_coords:
                        xs, ys = zip(*diag_coords)
                        ax.plot(xs, ys, color="red", linewidth=2.0, solid_capstyle="round")
                        x_start, x_end = xs[0], xs[-1]
                        y_start, y_end = ys[0], ys[-1]
                        x_mid = 0.5 * (x_start + x_end)
                        y_mid = 0.5 * (y_start + y_end)
                        if len(xs) > 1:
                            rotation = math.degrees(math.atan2(y_end - y_start, x_end - x_start))
                        else:
                            rotation = 45.0
                        ax.text(
                            x_mid,
                            y_mid - 0.35,
                            "n = m",
                            color="red",
                            fontsize=8,
                            rotation=rotation,
                            rotation_mode="anchor",
                            ha="center",
                            va="center",
                        )
                cbar = fig.colorbar(im, ax=ax, pad=0.02)          
                cbar.set_label(f"{score_name} score (mean)")
                fig.tight_layout()
                fig.canvas.draw()
                if special_state_under and pivot.size:
                    #arrowprops = dict(
                    #    arrowstyle="-|>", color="red", linewidth=2.4, mutation_scale=12
                    #)
                    ax_pos = ax.get_position()
                    cbar_pos = cbar.ax.get_position()
                    horizontal_y = ax_pos.y0 - 0.07
                    horizontal_start = ax_pos.x0 + 0.02 * ax_pos.width
                    horizontal_end = ax_pos.x1 - 0.02 * ax_pos.width
                    ax.annotate(
                        "",
                        xy=(horizontal_end, horizontal_y),
                        xytext=(horizontal_start, horizontal_y),
                        xycoords=fig.transFigure,
                        textcoords=fig.transFigure,
                        #arrowprops=arrowprops,
                        annotation_clip=False,
                    )
                    vertical_x = cbar_pos.x1 + 0.02
                    vertical_top = min(cbar_pos.y1, 0.96)
                    vertical_bottom = max(cbar_pos.y0, 0.08)
                    ax.annotate(
                        "",
                        xy=(vertical_x, vertical_bottom),
                        xytext=(vertical_x, vertical_top),
                        xycoords=fig.transFigure,
                        textcoords=fig.transFigure,
                        #arrowprops=arrowprops,
                        annotation_clip=False,
                    )               
                name = f"{score_name}_heatmap_{axes[0]}_{axes[1]}".replace(",", "_")
                plot_path = outdir / "plots" / f"{name}.png"
                fig.savefig(plot_path, dpi=200)
                base_cmap = im.get_cmap()
                base_norm = im.norm               
                plt.close(fig)

 
                thr_fig, thr_ax = plt.subplots(figsize=(7.2, 5.2))
                thr_im = thr_ax.imshow(
                    data,
                    origin="lower",
                    aspect="auto",
                    extent=(-0.5, n_cols - 0.5, -0.5, n_rows - 0.5),
                    cmap=base_cmap,
                    norm=base_norm,
                )
                thr_ax.set_xlim(-0.5, n_cols - 0.5)
                thr_ax.set_ylim(-0.5, n_rows - 0.5)
                thr_ax.set_xticks(np.arange(n_cols))
                thr_ax.set_yticks(np.arange(n_rows))
                thr_ax.set_xticklabels(
                    [format_axis_tick(axes[0], value) for value in pivot.columns]
                )
                thr_ax.set_yticklabels(
                    [format_axis_tick(axes[1], value) for value in pivot.index]
                )

                thr_ax.set_xlabel(x_label)
                thr_ax.set_ylabel(y_label)

                threshold_title = make_heatmap_title(axes[0], axes[1], score_name, sub)
                thr_ax.set_title(
                    f"{threshold_title} (red < {heat_threshold:.1e})"
                )

                if special_state_under and pivot.size:
                    column_values = list(pivot.columns)
                    row_values = list(pivot.index)
                    diag_coords: list[tuple[float, float]] = []
                    for col_idx, col_val in enumerate(column_values):
                        for row_idx, row_val in enumerate(row_values):
                            if math.isclose(float(col_val), float(row_val), rel_tol=0.0, abs_tol=1e-9):
                                diag_coords.append((float(col_idx), float(row_idx)))
                                break
                    if diag_coords:
                        xs, ys = zip(*diag_coords)
                        thr_ax.plot(xs, ys, color="red", linewidth=2.0, solid_capstyle="round")
                        x_start, x_end = xs[0], xs[-1]
                        y_start, y_end = ys[0], ys[-1]
                        x_mid = 0.5 * (x_start + x_end)
                        y_mid = 0.5 * (y_start + y_end)
                        if len(xs) > 1:
                            rotation = math.degrees(math.atan2(y_end - y_start, x_end - x_start))
                        else:
                            rotation = 45.0
                        thr_ax.text(
                            x_mid,
                            y_mid - 0.35,
                            "n = m",
                            color="red",
                            fontsize=8,
                            rotation=rotation,
                            rotation_mode="anchor",
                            ha="center",
                            va="center",
                        )

                if not math.isnan(heat_threshold):
                    mask = data < heat_threshold
                else:
                    mask = np.zeros_like(data, dtype=bool)
                if np.any(mask):
                    red_overlay = np.zeros((n_rows, n_cols, 4), dtype=float)
                    red_overlay[mask] = (1.0, 0.0, 0.0, 1.0)
                    thr_ax.imshow(
                        red_overlay,
                        origin="lower",
                        aspect="auto",
                        extent=(-0.5, n_cols - 0.5, -0.5, n_rows - 0.5),
                    )

                thr_cbar = thr_fig.colorbar(thr_im, ax=thr_ax, pad=0.02)
                thr_cbar.set_label(
                    f"{score_name} score (mean; red < {heat_threshold:.1e})"
                )
                thr_fig.tight_layout()
                thr_name = f"{score_name}_heatmap_{axes[0]}_{axes[1]}_thr".replace(",", "_")
                thr_path = outdir / "plots" / f"{thr_name}.png"
                thr_fig.savefig(thr_path, dpi=200)
                plt.close(thr_fig)

                # Create a companion plot that visualizes descent directions with arrows.
                grad_y, grad_x = np.gradient(data) if data.size else (data, data)
                descent_x = -grad_x
                descent_y = -grad_y
                magnitudes = np.hypot(descent_x, descent_y)
                with np.errstate(divide="ignore", invalid="ignore"):
                    normalized_x = np.divide(
                        descent_x,
                        magnitudes,
                        out=np.zeros_like(descent_x),
                        where=magnitudes > 0.0,
                    )
                    normalized_y = np.divide(
                        descent_y,
                        magnitudes,
                        out=np.zeros_like(descent_y),
                        where=magnitudes > 0.0,
                    )

                arrow_length = 0.35
                arrow_dx = normalized_x * arrow_length
                arrow_dy = normalized_y * arrow_length

                arrow_fig, arrow_ax = plt.subplots(figsize=(7.2, 5.2))
                arrow_ax.set_xlim(-0.5, n_cols - 0.5)
                arrow_ax.set_ylim(-0.5, n_rows - 0.5)
                arrow_ax.set_facecolor("white")

                x_coords, y_coords = np.meshgrid(
                    np.arange(n_cols), np.arange(n_rows)
                )
                arrow_ax.quiver(
                    x_coords,
                    y_coords,
                    arrow_dx,
                    arrow_dy,
                    color="red",
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    width=0.01,
                    headwidth=4,
                    headlength=6,
                )
                arrow_ax.set_xticks(np.arange(n_cols))
                arrow_ax.set_yticks(np.arange(n_rows))
                arrow_ax.set_xticklabels(
                    [format_axis_tick(axes[0], value) for value in pivot.columns]
                )
                arrow_ax.set_yticklabels(
                    [format_axis_tick(axes[1], value) for value in pivot.index]
                )

                arrow_ax.set_xlabel(x_label)
                arrow_ax.set_ylabel(y_label)
                arrow_title = make_heatmap_title(axes[0], axes[1], score_name, sub)
                arrow_ax.set_title(f"{arrow_title} (descent directions)")

                if special_state_under and pivot.size:
                    column_values = list(pivot.columns)
                    row_values = list(pivot.index)
                    diag_coords: list[tuple[float, float]] = []
                    for col_idx, col_val in enumerate(column_values):
                        for row_idx, row_val in enumerate(row_values):
                            if math.isclose(float(col_val), float(row_val), rel_tol=0.0, abs_tol=1e-9):
                                diag_coords.append((float(col_idx), float(row_idx)))
                                break
                    if diag_coords:
                        xs, ys = zip(*diag_coords)
                        arrow_ax.plot(
                            xs, ys, color="red", linewidth=2.0, solid_capstyle="round"
                        )
                        x_start, x_end = xs[0], xs[-1]
                        y_start, y_end = ys[0], ys[-1]
                        x_mid = 0.5 * (x_start + x_end)
                        y_mid = 0.5 * (y_start + y_end)
                        if len(xs) > 1:
                            rotation = math.degrees(
                                math.atan2(y_end - y_start, x_end - x_start)
                            )
                        else:
                            rotation = 45.0
                        arrow_ax.text(
                            x_mid,
                            y_mid - 0.35,
                            "n = m",
                            color="red",
                            fontsize=8,
                            rotation=rotation,
                            rotation_mode="anchor",
                            ha="center",
                            va="center",
                        )

                arrow_fig.tight_layout()
                arrow_name = f"{score_name}_descent_{axes[0]}_{axes[1]}".replace(",", "_")
                arrow_path = outdir / "plots" / f"{arrow_name}.png"
                arrow_fig.savefig(arrow_path, dpi=200)
                plt.close(arrow_fig)

            return

        axis = axes[0]
        axis_col = axis_columns[0]
        axis_label = AXIS_LABEL[axis]

        for score_name in summary_df["score"].unique():
            sub = summary_df[summary_df["score"] == score_name]
            if sub.empty or axis_col not in sub.columns:
                continue

            sub = sub.sort_values(axis_col)
            x = sub[axis_col].to_numpy()
            mean = sub["mean"].to_numpy()
            std = sub["std"].to_numpy()

            plt.figure(figsize=(6.4, 4.0))
            plt.plot(x, mean, marker="o", label=f"{score_name} mean")
            plt.fill_between(x, mean - std, mean + std, alpha=0.2, label="±1 std")
            plt.xlabel(axis_label)
            plt.ylabel(f"{score_name} score")
            plt.title(f"{score_name} vs {axis}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plot_path = outdir / "plots" / f"{score_name}_vs_{axis}.png"

            plt.tight_layout()
            plt.savefig(plot_path, dpi=200)
            plt.close()
        return

    xlabel = {
        "density": "Density level",
        "deficiency": "Controllability deficiency",
        "state_dimension": "State dimension n",
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
    parser.add_argument("--n", type=int, default=10, help="state dimension")
    parser.add_argument("--m", type=int, default=2, help="input dimension")
    parser.add_argument(
        "--samples", type=int, default=200, help="number of (A,B) draws per conditioning value"
    )
    parser.add_argument(
        "--x0-samples", type=int, default=1000, help="number of x0 draws reused for every system"
    )
    parser.add_argument(
        "--axes",
        default=None,
        help="comma-separated list of axes to sweep (subset of: sparsity, ndim, underactuation)",
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
    parser.add_argument(
        "--sparse-density",
        type=float,
        default=0.3,
        help="baseline density used when property is not 'density'",
    )

    parser.add_argument(
        "--heatthr",
        type=float,
        default=1e-12,
        help=(
            "heatmap threshold: cells with mean scores below this value are coloured red"
        ),
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