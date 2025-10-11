from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import argparse
import json
import math
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..config import ExperimentConfig
from ..estimators import dmdc_tls
from ..metrics import build_visible_basis_dt, projected_errors
from ..projectors import projector_from_basis
from ..signals import estimate_pe_order
from ..simulation import prbs as prbs_dt
from ..simulation import simulate_dt
from .visible_sampling import VisibleDrawConfig, draw_system_state_with_visible_dim


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PEVisibilityConfig(ExperimentConfig):
    """Configuration for the PE-order versus visibility deficiency study."""

    n_grid: Sequence[int] = (4, 6, 8, 10, 12)
    d_max: int = 3
    m: int = 1
    dt: float = 0.05
    r_max_cap: int = 12
    beta_grid: Sequence[float] = (1.0, 1.5)
    N_sys: int = 10
    N_trials: int = 6
    sigma: float = 0.0
    tau: float = 0.05
    visible_tol: float = 1e-8
    deterministic_x0: bool = False
    max_system_attempts: int = 200
    max_x0_attempts: int = 200
    outdir: pathlib.Path = pathlib.Path("results/sim_pe2")
    eps_norm: float = 1e-12
    dwell: int = 1
    u_scale: float = 3.0

    def __post_init__(self) -> None:  # type: ignore[override]
        super().__post_init__()
        self.n_grid = tuple(int(n) for n in self.n_grid)
        self.beta_grid = tuple(float(b) for b in self.beta_grid)
        if not self.n_grid:
            raise ValueError("n_grid cannot be empty.")
        if not self.beta_grid:
            raise ValueError("beta_grid cannot be empty.")
        if self.d_max < 0:
            raise ValueError("d_max must be non-negative.")
        if self.r_max_cap <= 0:
            raise ValueError("r_max_cap must be positive.")
        if self.N_sys <= 0:
            raise ValueError("N_sys must be positive.")
        if self.N_trials <= 0:
            raise ValueError("N_trials must be positive.")
        if any(n <= 0 for n in self.n_grid):
            raise ValueError("n_grid entries must be positive integers.")
        if any(b <= 0.0 for b in self.beta_grid):
            raise ValueError("beta_grid entries must be positive.")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive.")
        if self.sigma < 0.0:
            raise ValueError("sigma must be non-negative.")
        if not isinstance(self.outdir, pathlib.Path):
            self.outdir = pathlib.Path(self.outdir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def minimal_length_for_pe_order(m: int, r: int) -> int:
    """A conservative lower bound on horizon length achieving block-PE order ``r``."""

    if r <= 0:
        raise ValueError("PE order r must be positive.")
    base = m * r + r - 1
    heuristic = r * (m * r + 5)
    return max(base, heuristic)


def gen_prbs_for_order(
    cfg: PEVisibilityConfig,
    r: int,
    beta: float,
    rng: np.random.Generator,
    max_attempts: int = 200,
) -> Tuple[np.ndarray, int]:
    """Generate a PRBS input that attains block-PE order ``r`` exactly."""

    N_min = minimal_length_for_pe_order(cfg.m, r)
    N = int(math.ceil(beta * N_min))
    step = max(1, N // 12)
    for attempt in range(max_attempts):
        U = prbs_dt(N, cfg.m, scale=cfg.u_scale, dwell=cfg.dwell, rng=rng)
        try:
            r_est = int(estimate_pe_order(U, s_max=r))
        except ValueError:
            r_est = 0
        if r_est == r:
            r_plus = int(estimate_pe_order(U, s_max=r + 1)) if r + 1 <= cfg.r_max_cap + 1 else r
            if r_plus < r + 1:
                return U, r_est
        N += step
        step = max(1, N // 12)
    # Fall back to best-effort generation by allowing relaxed condition
    U = prbs_dt(N, cfg.m, scale=cfg.u_scale, dwell=cfg.dwell, rng=rng)
    r_est = int(estimate_pe_order(U, s_max=r))
    return U, r_est


def _draw_system_with_visible_dim(
    n: int,
    d: int,
    cfg: PEVisibilityConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``(A, B, Ad, Bd, x0, Qv)`` with visible dimension ``n-d``."""

    k = n - d
    draw_cfg = VisibleDrawConfig(
        n=n,
        m=cfg.m,
        dt=cfg.dt,
        dim_visible=k,
        ensemble="stable",
        max_system_attempts=cfg.max_system_attempts,
        max_x0_attempts=cfg.max_x0_attempts,
        tol=cfg.visible_tol,
        deterministic_x0=cfg.deterministic_x0,
    )
    for _ in range(cfg.max_system_attempts):
        A, B, Ad, Bd, x0, Vbasis = draw_system_state_with_visible_dim(draw_cfg, rng)
        Qv = build_visible_basis_dt(Ad, Bd, x0, tol=cfg.visible_tol)
        if Qv.shape[1] == k:
            return A, B, Ad, Bd, x0, Qv
    raise RuntimeError(
        "Failed to draw a system whose discrete-time visible subspace matches the requested dimension."
    )


def _safe_rel(value: float, denom: float, eps: float) -> float:
    if denom <= eps:
        return 0.0 if abs(value) <= eps else float("nan")
    return value / denom


def _trial_record(
    cfg: PEVisibilityConfig,
    n: int,
    d: int,
    system_id: int,
    system_seed: int,
    beta: float,
    r_target: int,
    U: np.ndarray,
    r_actual: int,
    X0: np.ndarray,
    X1: np.ndarray,
    Ad: np.ndarray,
    Bd: np.ndarray,
    Qv: np.ndarray,
    tau: float,
    seed: int,
) -> Dict[str, float | int]:
    Ahat, Bhat = dmdc_tls(X0, X1, U)
    errA = float(np.linalg.norm(Ahat - Ad, ord="fro"))
    errB = float(np.linalg.norm(Bhat - Bd, ord="fro"))
    normA = float(np.linalg.norm(Ad, ord="fro"))
    normB = float(np.linalg.norm(Bd, ord="fro"))
    errA_rel = _safe_rel(errA, normA, cfg.eps_norm)
    errB_rel = _safe_rel(errB, normB, cfg.eps_norm)
    PV = projector_from_basis(Qv)
    errA_vis, errB_vis = projected_errors(Ahat, Bhat, Ad, Bd, Qv)
    normA_vis = float(np.linalg.norm(PV @ Ad @ PV, ord="fro"))
    normB_vis = float(np.linalg.norm(PV @ Bd, ord="fro"))
    errA_vis_rel = _safe_rel(errA_vis, normA_vis, cfg.eps_norm)
    errB_vis_rel = _safe_rel(errB_vis, normB_vis, cfg.eps_norm)
    unified = 0.5 * (errA_rel + errB_rel)
    unified_vis = 0.5 * (errA_vis_rel + errB_vis_rel)
    success = 1 if unified_vis <= tau else 0
    return {
        "seed": seed,
        "system_seed": system_seed,
        "system_id": system_id,
        "n": n,
        "k": n - d,
        "d": d,
        "m": cfg.m,
        "beta": beta,
        "sigma": cfg.sigma,
        "r_target": r_target,
        "r_actual": r_actual,
        "N": int(U.shape[0]),
        "estimator": "dmdc_tls",
        "errA_rel": errA_rel,
        "errB_rel": errB_rel,
        "errA_V_rel": errA_vis_rel,
        "errB_V_rel": errB_vis_rel,
        "err_unified": unified,
        "err_unified_V": unified_vis,
        "success_V": success,
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size <= 1 or y.size <= 1:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size <= 1 or y.size <= 1:
        return float("nan")
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return _pearson(xr, yr)


def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    r_xy = _pearson(x, y)
    r_xz = _pearson(x, z)
    r_yz = _pearson(y, z)
    if not (np.isfinite(r_xy) and np.isfinite(r_xz) and np.isfinite(r_yz)):
        return float("nan")
    denom = math.sqrt((1.0 - r_xz**2) * (1.0 - r_yz**2))
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float((r_xy - r_xz * r_yz) / denom)


def _ols_summary(d_vals: np.ndarray, n_vals: np.ndarray, r_vals: np.ndarray) -> Dict[str, object]:
    m = r_vals.size
    if m == 0:
        return {
            "coefficients": None,
            "n_samples": 0,
            "r2": float("nan"),
            "adj_r2": float("nan"),
            "aic": float("nan"),
            "bic": float("nan"),
            "sigma2": float("nan"),
        }
    X = np.column_stack([np.ones(m), d_vals, n_vals])
    coeffs, _, _, _ = np.linalg.lstsq(X, r_vals, rcond=None)
    yhat = X @ coeffs
    resid = r_vals - yhat
    rss = float(np.sum(resid**2))
    tss = float(np.sum((r_vals - np.mean(r_vals))**2))
    r2 = float("nan") if tss == 0 else float(1.0 - rss / tss)
    p = X.shape[1]
    if m > p:
        adj_r2 = float(1.0 - (1.0 - r2) * (m - 1) / (m - p))
        sigma2 = float(rss / (m - p))
    else:
        adj_r2 = float("nan")
        sigma2 = float("nan")
    if rss > 0 and m > 0:
        aic = float(m * math.log(rss / m) + 2 * p)
        bic = float(m * math.log(rss / m) + p * math.log(m))
    else:
        aic = float("nan")
        bic = float("nan")
    return {
        "coefficients": {
            "intercept": float(coeffs[0]),
            "beta_d": float(coeffs[1]),
            "beta_n": float(coeffs[2]),
        },
        "n_samples": int(m),
        "rss": rss,
        "r2": r2,
        "adj_r2": adj_r2,
        "aic": aic,
        "bic": bic,
        "sigma2": sigma2,
    }


def _summaries(rstar_df: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    if rstar_df.empty:
        summary["counts"] = {"n_samples": 0}
        return summary

    def stats_for(df: pd.DataFrame) -> Dict[str, object]:
        r_vals = df["r_star"].to_numpy(dtype=float)
        d_vals = df["d"].to_numpy(dtype=float)
        n_vals = df["n"].to_numpy(dtype=float)
        spearman_d = _spearman(r_vals, d_vals)
        spearman_n = _spearman(r_vals, n_vals)
        pearson_d = _pearson(r_vals, d_vals)
        pearson_n = _pearson(r_vals, n_vals)
        ranks_r = pd.Series(r_vals).rank(method="average").to_numpy()
        ranks_d = pd.Series(d_vals).rank(method="average").to_numpy()
        ranks_n = pd.Series(n_vals).rank(method="average").to_numpy()
        partial_d = _partial_corr(ranks_r, ranks_d, ranks_n)
        partial_n = _partial_corr(ranks_r, ranks_n, ranks_d)
        ols = _ols_summary(d_vals, n_vals, r_vals)
        return {
            "spearman_r_d": spearman_d,
            "spearman_r_n": spearman_n,
            "pearson_r_d": pearson_d,
            "pearson_r_n": pearson_n,
            "partial_spearman_d_given_n": partial_d,
            "partial_spearman_n_given_d": partial_n,
            "ols": ols,
            "n_samples": int(df.shape[0]),
        }

    summary["all"] = stats_for(rstar_df)
    uncensored = rstar_df.loc[rstar_df["censored"] == 0]
    summary["uncensored"] = stats_for(uncensored) if not uncensored.empty else {"n_samples": 0}

    within: Dict[str, object] = {}
    for n_val, sub in rstar_df.groupby("n"):
        within[str(int(n_val))] = {
            "spearman_r_d": _spearman(sub["r_star"].to_numpy(dtype=float), sub["d"].to_numpy(dtype=float)),
            "pearson_r_d": _pearson(sub["r_star"].to_numpy(dtype=float), sub["d"].to_numpy(dtype=float)),
            "count": int(sub.shape[0]),
            "censored_fraction": float(sub["censored"].mean()) if sub.shape[0] else float("nan"),
        }
    summary["within_n"] = within
    summary["counts"] = {
        "total": int(rstar_df.shape[0]),
        "censored": int(rstar_df["censored"].sum()),
    }
    return summary


def _make_heatmap(rstar_df: pd.DataFrame, outdir: pathlib.Path) -> None:
    if rstar_df.empty:
        return
    n_vals = sorted(rstar_df["n"].unique())
    k_vals = sorted(rstar_df["k"].unique())
    grid = np.full((len(k_vals), len(n_vals)), np.nan)
    for i, k in enumerate(k_vals):
        for j, n in enumerate(n_vals):
            subset = rstar_df[(rstar_df["n"] == n) & (rstar_df["k"] == k)]
            if not subset.empty:
                grid[i, j] = float(np.median(subset["r_star"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(n_vals)))
    ax.set_xticklabels([str(int(v)) for v in n_vals])
    ax.set_yticks(range(len(k_vals)))
    ax.set_yticklabels([str(int(v)) for v in k_vals])
    ax.set_xlabel("n")
    ax.set_ylabel("dim V(x_0) (k)")
    ax.set_title("Median minimum PE order $r_*$")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("median $r_*$")
    fig.tight_layout()
    fig.savefig(outdir / "heatmap_rstar.png", dpi=200)
    plt.close(fig)


def _make_error_curves(
    agg_df: pd.DataFrame,
    cfg: PEVisibilityConfig,
    outdir: pathlib.Path,
) -> None:
    if agg_df.empty:
        return
    beta0 = min(cfg.beta_grid)
    curves = (
        agg_df.groupby(["n", "k", "beta", "r_target"])
        .agg(
            err_median=("err_unified_V_mean", "median"),
            err_q1=("err_unified_V_mean", lambda x: float(np.quantile(x, 0.25))),
            err_q3=("err_unified_V_mean", lambda x: float(np.quantile(x, 0.75))),
        )
        .reset_index()
    )
    n_vals = sorted(curves["n"].unique())
    fig, axes = plt.subplots(len(n_vals), 1, figsize=(6, 3 * len(n_vals)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, n_val in zip(np.atleast_1d(axes), n_vals):
        sub = curves[(curves["n"] == n_val) & (curves["beta"] == beta0)]
        for k in sorted(sub["k"].unique()):
            line = sub[sub["k"] == k].sort_values("r_target")
            r = line["r_target"].to_numpy()
            med = line["err_median"].to_numpy()
            q1 = line["err_q1"].to_numpy()
            q3 = line["err_q3"].to_numpy()
            ax.plot(r, med, marker="o", label=f"k={int(k)}")
            ax.fill_between(r, q1, q3, alpha=0.2)
        ax.axhline(cfg.tau, color="k", linestyle="--", linewidth=1)
        ax.set_ylabel(r"$\overline{E}_\text{vis}(r)$")
        ax.set_title(f"n = {int(n_val)}")
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[-1].set_xlabel("PE order r")
    axes[0].legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "visible_error_curves.png", dpi=200)
    plt.close(fig)


def _make_scatter(
    rstar_df: pd.DataFrame,
    stats: Dict[str, object],
    cfg: PEVisibilityConfig,
    outdir: pathlib.Path,
) -> None:
    if rstar_df.empty:
        return
    rng = np.random.default_rng(cfg.seed + 1)
    jitter = rng.normal(scale=0.08, size=rstar_df.shape[0])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axes[0].scatter(rstar_df["d"] + jitter, rstar_df["r_star"], alpha=0.6, edgecolors="none")
    axes[0].set_xlabel("deficiency d = n - k")
    axes[0].set_ylabel("minimum PE order $r_*$")
    rho_d = float(stats.get("spearman_r_d", float("nan"))) if stats else float("nan")
    axes[0].set_title(f"Spearman ρ = {rho_d:.3f}" if np.isfinite(rho_d) else "r* vs d")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[1].scatter(rstar_df["n"] + jitter, rstar_df["r_star"], alpha=0.6, edgecolors="none")
    axes[1].set_xlabel("state dimension n")
    rho_n = float(stats.get("spearman_r_n", float("nan"))) if stats else float("nan")
    axes[1].set_title(f"Spearman ρ = {rho_n:.3f}" if np.isfinite(rho_n) else "r* vs n")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "rstar_scatter.png", dpi=200)
    plt.close(fig)


def _make_success_curves(agg_df: pd.DataFrame, cfg: PEVisibilityConfig, outdir: pathlib.Path) -> None:
    if agg_df.empty:
        return
    beta0 = min(cfg.beta_grid)
    curves = (
        agg_df.groupby(["k", "beta", "r_target"])
        .agg(
            success_median=("success_rate", "median"),
            success_q1=("success_rate", lambda x: float(np.quantile(x, 0.25))),
            success_q3=("success_rate", lambda x: float(np.quantile(x, 0.75))),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in sorted(curves["k"].unique()):
        sub = curves[(curves["k"] == k) & (curves["beta"] == beta0)].sort_values("r_target")
        r = sub["r_target"].to_numpy()
        med = sub["success_median"].to_numpy()
        q1 = sub["success_q1"].to_numpy()
        q3 = sub["success_q3"].to_numpy()
        ax.plot(r, med, marker="o", label=f"k={int(k)}")
        ax.fill_between(r, q1, q3, alpha=0.2)
    ax.set_xlabel("PE order r")
    ax.set_ylabel("success probability")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "success_curves.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def run(cfg: PEVisibilityConfig) -> None:
    outdir = cfg.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rng_master = np.random.default_rng(cfg.seed)
    trial_records: List[Dict[str, float | int]] = []

    for n in cfg.n_grid:
        for d in range(0, min(cfg.d_max, n - 1) + 1):
            system_count = 0
            while system_count < cfg.N_sys:
                system_seed = int(rng_master.integers(2**63 - 1))
                rng_sys = np.random.default_rng(system_seed)
                try:
                    _, _, Ad, Bd, x0, Qv = _draw_system_with_visible_dim(n, d, cfg, rng_sys)
                except RuntimeError:
                    continue
                r_max = min(cfg.r_max_cap, n + 2)
                for beta in cfg.beta_grid:
                    for r in range(1, r_max + 1):
                        for t in range(cfg.N_trials):
                            seed_trial = int(rng_sys.integers(2**63 - 1))
                            rng_trial = np.random.default_rng(seed_trial)
                            U, r_act = gen_prbs_for_order(cfg, r, beta, rng_trial)
                            X = simulate_dt(x0, Ad, Bd, U, noise_std=cfg.sigma, rng=rng_trial)
                            X0, X1 = X[:, :-1], X[:, 1:]
                            rec = _trial_record(
                                cfg,
                                n,
                                d,
                                system_count,
                                system_seed,
                                beta,
                                r,
                                U,
                                r_act,
                                X0,
                                X1,
                                Ad,
                                Bd,
                                Qv,
                                cfg.tau,
                                seed_trial,
                            )
                            trial_records.append(rec)
                system_count += 1

    if not trial_records:
        raise RuntimeError("No trial data generated; check configuration.")

    trials_df = pd.DataFrame(trial_records)
    trials_df.to_csv(outdir / "trials.csv", index=False)

    agg_df = (
        trials_df.groupby(
            [
                "n",
                "k",
                "d",
                "system_id",
                "system_seed",
                "beta",
                "sigma",
                "r_target",
            ],
            as_index=False,
        )
        .agg(
            errA_rel_mean=("errA_rel", "mean"),
            errB_rel_mean=("errB_rel", "mean"),
            errA_V_rel_mean=("errA_V_rel", "mean"),
            errB_V_rel_mean=("errB_V_rel", "mean"),
            err_unified_mean=("err_unified", "mean"),
            err_unified_V_mean=("err_unified_V", "mean"),
            success_rate=("success_V", "mean"),
            r_actual_median=("r_actual", "median"),
            trials=("success_V", "size"),
            N_mean=("N", "mean"),
        )
    )
    agg_df.to_csv(outdir / "agg.csv", index=False)

    rstar_records: List[Dict[str, object]] = []
    for keys, group in agg_df.groupby(
        ["n", "k", "d", "system_id", "system_seed", "beta", "sigma"], as_index=False
    ):
        group_sorted = group.sort_values("r_target")
        success_mask = group_sorted["err_unified_V_mean"] <= cfg.tau
        if success_mask.any():
            r_star = int(group_sorted.loc[success_mask, "r_target"].iloc[0])
            censored = 0
        else:
            r_star = int(group_sorted["r_target"].max() + 1)
            censored = 1
        rstar_records.append(
            {
                "n": int(group_sorted["n"].iloc[0]),
                "k": int(group_sorted["k"].iloc[0]),
                "d": int(group_sorted["d"].iloc[0]),
                "system_id": int(group_sorted["system_id"].iloc[0]),
                "system_seed": int(group_sorted["system_seed"].iloc[0]),
                "beta": float(group_sorted["beta"].iloc[0]),
                "sigma": float(group_sorted["sigma"].iloc[0]),
                "r_star": r_star,
                "r_max": int(group_sorted["r_target"].max()),
                "censored": censored,
            }
        )
    rstar_df = pd.DataFrame(rstar_records)
    rstar_df.to_csv(outdir / "rstar.csv", index=False)

    summary = _summaries(rstar_df)
    summary["tau"] = cfg.tau
    summary_path = outdir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    config_dict = asdict(cfg)
    config_dict["outdir"] = str(cfg.outdir)
    with (outdir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config_dict, fh, indent=2)

    _make_heatmap(rstar_df, plots_dir)
    _make_error_curves(agg_df, cfg, plots_dir)
    _make_success_curves(agg_df, cfg, plots_dir)
    _make_scatter(rstar_df, summary.get("all", {}), cfg, plots_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_int_grid(text: str) -> Sequence[int]:
    if ":" in text:
        parts = text.split(":")
        if len(parts) == 3:
            start, step, stop = map(int, parts)
            return list(range(start, stop + 1, step))
    return [int(tok) for tok in text.split(",") if tok]


def _parse_float_grid(text: str) -> Sequence[float]:
    return [float(tok) for tok in text.split(",") if tok]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PE order vs visible subspace experiment (v2).")
    parser.add_argument("--n-grid", type=str, default="4,6,8,10,12")
    parser.add_argument("--d-max", type=int, default=3)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--beta-grid", type=str, default="1.0,1.5")
    parser.add_argument("--systems", type=int, default=10)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--r-max-cap", type=int, default=12)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="results/sim_pe2")
    parser.add_argument("--deterministic-x0", action="store_true")
    args = parser.parse_args(argv)

    n_grid_vals = tuple(_parse_int_grid(args.n_grid))
    beta_vals = tuple(_parse_float_grid(args.beta_grid))

    cfg = PEVisibilityConfig(
        n=max(n_grid_vals) if n_grid_vals else args.m,
        m=args.m,
        dt=args.dt,
        seed=args.seed,
        n_grid=n_grid_vals,
        d_max=args.d_max,
        beta_grid=beta_vals,
        N_sys=args.systems,
        N_trials=args.trials,
        r_max_cap=args.r_max_cap,
        sigma=args.sigma,
        tau=args.tau,
        outdir=pathlib.Path(args.outdir),
        deterministic_x0=args.deterministic_x0,
    )
    run(cfg)


if __name__ == "__main__":
    main()