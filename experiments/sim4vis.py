from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Iterable, Tuple, Sequence

import argparse
import pathlib

import numpy as np
import numpy.linalg as npl
import pandas as pd
from scipy.linalg import null_space
from scipy import stats

from ..config import ExperimentConfig
from ..ensembles import draw_with_ctrb_rank, draw_initial_state
from ..metrics import (
    build_visible_basis_dt,
    cont2discrete_zoh,
    regressor_stats,
)
from ..simulation import simulate_dt, prbs



# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def _generate_figures(df: pd.DataFrame, cfg: EqvMembershipConfig, outdir: pathlib.Path) -> None:
    figs_dir = outdir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    _plot_outcomes(df, cfg, figs_dir / "F0_outcomes.png")
    _plot_visible_ecdfs(df, cfg, figs_dir / "F1_ecdf_visible.png")
    _plot_leak_ecdf(df, cfg, figs_dir / "F2_leak_ecdf.png")
    _plot_angles_violin(df, figs_dir / "F3_angles_violin.png")
    _plot_raw_vs_visible(df, figs_dir / "F4_raw_vs_visible.png")
    _plot_wblock_effect(df, figs_dir / "F5_wblock_effect.png")
    _plot_markov_profile(df, cfg, figs_dir / "F6_markov_profile.png")
    _plot_sim_err(df, figs_dir / "F7_sim_errV.png")
    _plot_cond_vs_errors(df, cfg, figs_dir / "F8_cond_vs_errors.png")
    _plot_funnel(df, figs_dir / "F9_funnel.png")


def _status_counts(df: pd.DataFrame) -> Dict[str, int]:
    statuses = [
        "success",
        "fail_markov",
        "fail_visible",
        "poor_excitation",
        "x0_dim_mismatch",
    ]
    counts = {status: int((df["status"] == status).sum()) for status in statuses}
    others = int(len(df) - sum(counts.values()))
    if others:
        counts["other"] = others
    return counts


def _plot_outcomes(df: pd.DataFrame, cfg: EqvMembershipConfig, outfile: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    counts = _status_counts(df)
    statuses = [k for k in ["success", "fail_markov", "fail_visible", "poor_excitation", "x0_dim_mismatch", "other"] if k in counts]
    colors = {
        "success": "#2ca02c",
        "fail_markov": "#d62728",
        "fail_visible": "#ff7f0e",
        "poor_excitation": "#9467bd",
        "x0_dim_mismatch": "#8c564b",
        "other": "#7f7f7f",
    }

    total = sum(counts.values())
    successes = counts.get("success", 0)
    ci_low, ci_high = _wilson_ci(successes, total)
    success_rate = successes / total if total else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    bottom = 0
    for status in statuses:
        frac = counts[status] / total if total else 0.0
        axes[0].bar(["overall"], [frac], bottom=bottom, color=colors.get(status, "#999999"), label=status)
        bottom += frac
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("fraction of trials")
    axes[0].set_title("Outcome breakdown")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    if total:
        lower_err = success_rate - ci_low
        upper_err = ci_high - success_rate
        axes[1].errorbar(
            [0],
            [success_rate],
            yerr=[[max(lower_err, 0.0)], [max(upper_err, 0.0)]],
            fmt="o",
            color="#1f77b4",
        )
    else:
        axes[1].scatter([0], [0.0], color="#1f77b4")
    axes[1].set_xticks([0])
    axes[1].set_xticklabels([f"n={total}"])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("success rate")
    axes[1].set_title("Success rate (95% Wilson CI)")
    axes[1].grid(True, axis="y", ls="--", alpha=0.4)

    fig.suptitle(
        "sim4 outcomes"
        + f" | tol_visible={cfg.tol_visible:g}, tol_markov={cfg.tol_markov:g}, tol_leak={cfg.tol_leak:g}"
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_visible_ecdfs(df: pd.DataFrame, cfg: EqvMembershipConfig, outfile: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    mask = df["status"].isin(["success", "fail_visible", "fail_markov"])
    dA = df.loc[mask, "dA_V"].dropna().astype(float)
    dB = df.loc[mask, "dB_V"].dropna().astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, data, label in zip(axes, [dA, dB], ["dA_V", "dB_V"]):
        if len(data):
            xs = np.sort(data.to_numpy())
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.step(xs, ys, where="post", label="ECDF")
            ax.axvline(cfg.tol_visible, color="r", ls="--", lw=1, label="tol_visible")
            ax.set_xlabel(label)
        else:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        ax.set_ylabel("ECDF")
        ax.grid(True, ls="--", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="lower right")
    fig.suptitle("Visible-subspace relative errors")
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_leak_ecdf(df: pd.DataFrame, cfg: EqvMembershipConfig, outfile: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    leak = df.loc[df["leak"].notna(), "leak"].astype(float)
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(leak):
        xs = np.sort(leak.to_numpy())
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post")
    else:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
    ax.axvline(cfg.tol_leak, color="r", ls="--", lw=1, label="tol_leak")
    ax.set_xlabel("leak")
    ax.set_ylabel("ECDF")
    ax.set_title("Leakage outside V(x0)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_angles_violin(df: pd.DataFrame, outfile: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    angles_max = df.loc[df["angles_max"].notna(), "angles_max"].astype(float)
    angles_mean = df.loc[df["angles_mean"].notna(), "angles_mean"].astype(float)
    fig, ax = plt.subplots(figsize=(6, 4))
    datasets = []
    labels = []
    if len(angles_max):
        datasets.append(angles_max.to_numpy())
        labels.append("max")
    if len(angles_mean):
        datasets.append(angles_mean.to_numpy())
        labels.append("mean")
    if datasets:
        positions = np.arange(1, len(datasets) + 1)
        ax.violinplot(datasets, positions=positions, showmeans=True)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
    else:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        ax.set_xticks([])
    ax.set_ylabel("principal angle (rad)")
    ax.set_title("Visible subspace alignment")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_raw_vs_visible(df: pd.DataFrame, outfile: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    mask = df["status"].isin(["success", "fail_visible", "fail_markov"])
    data = df.loc[mask, ["raw_err_A", "dA_V", "raw_err_B", "dB_V"]].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    labels = [("raw_err_A", "dA_V"), ("raw_err_B", "dB_V")]

    for ax, (raw_label, vis_label) in zip(axes, labels):
        if raw_label in data.columns and vis_label in data.columns and len(data):
            ax.scatter(data[raw_label], data[vis_label], alpha=0.6, s=20)
            pearson = data[[raw_label, vis_label]].corr(method="pearson").iloc[0, 1]
            spearman = data[[raw_label, vis_label]].corr(method="spearman").iloc[0, 1]
            ax.text(0.05, 0.95, f"Pearson={pearson:.2f}\nSpearman={spearman:.2f}", transform=ax.transAxes, va="top")
            ax.set_xlabel(raw_label)
            ax.set_ylabel(vis_label)
            ax.grid(True, ls="--", alpha=0.4)
        else:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
    fig.suptitle("Raw vs visible errors")
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_wblock_effect(df: pd.DataFrame, outfile: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    cols = ["raw_err_A", "dA_sharp_W"]
    subset = df[cols].dropna()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    if len(subset):
        ax.scatter(subset["raw_err_A"], subset["dA_sharp_W"], alpha=0.6, s=20)
        lo = float(min(subset["raw_err_A"].min(), subset["dA_sharp_W"].min()))
        hi = float(max(subset["raw_err_A"].max(), subset["dA_sharp_W"].max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        diff = subset["dA_sharp_W"] - subset["raw_err_A"]
        mean_diff = float(diff.mean())
        ci_low, ci_high = _bootstrap_ci(diff.to_numpy())
        ax.text(
            0.05,
            0.95,
            f"Δ mean={mean_diff:.2e}\n95% CI=[{ci_low:.2e}, {ci_high:.2e}]",
            transform=ax.transAxes,
            va="top",
        )
    else:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("raw_err_A")
    ax.set_ylabel("dA_sharp_W")
    ax.set_title("Adversarial W-block effect")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_markov_profile(df: pd.DataFrame, cfg: EqvMembershipConfig, outfile: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    rows = []
    for _, row in df.iterrows():
        errs = row.get("markov_errs")
        if isinstance(errs, Sequence):
            for k, val in enumerate(errs):
                rows.append({"k": k, "markov_err": float(val), "status": row.get("status", "")})
    fig, ax = plt.subplots(figsize=(6, 4))
    if rows:
        markov_df = pd.DataFrame(rows)
        grouped = markov_df.groupby("k")["markov_err"]
        ks = sorted(markov_df["k"].unique())
        med = grouped.median()
        q1 = grouped.quantile(0.25)
        q3 = grouped.quantile(0.75)

        ax.plot(ks, med[ks], marker="o", label="median")
        ax.fill_between(ks, q1[ks], q3[ks], alpha=0.3, label="IQR")
        ax.axhline(cfg.tol_markov, color="r", ls="--", lw=1, label="tol_markov")
        ax.set_ylabel("relative error")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Markov horizon k")
    ax.set_title("Projected Markov parameter errors")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_sim_err(df: pd.DataFrame, outfile: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    sim_err = df.loc[df["sim_err_V"].notna(), "sim_err_V"].astype(float)
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(sim_err):
        xs = np.sort(sim_err.to_numpy())
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post")
    else:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("sim_err_V")
    ax.set_ylabel("ECDF")
    ax.set_title("Projected simulation error")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_cond_vs_errors(df: pd.DataFrame, cfg: EqvMembershipConfig, outfile: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    cols = ["smin", "cond", "dA_V", "leak", "markov_err_max"]
    subset = df[cols].dropna()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    if len(subset):
        axes[0].scatter(subset["smin"], subset["dA_V"], alpha=0.6, s=20)
        axes[0].axvline(cfg.min_regressor_smin, color="r", ls="--", lw=1)
        axes[0].set_xlabel("sigma_min(Z)")
        axes[0].set_ylabel("dA_V")
        axes[0].grid(True, ls="--", alpha=0.4)

        axes[1].scatter(subset["smin"], subset["leak"], alpha=0.6, s=20)
        axes[1].axvline(cfg.min_regressor_smin, color="r", ls="--", lw=1)
        axes[1].set_xlabel("sigma_min(Z)")
        axes[1].set_ylabel("leak")
        axes[1].grid(True, ls="--", alpha=0.4)

        axes[2].scatter(subset["cond"], subset["markov_err_max"], alpha=0.6, s=20)
        axes[2].axvline(cfg.max_regressor_cond, color="r", ls="--", lw=1)
        axes[2].set_xlabel("cond(Z)")
        axes[2].set_ylabel("max Markov err")
        axes[2].grid(True, ls="--", alpha=0.4)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
    fig.suptitle("Excitation diagnostics vs errors")
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_funnel(df: pd.DataFrame, outfile: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    funnel = pd.Series(
        {
            "drawn": len(df),
            "dim_ok": int((df["status"] != "x0_dim_mismatch").sum()),
            "pe_ok": int((~df["status"].isin(["x0_dim_mismatch", "poor_excitation"])).sum()),
            "visible_ok": int(df["status"].isin(["success", "fail_markov"]).sum()),
            "markov_ok": int((df["status"] == "success").sum()),
        }
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    steps = funnel.index.tolist()
    vals = funnel.values
    ax.barh(range(len(steps)), vals, color="#1f77b4")
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels(steps)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, str(int(v)), va="center")
    ax.set_xlabel("# trials")
    ax.set_title("Gate attrition")
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _bootstrap_ci(values: np.ndarray, alpha: float = 0.05, n_resamples: int = 2000) -> Tuple[float, float]:
    if values.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(12345)
    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        resample = rng.choice(values, size=values.size, replace=True)
        samples[i] = np.mean(resample)
    lower = float(np.quantile(samples, alpha / 2))
    upper = float(np.quantile(samples, 1 - alpha / 2))
    return lower, upper


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EqvMembershipConfig(ExperimentConfig):
    """Configuration for the equivalence-class membership study."""

    target_rank: int = 4
    max_x0_tries: int = 25
    burn_in: int = 25
    T_effective: int = 200
    min_regressor_smin: float = 1e-3
    max_regressor_cond: float = 1e8
    max_regen_inputs: int = 20
    ridge_lambda: float = 1e-8
    eps_denom: float = 1e-12
    tol_visible: float = 5e-2
    tol_leak: float = 5e-2
    tol_markov: float = 5e-2
    markov_horizon: int = 6
    eval_T: int = 120
    eval_dwell: int = 1
    complement_scale: float = 10.0
    complement_jitter: float = 0.5
    rtol_rank: float = 1e-12

    def __post_init__(self) -> None:  # type: ignore[override]
        super().__post_init__()
        if self.target_rank < 0 or self.target_rank > self.n:
            raise ValueError(
                f"target_rank must be in [0, n]; got {self.target_rank} for n={self.n}."
            )
        if self.T_effective <= 0:
            raise ValueError("T_effective must be positive.")
        if self.burn_in < 0:
            raise ValueError("burn_in cannot be negative.")
        if self.markov_horizon <= 0:
            raise ValueError("markov_horizon must be positive.")
        if self.eval_T <= 0:
            raise ValueError("eval_T must be positive.")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _safe_norm(x: np.ndarray) -> float:
    return float(npl.norm(x, ord="fro"))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serialisable")


def _wilson_ci(successes: int, total: int, alpha: float = 0.05) -> Tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    phat = successes / total
    denom = 1 + z**2 / total
    centre = phat + z**2 / (2 * total)
    radius = z * np.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2))
    lower = (centre - radius) / denom
    upper = (centre + radius) / denom
    return float(lower), float(upper)


def _orth_complement(P: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Return an orthonormal basis for V^⊥ given an orthonormal basis P of V."""

    if P.size == 0:
        return np.eye(P.shape[0])
    ns = null_space(P.T, rcond=tol)
    if ns.size == 0:
        return np.zeros((P.shape[0], 0))
    return ns


def _principal_angles(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    if P.size == 0 or Q.size == 0:
        return np.zeros(0)
    M = P.T @ Q
    sv = npl.svd(M, compute_uv=False)
    sv = np.clip(sv, -1.0, 1.0)
    return np.arccos(sv)


def _maybe_parse_sequence(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def _coerce_trial_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    list_columns = ("markov_errs", "markov_denoms")
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(_maybe_parse_sequence)
    return df


def _project_markov_errors(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Ahat: np.ndarray,
    Bhat: np.ndarray,
    P: np.ndarray,
    horizon: int,
    eps: float,
) -> Dict[str, Any]:
    def _proj_markov(A: np.ndarray, B: np.ndarray) -> Iterable[np.ndarray]:
        Ak = np.eye(A.shape[0])
        for _ in range(horizon):
            yield P.T @ (Ak @ B)
            Ak = A @ Ak

    errs = []
    denom_vals = []
    for Mh, Mt in zip(_proj_markov(Ahat, Bhat), _proj_markov(Ad, Bd)):
        denom = max(_safe_norm(Mt), eps)
        denom_vals.append(denom)
        errs.append(_safe_norm(Mh - Mt) / denom)
    return {
        "markov_err_max": float(np.max(errs) if errs else 0.0),
        "markov_errs": errs,
        "markov_denoms": denom_vals,
    }


def _simulate_and_project(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Ahat: np.ndarray,
    Bhat: np.ndarray,
    x0: np.ndarray,
    U: np.ndarray,
    P: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, float]:
    X_true = simulate_dt(x0, Ad, Bd, U, noise_std=0.0, rng=rng)
    X_est = simulate_dt(x0, Ahat, Bhat, U, noise_std=0.0, rng=rng)
    V_true = P.T @ X_true
    V_est = P.T @ X_est
    diff = V_est - V_true
    denom = max(_safe_norm(V_true), 1e-12)
    err = _safe_norm(diff) / denom
    return {"sim_err_V": float(err)}


def _adversarial_W_block(
    Ad: np.ndarray,
    P: np.ndarray,
    rng: np.random.Generator,
    scale: float,
    jitter: float,
) -> Tuple[np.ndarray, np.ndarray]:
    Q = _orth_complement(P)
    if Q.size == 0:
        return Ad, Q
    dim_w = Q.shape[1]
    diag = scale + jitter * rng.standard_normal(dim_w)
    Delta = np.diag(diag)
    Asharp = Ad + Q @ Delta @ Q.T
    return Asharp, Q


def _visible_errors(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Ahat: np.ndarray,
    Bhat: np.ndarray,
    P: np.ndarray,
    eps: float,
) -> Dict[str, float]:
    if P.size == 0:
        return {k: 0.0 for k in ["dA_V", "dB_V", "leak", "dA_W", "dB_W"]}

    I = np.eye(Ad.shape[0])
    denom_A = max(_safe_norm(P.T @ Ad @ P), eps * max(1.0, _safe_norm(Ad)))
    denom_B = max(_safe_norm(P.T @ Bd), eps * max(1.0, _safe_norm(Bd)))
    dA_V = _safe_norm(P.T @ (Ahat - Ad) @ P) / denom_A
    dB_V = _safe_norm(P.T @ (Bhat - Bd)) / denom_B

    leak = _safe_norm((I - P @ P.T) @ Ahat @ P) / max(_safe_norm(Ahat @ P), eps)
    dA_W = _safe_norm((I - P @ P.T) @ (Ahat - Ad)) / max(_safe_norm(Ahat), eps)
    dB_W = _safe_norm((I - P @ P.T) @ Bhat) / max(_safe_norm(Bhat), eps)

    return {
        "dA_V": float(dA_V),
        "dB_V": float(dB_V),
        "leak": float(leak),
        "dA_W": float(dA_W),
        "dB_W": float(dB_W),
    }


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------


def run_experiment(
    cfg: EqvMembershipConfig,
    *,
    outdir: str | pathlib.Path | None = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)

    outdir_path = pathlib.Path(outdir) if outdir is not None else pathlib.Path("out_sim4")
    if verbose:
        print(
            "[sim4] Running"
            f" {cfg.n_trials} trial(s) with n={cfg.n}, m={cfg.m},"
            f" target_rank={cfg.target_rank}, T_eff={cfg.T_effective}, burn_in={cfg.burn_in}"
        )

    A, B, meta = draw_with_ctrb_rank(
        n=cfg.n,
        m=cfg.m,
        r=cfg.target_rank,
        rng=rng,
        ensemble_type="ginibre",
        embed_random_basis=True,
    )
    Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)

    T_total = cfg.burn_in + cfg.T_effective

    rows = []

    for trial in range(cfg.n_trials):
        trial_seed = rng.integers(0, 2**32 - 1)
        trial_rng = np.random.default_rng(int(trial_seed))

        if verbose:
            print(f"[sim4] Trial {trial + 1}/{cfg.n_trials}: seed={int(trial_seed)}")

        # Draw x0 ensuring dim V(x0) == target_rank
        for attempt in range(cfg.max_x0_tries):
            x0 = draw_initial_state(cfg.n, cfg.x0_mode, trial_rng)
            nrm = npl.norm(x0)
            if nrm == 0.0:
                continue
            x0 = x0 / nrm
            P = build_visible_basis_dt(Ad, Bd, x0, tol=cfg.rtol_rank)
            dim_V = P.shape[1]
            if dim_V == cfg.target_rank:
                break
        else:
            if verbose:
                print(
                    f"[sim4]   aborted: dim_V={int(dim_V)} did not match target "
                    f"{cfg.target_rank}"
                )
            rows.append(
                {
                    "trial": trial,
                    "status": "x0_dim_mismatch",
                    "dim_V": int(dim_V),
                }
            )
            continue

        # Generate persistently exciting input (with burn-in)
        regen_ok = False
        for regen in range(cfg.max_regen_inputs):
            U = prbs(T_total, cfg.m, scale=cfg.u_scale, dwell=cfg.dwell, rng=trial_rng)
            X = simulate_dt(x0, Ad, Bd, U, noise_std=cfg.noise_std, rng=trial_rng)

            start = cfg.burn_in
            end = start + cfg.T_effective
            X_eff = X[:, start : end + 1]
            X0 = X_eff[:, :-1]
            X1 = X_eff[:, 1:]
            U_eff = U[start:end, :]
            U_cm = U_eff.T

            zstats = regressor_stats(X0, U_cm, rtol_rank=cfg.rtol_rank)
            if (
                zstats["smin"] >= cfg.min_regressor_smin
                and zstats["cond"] <= cfg.max_regressor_cond
            ):
                regen_ok = True
                break
        if not regen_ok:
            zstats["smin"] = float(zstats.get("smin", 0.0))
            if verbose:
                cond_val = zstats.get("cond", float("nan"))
                print(
                    "[sim4]   aborted: failed to generate persistently exciting input"
                    f" (smin(Z)={zstats['smin']:.3e}, cond(Z)={cond_val:.3e})"
                )
            rows.append(
                {
                    "trial": trial,
                    "status": "poor_excitation",
                    "dim_V": int(dim_V),
                    **zstats,
                }
            )
            continue

        # Ridge-regularised least squares via normal equations
        Z = np.vstack([X0, U_cm])
        ZZt = Z @ Z.T
        lam = cfg.ridge_lambda
        G = ZZt + lam * np.eye(ZZt.shape[0])
        Theta = (X1 @ Z.T) @ npl.solve(G, np.eye(G.shape[0]))
        Ahat = Theta[:, : cfg.n]
        Bhat = Theta[:, cfg.n :]

        vis_errs = _visible_errors(Ad, Bd, Ahat, Bhat, P, cfg.eps_denom)

        Asharp, Q = _adversarial_W_block(
            Ad, P, trial_rng, scale=cfg.complement_scale, jitter=cfg.complement_jitter
        )
        if Q.size == 0:
            dA_sharp = 0.0
        else:
            denom_sharp = max(_safe_norm(Q.T @ Asharp @ Q), cfg.eps_denom)
            dA_sharp = _safe_norm(Q.T @ (Ahat - Asharp) @ Q) / denom_sharp

        raw_err_A = _safe_norm(Ahat - Ad) / max(_safe_norm(Ad), cfg.eps_denom)
        raw_err_B = _safe_norm(Bhat - Bd) / max(_safe_norm(Bd), cfg.eps_denom)

        markov_diag = _project_markov_errors(
            Ad, Bd, Ahat, Bhat, P, cfg.markov_horizon, cfg.eps_denom
        )

        U_eval = prbs(cfg.eval_T, cfg.m, scale=cfg.u_scale, dwell=cfg.eval_dwell, rng=trial_rng)
        sim_diag = _simulate_and_project(Ad, Bd, Ahat, Bhat, x0, U_eval, P, trial_rng)

        P_hat = build_visible_basis_dt(Ahat, Bhat, x0, tol=cfg.rtol_rank)
        angles = _principal_angles(P, P_hat)

        status = "success"
        if not (
            vis_errs["dA_V"] <= cfg.tol_visible
            and vis_errs["dB_V"] <= cfg.tol_visible
            and vis_errs["leak"] <= cfg.tol_leak
        ):
            status = "fail_visible"
        if markov_diag["markov_err_max"] > cfg.tol_markov:
            status = "fail_markov"

        rows.append(
            {
                "trial": trial,
                "status": status,
                "dim_V": int(P.shape[1]),
                "seed": int(trial_seed),
                **zstats,
                **vis_errs,
                **markov_diag,
                **sim_diag,
                "raw_err_A": float(raw_err_A),
                "raw_err_B": float(raw_err_B),
                "dA_sharp_W": float(dA_sharp),
                "angles_max": float(np.max(angles) if angles.size else 0.0),
                "angles_mean": float(np.mean(angles) if angles.size else 0.0),
            }
        )

        if verbose:
            cond_val = zstats.get("cond", float("nan"))
            smin_val = zstats.get("smin", float("nan"))
            print(
                "[sim4]   status={status:<15} dim_V={dim:2d}"
                " smin(Z)={smin:.3e} cond(Z)={cond:.3e}"
                " dA_V={dA:.3e} dB_V={dB:.3e} leak={leak:.3e}".format(
                    status=status,
                    dim=int(P.shape[1]),
                    smin=smin_val,
                    cond=cond_val,
                    dA=vis_errs["dA_V"],
                    dB=vis_errs["dB_V"],
                    leak=vis_errs["leak"],
                )
            )

    df = pd.DataFrame(rows)
    success_rate = float((df["status"] == "success").mean()) if not df.empty else 0.0

    outdir_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir_path / "trial_logs.csv", index=False)

    summary = pd.DataFrame(
        {
            "success_rate": [success_rate],
            "n": [cfg.n],
            "m": [cfg.m],
            "target_rank": [cfg.target_rank],
            "T_effective": [cfg.T_effective],
            "burn_in": [cfg.burn_in],
            "noise_std": [cfg.noise_std],
        }
    )
    summary.to_csv(outdir_path / "summary.csv", index=False)

    with (outdir_path / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=2, sort_keys=True, default=_json_default)

    if not df.empty:
        _generate_figures(df, cfg, outdir_path)

    if verbose:
        successes = int((df["status"] == "success").sum()) if not df.empty else 0
        print(
            f"[sim4] Completed {len(df)} trial(s); success rate = {success_rate:.3f}"
            f" ({successes}/{len(df)})"
        )
        print(f"[sim4] Results written to {outdir_path.resolve()}")

    return {
        "A": A,
        "B": B,
        "Ad": Ad,
        "Bd": Bd,
        "meta": meta,
        "logs": df,
        "summary": summary,
        "outdir": outdir_path,
    }


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Equivalence-class membership test (sim4)")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--burn", type=int, default=25)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory for outputs (defaults to ./out_sim4).",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-trial logging (final summary is still printed).",
    )
    ap.add_argument(
        "--replot",
        type=str,
        default=None,
        help="Re-render figures from an existing trial log directory or CSV file.",
    )
    ap.add_argument(
        "--replot-outdir",
        type=str,
        default=None,
        help="Destination directory for figures when using --replot (defaults to the source directory).",
    )
    args = ap.parse_args()

    cfg = EqvMembershipConfig(
        n=args.n,
        m=args.m,
        target_rank=args.rank,
        seed=args.seed,
        n_trials=args.trials,
        T_effective=args.T,
        burn_in=args.burn,
        noise_std=args.noise,
    )

    if args.replot:
        source_path = pathlib.Path(args.replot)
        if source_path.is_dir():
            logs_path = source_path / "trial_logs.csv"
            config_path = source_path / "config.json"
        else:
            logs_path = source_path
            config_path = source_path.with_name("config.json")
        if not logs_path.exists():
            raise FileNotFoundError(f"Could not find trial log at {logs_path}")
        df = pd.read_csv(logs_path)
        df = _coerce_trial_dataframe(df)

        cfg_for_plots = cfg
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            merged = {**asdict(cfg), **loaded}
            cfg_for_plots = EqvMembershipConfig(**merged)

        destination = pathlib.Path(args.replot_outdir) if args.replot_outdir else logs_path.parent
        destination.mkdir(parents=True, exist_ok=True)
        _generate_figures(df, cfg_for_plots, destination)
        print(f"[sim4] Regenerated figures from {logs_path} into {destination / 'figs'}")
        return

    result = run_experiment(cfg, outdir=args.outdir, verbose=not args.quiet)

    outdir_path = pathlib.Path(result["outdir"])
    matrices_path = outdir_path / "system_matrices.npz"
    np.savez(
        matrices_path,
        **{k: v for k, v in result.items() if k in {"A", "B", "Ad", "Bd"}},
    )

    logs_df = result["logs"]
    total_trials = len(logs_df)
    successes = int((logs_df["status"] == "success").sum()) if total_trials else 0
    success_rate = (
        float(result["summary"].iloc[0]["success_rate"]) if not result["summary"].empty else float("nan")
    )

    print(f"[sim4] Success rate: {success_rate:.3f} ({successes}/{total_trials} trials)")
    print(f"[sim4] Trial log : {outdir_path / 'trial_logs.csv'}")
    print(f"[sim4] Summary  : {outdir_path / 'summary.csv'}")
    print(f"[sim4] Matrices : {matrices_path}")
    figs_dir = outdir_path / "figs"
    if figs_dir.exists():
        print(f"[sim4] Figures  : {figs_dir}")


if __name__ == "__main__":
    main()

