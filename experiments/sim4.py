r"""Equivalence-class membership experiment with robust diagnostics.

This script implements the validation procedure discussed in the prompt.  It
draws a random (A, B) pair with a prescribed controllability rank, generates
trajectories under persistently exciting PRBS inputs, and evaluates a suite of
identification algorithms (currently ridge-regularised OLS) on the resulting
data.  Errors are analysed both on the visible subspace V(x0) and on its
orthogonal complement so that we can distinguish "easy estimation" from the
theoretical equivalence-class invariance.

Key features compared to the baseline pseudocode:

* the visible subspace is constructed from the discrete-time pair (A_d, B_d)
  using a pivoted QR (via :func:`metrics.build_visible_basis_dt`);
* PRBS signals are regenerated until the regressor matrix is well conditioned;
* burn-in periods remove start-up transients before estimation;
* regression uses a numerically stable ridge solver instead of an explicit
  pseudo-inverse;
* relative errors on V(x0) use safe denominators, and complementary errors on
  V(x0)^{\perp} are reported explicitly;
* Markov-parameter, simulation, and subspace-alignment diagnostics are
  produced to characterise the estimates;
* an adversarial "W-block" perturbation highlights that raw errors can remain
  large even when the equivalence-class claim holds on V(x0).

The main entry point is :func:`run_experiment`, which returns a dictionary with
the sampled system, estimates, and the per-trial diagnostics table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Iterable, Tuple

import argparse
import pathlib

import numpy as np
import numpy.linalg as npl
import pandas as pd
from scipy.linalg import null_space

from ..config import ExperimentConfig
from ..ensembles import draw_with_ctrb_rank, draw_initial_state
from ..metrics import (
    build_visible_basis_dt,
    cont2discrete_zoh,
    regressor_stats,
)
from ..simulation import simulate_dt, prbs


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


def run_experiment(cfg: EqvMembershipConfig) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)

    A, B, meta = draw_with_ctrb_rank(
        n=cfg.n,
        m=cfg.m,
        r=cfg.target_rank,
        rng=rng,
        ensemble_type="ginibre",
        base_u="ginibre",
        embed_random_basis=True,
    )
    Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)

    T_total = cfg.burn_in + cfg.T_effective

    rows = []

    for trial in range(cfg.n_trials):
        trial_seed = rng.integers(0, 2**32 - 1)
        trial_rng = np.random.default_rng(int(trial_seed))

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

    df = pd.DataFrame(rows)
    success_rate = float((df["status"] == "success").mean()) if not df.empty else 0.0

    outdir = pathlib.Path("out_eqv_membership")
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "trial_logs.csv", index=False)

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
    summary.to_csv(outdir / "summary.csv", index=False)

    return {
        "A": A,
        "B": B,
        "Ad": Ad,
        "Bd": Bd,
        "meta": meta,
        "logs": df,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Equivalence-class membership test")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--burn", type=int, default=25)
    ap.add_argument("--noise", type=float, default=0.0)
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

    result = run_experiment(cfg)
    np.savez("eqv_membership_results.npz", **{k: v for k, v in result.items() if k in {"A", "B", "Ad", "Bd"}})


if __name__ == "__main__":
    main()
