"""Stratified variant of ``sim_scoree`` with controllability-rank sweeps.

This script reuses the identifiability/estimation correlation machinery from
``experiments.sim_scoree`` but stratifies the underlying systems so that
``dim V(x0)`` (the reachable dimension from the sampled initial state) covers
all values ``r = 1, ..., n`` uniformly.  The construction of ``(A, B, x0)``
triples mirrors the deterministic / rejection-based routines used in
``experiments.sim_escon``.
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

try:  # package mode
    from ..ensembles import draw_with_ctrb_rank
    from ..metrics import (
        cont2discrete_zoh,
        unified_generator,
        build_visible_basis_dt,
    )
    from ..simulation import simulate_dt, prbs
    from ..estimators import (
        dmdc_pinv,
        moesp_fit,
        sindy_fit,
        node_fit,
    )
except Exception:  # pragma: no cover - script fallback
    from ensembles import draw_with_ctrb_rank
    from metrics import (
        cont2discrete_zoh,
        unified_generator,
        build_visible_basis_dt,
    )
    from simulation import simulate_dt, prbs
    from estimators import (
        dmdc_pinv,
        moesp_fit,
        sindy_fit,
        node_fit,
    )

from .sim_scoree import (
    compute_core_metrics,
    add_transforms,
    spearman_table,
    scatter_plots,
    scatter_plots_zoom,
    parse_estimators,
    relative_error_fro,
)

# ``make_estimator`` lives in ``sim_scoree`` but depends on locally scoped
# estimator helpers; we reproduce a minimal version here to avoid circular
# imports when running as a stand-alone script.
def _make_estimator(name: str):
    lname = name.lower()
    if lname == "dmdc":
        def _f(X0, X1, U, n=None, dt=None):
            return dmdc_pinv(X0, X1, U)
        return _f
    if lname == "moesp":
        def _f(X0, X1, U, n=None, dt=None):
            return moesp_fit(X0, X1, U, n=n)
        return _f
    if lname == "sindy":
        def _f(X0, X1, U, n=None, dt=None):
            if dt is None:
                raise ValueError("sindy requires dt")
            return sindy_fit(X0, X1, U, dt)
        return _f
    if lname == "node":
        def _f(X0, X1, U, n=None, dt=None):
            if dt is None:
                raise ValueError("node requires dt")
            return node_fit(X0, X1, U, dt)
        return _f
    raise ValueError(f"Unknown estimator '{name}'. Choose from dmdc, moesp, sindy, node.")


def _orthonormal_basis(M: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    if M.size == 0:
        return np.zeros((M.shape[0], 0))
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    if s.size == 0:
        return np.zeros((M.shape[0], 0))
    cutoff = tol * max(M.shape)
    rank = int(np.sum(s > cutoff))
    return U[:, :rank]


def _reachable_basis(Ad: np.ndarray, Bd: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    zero = np.zeros(Ad.shape[0])
    K = unified_generator(Ad, Bd, zero, mode="unrestricted")
    return _orthonormal_basis(K, tol=tol)


def _visible_basis(Ad: np.ndarray, Bd: np.ndarray, x0: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    return build_visible_basis_dt(Ad, Bd, x0, tol=tol)


def _construct_x0_with_dimV(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Rbasis: np.ndarray,
    dim_visible: int,
    rng: np.random.Generator,
    tol: float = 1e-12,
    max_tries: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    r = int(Rbasis.shape[1])
    k = int(dim_visible)
    if not (1 <= k <= r):
        raise ValueError(f"Requested dim V(x0) k={k} must lie in [1, r={r}].")

    Ar = Rbasis.T @ Ad @ Rbasis
    for _ in range(max_tries):
        y0 = rng.standard_normal(r)
        nrm = float(np.linalg.norm(y0))
        if nrm <= tol:
            continue
        y0 /= nrm

        cols = []
        v = y0
        for _ in range(k):
            cols.append(v)
            v = Ar @ v
        K = np.column_stack(cols)
        Q, _ = np.linalg.qr(K, mode="reduced")
        if Q.shape[1] < k:
            continue

        y = Q[:, k - 1]
        x0 = Rbasis @ y
        x0 /= float(np.linalg.norm(x0) + 1e-15)
        Vbasis = _visible_basis(Ad, Bd, x0, tol=tol)
        if Vbasis.shape[1] == k:
            return x0, Vbasis

        if k >= 2:
            y = Q[:, k - 1] + 1e-3 * Q[:, k - 2]
        else:
            y = Q[:, 0] + 1e-3 * rng.standard_normal(r)
        y /= float(np.linalg.norm(y) + 1e-15)
        x0 = Rbasis @ y
        x0 /= float(np.linalg.norm(x0) + 1e-15)
        Vbasis = _visible_basis(Ad, Bd, x0, tol=tol)
        if Vbasis.shape[1] == k:
            return x0, Vbasis

    # fallback to rejection sampling
    return _sample_visible_initial_state(Ad, Bd, Rbasis, k, rng, max_attempts=256, tol=tol)


def _sample_visible_initial_state_det(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Rbasis: np.ndarray,
    dim_visible: int,
    rng: np.random.Generator,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    return _construct_x0_with_dimV(Ad, Bd, Rbasis, dim_visible, rng, tol=tol)


def _sample_visible_initial_state(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Rbasis: np.ndarray,
    dim_visible: int,
    rng: np.random.Generator,
    max_attempts: int,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    n = Ad.shape[0]
    for _ in range(max_attempts):
        if dim_visible >= n:
            coeff = rng.standard_normal(n)
            x0 = coeff / float(np.linalg.norm(coeff) + 1e-15)
        else:
            coeff = rng.standard_normal(dim_visible)
            x0 = Rbasis @ coeff
            nrm = float(np.linalg.norm(x0))
            if nrm <= tol:
                continue
            x0 /= nrm
        Vbasis = _visible_basis(Ad, Bd, x0, tol=tol)
        if Vbasis.shape[1] == dim_visible:
            return x0, Vbasis
    raise RuntimeError(
        f"Failed to draw x0 with dim V(x0)={dim_visible} after {max_attempts} attempts."
    )


@dataclass
class StratificationConfig:
    n: int
    m: int
    dt: float
    ensemble: str
    max_system_draws: int
    max_x0_draws: int
    det_x0: bool
    tol: float = 1e-12


_BASE_ENSEMBLE_MAP = {
    "ginibre": "ginibre",
    "stable": "stable",
    "sparse": "sparse",
    "binary": "binary",
}


def _draw_system_with_rank(
    cfg: StratificationConfig,
    rng: np.random.Generator,
    rank: int,
):
    if rank < 1 or rank > cfg.n:
        raise ValueError(f"Requested rank {rank} outside [1, n={cfg.n}].")
    base = _BASE_ENSEMBLE_MAP.get(cfg.ensemble)
    if base is None:
        raise ValueError(
            f"Unknown ensemble '{cfg.ensemble}'. Choose from {sorted(_BASE_ENSEMBLE_MAP)}."
        )

    attempts = 0
    while attempts < cfg.max_system_draws:
        attempts += 1
        A, B, meta = draw_with_ctrb_rank(
            n=cfg.n,
            m=cfg.m,
            r=rank,
            rng=rng,
            ensemble_type=base,
            embed_random_basis=True,
        )
        Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)
        Rbasis = _reachable_basis(Ad, Bd, tol=cfg.tol)
        if Rbasis.shape[1] != rank:
            continue

        if cfg.det_x0:
            x0, Vbasis = _sample_visible_initial_state_det(
                Ad, Bd, Rbasis, rank, rng, tol=cfg.tol
            )
        else:
            x0, Vbasis = _sample_visible_initial_state(
                Ad, Bd, Rbasis, rank, rng, cfg.max_x0_draws, tol=cfg.tol
            )
        if Vbasis.shape[1] != rank:
            continue
        return A, B, Ad, Bd, x0, Vbasis, meta

    raise RuntimeError(
        f"Unable to synthesise a system with reachable dimension {rank} after "
        f"{cfg.max_system_draws} attempts."
    )


def run_trials_stratified(
    *,
    n: int,
    m: int,
    T: int,
    dt: float,
    per_rank: int,
    noise_std: float,
    seed: int,
    ensemble: str,
    estimators: Tuple[str, ...],
    det_x0: bool,
    max_system_draws: int,
    max_x0_draws: int,
    u_scale: float = 1.0,
):
    rng = np.random.default_rng(seed)
    cfg = StratificationConfig(
        n=n,
        m=m,
        dt=dt,
        ensemble=ensemble,
        max_system_draws=max_system_draws,
        max_x0_draws=max_x0_draws,
        det_x0=det_x0,
    )

    est_funcs = {}
    for name in estimators:
        try:
            est_funcs[name] = _make_estimator(name)
        except Exception as exc:
            print(f"[warn] Skipping estimator '{name}': {exc}")
    if not est_funcs:
        raise RuntimeError("No valid estimators selected.")

    rows = []
    A_store = []
    B_store = []
    Ad_store = []
    Bd_store = []
    x0_store = []
    rank_store = []

    trial = 0
    for rank in range(1, n + 1):
        for rep in range(per_rank):
            try:
                A, B, Ad, Bd, x0, Vbasis, _ = _draw_system_with_rank(cfg, rng, rank)
            except RuntimeError as exc:
                print(f"[warn] Rank {rank} draw failed: {exc}")
                continue

            U = prbs(T, m, scale=u_scale, dwell=1, rng=rng)
            X = simulate_dt(x0, Ad, Bd, U, noise_std=noise_std, rng=rng)
            X0, X1 = X[:, :-1], X[:, 1:]
            U_cm = U.T

            crit = compute_core_metrics(A, B, x0)
            crit["rank"] = rank

            errs = {}
            for name, f in est_funcs.items():
                try:
                    Ahat, Bhat = f(X0, X1, U_cm, n=n, dt=dt)
                    errs[f"err_{name}"] = relative_error_fro(Ahat, Bhat, Ad, Bd)
                except Exception as exc:
                    print(f"[warn] Estimator '{name}' failed on trial {trial}: {exc}")
                    errs[f"err_{name}"] = np.nan

            rows.append(dict(trial=trial, rep=rep, **crit, **errs))
            trial += 1

            A_store.append(A)
            B_store.append(B)
            Ad_store.append(Ad)
            Bd_store.append(Bd)
            x0_store.append(x0)
            rank_store.append(rank)

    df = pd.DataFrame(rows)
    meta_out = {
        "A": np.stack(A_store) if A_store else np.zeros((0, n, n)),
        "B": np.stack(B_store) if B_store else np.zeros((0, n, m)),
        "Ad": np.stack(Ad_store) if Ad_store else np.zeros((0, n, n)),
        "Bd": np.stack(Bd_store) if Bd_store else np.zeros((0, n, m)),
        "x0": np.stack(x0_store) if x0_store else np.zeros((0, n)),
        "rank": np.asarray(rank_store, dtype=int),
        "ensemble": ensemble,
        "det_x0": det_x0,
        "estimators": list(est_funcs.keys()),
        "per_rank": per_rank,
        "T": T,
        "dt": dt,
    }
    return df, meta_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--per-r", type=int, default=20,
                    help="Number of (A, B, x0) samples per reachable rank r.")
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=31415)
    ap.add_argument("--outdir", type=str, default="ident_vs_error_strat")
    ap.add_argument("--tag", type=str, default="strat")
    ap.add_argument("--ensemble", type=str, default="ginibre",
                    choices=sorted(_BASE_ENSEMBLE_MAP.keys()))
    ap.add_argument("--estimators", type=str, default="dmdc,moesp",
                    help="Comma-separated list from {dmdc,moesp,sindy,node}.")
    ap.add_argument("--det-x0", action="store_true",
                    help="Use deterministic Krylov constructor for x0 (default: rejection sampling).")
    ap.add_argument("--max-system-draws", type=int, default=2048,
                    help="Maximum attempts to synthesise a system per rank.")
    ap.add_argument("--max-x0-draws", type=int, default=512,
                    help="Maximum attempts for rejection sampling x0 when --det-x0 is off.")
    ap.add_argument("--u-scale", type=float, default=1.0,
                    help="Amplitude of the PRBS input sequence.")
    ap.add_argument("--zoom", action="store_true",
                    help="Also save zoomed scatter plots (≈80% lower-left coverage by default).")
    ap.add_argument("--zoom-q", type=float, default=0.9,
                    help="Per-axis quantile for zoom box (default 0.9 → ≥80% joint coverage).")

    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    plotdir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)

    est_list = tuple(parse_estimators(args.estimators))

    df, meta = run_trials_stratified(
        n=args.n,
        m=args.m,
        T=args.T,
        dt=args.dt,
        per_rank=args.per_r,
        noise_std=args.noise_std,
        seed=args.seed,
        ensemble=args.ensemble,
        estimators=est_list,
        det_x0=args.det_x0,
        max_system_draws=args.max_system_draws,
        max_x0_draws=args.max_x0_draws,
        u_scale=args.u_scale,
    )
    df = add_transforms(df)

    tag = (
        f"{args.tag}_ens-{args.ensemble}_det-{int(args.det_x0)}_"
        f"perr-{args.per_r}_ests-{'-'.join(meta['estimators'])}"
    )
    df.to_csv(outdir / f"results_{tag}.csv", index=False)

    ycols = [c for c in df.columns if c.startswith("err_")]
    if ycols:
        stab = spearman_table(df, ycols)
        stab.to_csv(outdir / f"spearman_{tag}.csv", index=False)
        for y in ycols:
            scatter_plots(df, y, plotdir, tag)
            if args.zoom:
                scatter_plots_zoom(df, y, plotdir, tag, q_zoom=args.zoom_q)

    meta_path = outdir / f"systems_{tag}.npz"
    np.savez_compressed(
        meta_path,
        A=meta["A"],
        B=meta["B"],
        Ad=meta["Ad"],
        Bd=meta["Bd"],
        x0=meta["x0"],
        rank=meta["rank"],
        ensemble=meta["ensemble"],
        det_x0=int(meta["det_x0"]),
        estimators=np.asarray(meta["estimators"], dtype=object),
        per_rank=meta["per_rank"],
        T=meta["T"],
        dt=meta["dt"],
    )

    print("Saved:")
    print("  ", outdir / f"results_{tag}.csv")
    if ycols:
        print("  ", outdir / f"spearman_{tag}.csv")
    print("  ", meta_path)
    print("  plots ->", plotdir)


if __name__ == "__main__":
    main()