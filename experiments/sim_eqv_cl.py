"""
Discrete-time (DT) equivalence-class check for repeated estimation runs.

What this does
--------------
For each trial:
  1) Generate a single CT pair (A,B) with a specified controllability rank deficiency
     using ensembles.draw_with_ctrb_rank, then ZOH-discretize once -> (A_d, B_d).
  2) Simulate data and estimate (Â, B̂) with DMDc (or MOESP).
  3) Depending on --mode:
     - 'theory': validate membership in the theoretical class [A,B]_{x0}
                 via a two-shot (free + forced) design with relative thresholds.
     - 'data'  : test data-equivalence via simulation residual and (optional)
                 Markov-parameter match up to kmax.

Outputs:
  - outdir/system.npz  (A, B, Ad, Bd, n, m, dt, seed, crank_def, ctrb_rank)
  - summary.csv with per-trial diagnostics and ok ∈ {0,1}
  - NPZ artifacts per trial for provenance
"""

from __future__ import annotations
import argparse, pathlib, math
import numpy as np
import pandas as pd

from ..metrics import (
    cont2discrete_zoh,
    unified_generator,
    same_equiv_class_dt_rel,
    regressor_stats,
    data_equivalence_residual,
    markov_match_dt,
    controllability_subspace_basis,   
)
from ..estimators import (
    dmdc_pinv,
    moesp_fit,
)
from ..ensembles import (
    draw_with_ctrb_rank,              # <<— system generator with target controllability rank
)

# ---------- utilities ----------

def unit(v: np.ndarray, eps: float = 0.0) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n if n > eps else 1.0)

def prbs(T: int, m: int, rng: np.random.Generator, dwell: int = 1) -> np.ndarray:
    steps = math.ceil(T / dwell)
    seq = rng.choice([-1.0, 1.0], size=(steps, m))
    return np.repeat(seq, dwell, axis=0)[:T, :]

def simulate_dt(Ad: np.ndarray, Bd: np.ndarray, U: np.ndarray, x0: np.ndarray) -> np.ndarray:
    n, T = Ad.shape[0], U.shape[0]
    X = np.empty((n, T + 1))
    X[:, 0] = x0
    for k in range(T):
        X[:, k + 1] = Ad @ X[:, k] + Bd @ U[k, :]
    return X

def rel_fro(Mhat: np.ndarray, Mtrue: np.ndarray) -> float:
    den = np.linalg.norm(Mtrue, "fro")
    return float(np.linalg.norm(Mhat - Mtrue, "fro") / (den if den > 0 else 1.0))

def ctrb_rank(A: np.ndarray, B: np.ndarray, rtol_rank: float = 1e-12) -> int:
    """Rank of [B, AB, ..., A^{n-1}B] via SVD cutoff."""
    V = controllability_subspace_basis(A, B, rtol=rtol_rank)
    return int(V.shape[1])

def sample_system_with_deficiency(
    n: int,
    m: int,
    crank_def: int,
    rng: np.random.Generator,
    dt: float,
    rtol_rank: float,
    max_tries: int,
):
    """
    Draw (A,B) such that rank([B,AB,...,A^{n-1}B]) = n - crank_def.
    Retries up to max_tries.
    """
    target_rank = n - int(crank_def)
    if target_rank < 0 or target_rank > n:
        raise ValueError(f"Invalid target rank {target_rank} for n={n} (crank_def={crank_def}).")

    last = None
    for _ in range(max_tries):
        A, B, meta = draw_with_ctrb_rank(n, m, target_rank, rng)
        r = ctrb_rank(A, B, rtol_rank)
        last = (A, B, r)
        if r == target_rank:
            Ad, Bd = cont2discrete_zoh(A, B, dt)
            return A, B, Ad, Bd, r
    # No exact match; return the last draw so the caller can decide
    if last is None:
        raise RuntimeError("Failed to sample a system with the desired controllability rank deficiency after max_tries.")
    A, B, r = last
    Ad, Bd = cont2discrete_zoh(A, B, dt)
    return A, B, Ad, Bd, r

# ---------- main experiment ----------

def run(
    trials: int = 200,
    dt: float = 0.05,
    T: int = 200,
    noise_std: float = 0.0,
    algo: str = "dmdc",
    seed: int = 7,
    outdir: str = "out_eqv_cl",
    mode: str = "theory",
    u_scale: float = 5.0,
    T_free: int = 60,
    T_forced: int = 200,
    rtol_eq: float = 1e-2,
    rtol_resid: float = 1e-10,
    rtol_rank: float = 1e-12,
    kmax_mp: int = 4,
    n: int = 3,
    m: int = 1,
    crank_def: int = 0,
    max_sys_tries: int = 100,
) -> pathlib.Path:

    rng = np.random.default_rng(seed)
    out = pathlib.Path(outdir); (out / "estimates").mkdir(parents=True, exist_ok=True)

    # ===== Generate ONE CT (A,B) with desired controllability rank deficiency =====
    A, B, Ad, Bd, actual_rank = sample_system_with_deficiency(
        n=n, m=m, crank_def=crank_def, rng=rng, dt=dt, rtol_rank=rtol_rank, max_tries=max_sys_tries
    )
    actual_def = n - actual_rank
    if actual_def != crank_def:
        print(f"[warn] Could not realize exact controllability rank deficiency {crank_def}; "
              f"using actual deficiency {actual_def} (rank={actual_rank})")

    # Persist the system once for provenance
    (out / "estimates").mkdir(parents=True, exist_ok=True)
    np.savez(out / "system.npz",
             A=A, B=B, Ad=Ad, Bd=Bd,
             n=n, m=m, dt=dt, seed=seed,
             crank_def=crank_def, ctrb_rank=actual_rank)

    rows = []
    for t in range(trials):
        # random x0 on S^{n-1}
        x0 = unit(rng.standard_normal(n))

        if mode == "theory":
            # (1) free response (pins A on K(A,x0))
            U_free = np.zeros((T_free, m))
            X_free = simulate_dt(Ad, Bd, U_free, x0)

            # (2) forced response with x0=0 (pins B and R(0) contribution)
            x0_forced = np.zeros(n)
            U_forced = u_scale * prbs(T_forced, m, rng, dwell=1)
            X_forced = simulate_dt(Ad, Bd, U_forced, x0_forced)

            # stack both for one fit
            X0 = np.concatenate([X_free[:, :-1],  X_forced[:, :-1]], axis=1)
            X1 = np.concatenate([X_free[:,  1:],  X_forced[:,  1:]], axis=1)
            U  = np.concatenate([U_free.T,        U_forced.T],       axis=1)

            # optional state noise
            if noise_std > 0:
                X0 = X0 + noise_std * rng.standard_normal(X0.shape)
                X1 = X1 + noise_std * rng.standard_normal(X1.shape)

            # estimate (Â, B̂)
            if algo == "moesp":
                Ahat, Bhat = moesp_fit(X0, X1, U, n=n)
            else:
                Ahat, Bhat = dmdc_pinv(X0, X1, U)

            # theoretical-class check (relative)
            ok_bool, info = same_equiv_class_dt_rel(
                Ad, Bd, Ahat, Bhat, x0,
                rtol_eq=rtol_eq, rtol_rank=rtol_rank, use_leak=True
            )

            # regressor diagnostics
            diagZ = regressor_stats(X0, U, rtol_rank=rtol_rank)
            errA_dt = rel_fro(Ahat, Ad); errB_dt = rel_fro(Bhat, Bd)

            rows.append(dict(
                trial=t, mode=mode, ok=int(ok_bool),
                dim_V=info.get("dim_V", 0),
                dA_V=info.get("dA_V", np.nan),
                leak=info.get("leak", np.nan),
                dB=info.get("dB", np.nan),
                thrA=info.get("thrA", np.nan),
                thrB=info.get("thrB", np.nan),
                thrLeak=info.get("thrLeak", np.nan),
                rank=diagZ.get("rank", 0),
                cond=diagZ.get("cond", np.inf),
                smin=diagZ.get("smin", 0.0),
                errA_dt=errA_dt, errB_dt=errB_dt,
                ctrb_rank=actual_rank, ctrb_def=actual_def,
            ))

            # Save per-trial artifacts
            np.savez(out / "estimates" / f"trial_{t:04d}.npz",
                     x0=x0,
                     U_free=U_free, X_free=X_free,
                     U_forced=U_forced, X_forced=X_forced,
                     Ahat=Ahat, Bhat=Bhat, Ad=Ad, Bd=Bd, A=A, B=B,
                     ctrb_rank=actual_rank, ctrb_def=actual_def)

        elif mode == "data":
            # single PRBS run
            U = u_scale * prbs(T, m, rng, dwell=1)
            X  = simulate_dt(Ad, Bd, U, x0)
            if noise_std > 0:
                X = X + noise_std * rng.standard_normal(X.shape)

            X0, X1, Utr = X[:, :-1], X[:, 1:], U.T

            # estimate (Â, B̂)
            if algo == "moesp":
                Ahat, Bhat = moesp_fit(X0, X1, Utr, n=n)
            else:
                Ahat, Bhat = dmdc_pinv(X0, X1, Utr)

            # data-equivalence: residual + optional MP match
            ok_res, info_res = data_equivalence_residual(X0, X1, Utr, Ahat, Bhat, rtol=rtol_resid)
            ok_mp,  info_mp  = markov_match_dt(Ad, Bd, Ahat, Bhat, kmax=kmax_mp, rtol=rtol_eq)
            ok_bool = bool(ok_res and ok_mp)

            diagZ = regressor_stats(X0, Utr, rtol_rank=rtol_rank)
            errA_dt = rel_fro(Ahat, Ad); errB_dt = rel_fro(Bhat, Bd)

            rows.append(dict(
                trial=t, mode=mode, ok=int(ok_bool),
                resid_rel=info_res.get("resid_rel", np.nan),
                markov_err_max=info_mp.get("markov_err_max", np.nan),
                rank=diagZ.get("rank", 0),
                cond=diagZ.get("cond", np.inf),
                smin=diagZ.get("smin", 0.0),
                errA_dt=errA_dt, errB_dt=errB_dt,
                ctrb_rank=actual_rank, ctrb_def=actual_def,
            ))

            # Save per-trial artifacts
            np.savez(out / "estimates" / f"trial_{t:04d}.npz",
                     x0=x0, U=U, X=X,
                     Ahat=Ahat, Bhat=Bhat, Ad=Ad, Bd=Bd, A=A, B=B,
                     ctrb_rank=actual_rank, ctrb_def=actual_def)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    df = pd.DataFrame(rows)
    (out / "summary.csv").parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)

    prop = float(df["ok"].mean()) if len(df) else 0.0
    print(f"[DT] Proportion hits over {trials} trials ({mode}), ctrb_def={actual_def}: {prop:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--algo", type=str, default="dmdc", choices=["dmdc", "moesp"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", type=str, default="out_eqv_cl")

    # modes & thresholds
    ap.add_argument("--mode", type=str, default="theory", choices=["theory", "data"])
    ap.add_argument("--u-scale", type=float, default=5.0)
    ap.add_argument("--T-free", type=int, default=60)
    ap.add_argument("--T-forced", type=int, default=200)
    ap.add_argument("--rtol-eq", type=float, default=1e-2)
    ap.add_argument("--rtol-resid", type=float, default=1e-10)
    ap.add_argument("--rtol-rank", type=float, default=1e-12)
    ap.add_argument("--kmax-mp", type=int, default=4)

    # system size + controllability rank deficiency
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--crank-def", type=int, default=0, help="Controllability rank deficiency (n - rank)")
    ap.add_argument("--max-sys-tries", type=int, default=100)

    args = ap.parse_args()

    out = run(
        trials=args.trials,
        dt=args.dt,
        T=args.T,
        noise_std=args.noise_std,
        algo=args.algo,
        seed=args.seed,
        outdir=args.outdir,
        mode=args.mode,
        u_scale=args.u_scale,
        T_free=args.T_free,
        T_forced=args.T_forced,
        rtol_eq=args.rtol_eq,
        rtol_resid=args.rtol_resid,
        rtol_rank=args.rtol_rank,
        kmax_mp=args.kmax_mp,
        n=args.n,
        m=args.m,
        crank_def=args.crank_def,
        max_sys_tries=args.max_sys_tries,
    )
    print("Saved artifacts under:", out)


if __name__ == "__main__":
    main()
