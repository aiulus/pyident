"""
Identifiability criteria vs. estimation error (DD-ID setting, with inputs).

What it does (faithful to the manuscript + your request):
  1) Fix (A,B) with controllability rank n-1 (barely uncontrollable) using `ensembles.draw_with_ctrb_rank`.
  2) For each trial: draw x0 ~ Unif(S^{n-1}), simulate DT system under PRBS inputs,
     estimate (A,B) using DMDc (pinv) and MOESP (full-state wrapper).
  3) Compute identifiability/robustness criteria:
        (a) structured PBH margin proxy: delta_pbh = pbh_margin_structured(A,B,x0)
        (b) Krylov generator statistics (we use sigma_min(K_n))
        (c) left-eigenvector criterion: mu_min = min_i ||w_i^T [x0 B]|| / ||w_i||
  4) Transform metrics per your request and plot on the x-axis:
        X1 = 1 / delta_pbh,   X2 = sigma_min(K_n),   X3 = 1 / mu_min
     Y = REE(Ahat) against the *discrete-time* A_d (Frobenius relative error).
  5) Compute Spearman rho + p-values and save CSVs + scatter plots.

Outputs:
  out_dir / (CSV: trial-wise results, CSV: spearman, PNGs: scatter plots)
"""
from __future__ import annotations
import argparse, math, sys, os, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from ..metrics import(
    pbh_margin_structured,
    unified_generator,
    left_eigvec_overlap,
    cont2discrete_zoh,
    pair_distance
)
from ..estimators import(
    dmdc_pinv, 
    moesp_fit
)
from ..ensembles import(
    draw_with_ctrb_rank,
    sparse_continuous_column
)
  

# Local modules expected alongside this script or in sys.path

def controllable_subspace(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Returns an orthonormal basis for the controllable subspace Im([B, AB, A^2B]).
    """
    n = A.shape[0]
    m = B.shape[1] if B.ndim > 1 else 1
    AB = A @ B
    A2B = A @ AB
    ctrb_mat = np.hstack([B.reshape(n, m), AB.reshape(n, m), A2B.reshape(n, m)])
    # Orthonormal basis via QR
    Q, _ = np.linalg.qr(ctrb_mat)
    return Q

def projection_norm(x: np.ndarray, Q: np.ndarray) -> float:
    """
    Returns the norm of the projection of x onto the subspace spanned by columns of Q, normalized by ||x||.
    """
    proj = Q @ (Q.T @ x)
    return float(np.linalg.norm(proj) / np.linalg.norm(x))

def plot_pbh_vs_proj(n_samples=200, seed=42):
    np.random.seed(seed)
    # Fixed system
    A = np.array([[1, 1, 0],
                  [0, 2, 1],
                  [0, 0, 3]], dtype=float)
    B = np.array([[0, 1, 1]]).T  # shape (3,1)
    Q = controllable_subspace(A, B)
    pbh_scores = []
    proj_norms = []
    for _ in range(n_samples):
        # Use sparse_continuous from ensembles.py to sample x0
        x0 = np.asarray(sparse_continuous_column(3, rng=np.random.default_rng(), p_density=0.7))
        x0 /= np.linalg.norm(x0)
        proj_norm = projection_norm(x0, Q)
        score = pbh_margin_structured(A, B, x0)
        proj_norms.append(proj_norm)
        pbh_scores.append(score)
    plt.figure(figsize=(5.2, 4.0))
    plt.scatter(proj_norms, pbh_scores, s=18)
    plt.xlabel("Normalized projection of $x_0$ onto controllable subspace")
    plt.ylabel("PBH structured margin")
    plt.title("PBH margin vs. controllable subspace projection")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_pbh_vs_proj()
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
sys.path.append("/mnt/data")  # fallback for the uploaded modules


def sample_unit_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(n)
    nrm = np.linalg.norm(v)
    if nrm == 0.0:
        return sample_unit_sphere(n, rng)
    return v / nrm


def prbs(T: int, m: int, rng: np.random.Generator, dwell: int = 1) -> np.ndarray:
    """±1 PRBS with optional dwell; shape (T,m)."""
    steps = math.ceil(T / dwell)
    seq = rng.choice([-1.0, 1.0], size=(steps, m))
    u = np.repeat(seq, repeats=dwell, axis=0)[:T, :]
    return u


def simulate_dt(Ad: np.ndarray, Bd: np.ndarray, u: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """
    Simulate x_{k+1} = Ad x_k + Bd u_k.
    Returns X with shape (n, T+1).
    """
    n = Ad.shape[0]
    T = u.shape[0]
    X = np.empty((n, T + 1), dtype=float)
    X[:, 0] = x0
    for k in range(T):
        X[:, k + 1] = Ad @ X[:, k] + Bd @ u[k, :]
    return X

def relative_error_fro(Ahat: np.ndarray, Bhat: np.ndarray, Atrue: np.ndarray, Btrue:  np.ndarray) -> float:
    return pair_distance(Ahat, Bhat, Atrue, Btrue)


def compute_core_metrics(A: np.ndarray, B: np.ndarray, x0: np.ndarray) -> dict:
    """
    Compute the three manuscript-linked criteria we need:
      - PBH structured margin proxy: pbh_struct
      - Krylov sigma_min on unified generator K (unrestricted)
      - Left-eigen overlap mu_min for Xaug=[x0 B]
    """
    pbh_struct = float(pbh_margin_structured(A, B, x0))

    # unified generator (unrestricted) over A,B,x0
    K = unified_generator(A, B, x0, mode="unrestricted")
    svals = np.linalg.svd(K, compute_uv=False)
    krylov_smin = float(svals.min()) if svals.size else 0.0

    Xaug = np.concatenate([x0.reshape(-1, 1), B], axis=1)
    mu = left_eigvec_overlap(A, Xaug)
    mu_min = float(np.min(mu)) if mu.size else 0.0

    return dict(pbh_struct=pbh_struct, krylov_smin=krylov_smin, mu_min=mu_min)


def run_trials(n=6, m=2, T=200, dt=0.05, trials=64, noise_std=0.0, seed=123):
    rng = np.random.default_rng(seed)

    # Fix (A,B) with controllability rank n-1.
    A, B, meta = draw_with_ctrb_rank(
        n, m, r=n - 1, rng=rng,
        ensemble_type="ginibre", base_u="ginibre", embed_random_basis=True
    )
    Ad, Bd = cont2discrete_zoh(A, B, dt)

    rows = []
    for t in range(trials):
        x0 = sample_unit_sphere(n, rng)
        u = prbs(T, m, rng, dwell=1)  # shape (T,m)
        X = simulate_dt(Ad, Bd, u, x0)
        if noise_std > 0:
            X = X + noise_std * rng.standard_normal(X.shape)

        Xtrain, Xp, Utrain = X[:, :-1], X[:, 1:], u.T

        # Estimators (discrete-time): DMDc (pinv) and MOESP full-state wrapper
        A_dmdc, B_dmdc = dmdc_pinv(Xtrain, Xp, Utrain)
        A_moesp, B_moesp = moesp_fit(Xtrain, Xp, Utrain, n=n)

        # Criteria
        crit = compute_core_metrics(A, B, x0)

        # Relative errors vs Ad
        ree_dmdc = relative_error_fro(A_dmdc, B_dmdc, Ad, Bd)
        ree_moesp = relative_error_fro(A_moesp, B_moesp, Ad, Bd)

        rows.append(dict(trial=t, **crit, errA_dmdc=ree_dmdc, errA_moesp=ree_moesp))

    df = pd.DataFrame(rows)
    return df, {"A": A, "B": B, "Ad": Ad, "Bd": Bd, "meta": meta}


def add_transforms(df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """
    Add the three x-axis transforms the user requested:
      X1 = 1 / pbh_struct
      X2 = krylov_smin (no transform)
      X3 = 1 / mu_min
    """
    df = df.copy()
    df["x_inv_pbh"] = 1.0 / np.maximum(df["pbh_struct"].to_numpy(), eps)
    df["x_inv_krylov_smin"] = 1.0 / df["krylov_smin"].to_numpy()
    df["x_inv_mu"] = 1.0 / np.maximum(df["mu_min"].to_numpy(), eps)
    return df


def spearman_table(df: pd.DataFrame, ycols=("errA_dmdc", "errA_moesp")) -> pd.DataFrame:
    xcols = [("x_inv_pbh", "1 / PBH metric"),
             ("x_inv_krylov_smin", "σ_min(K_n)"),
             ("x_inv_mu", "1 / mu_min")]
    records = []
    for x, _ in xcols:
        rec = {"metric": x}
        for y in ycols:
            rho, p = spearmanr(df[x], df[y])
            rec[f"{y}_rho"] = str(rho)
            rec[f"{y}_p"] = str(p)
        records.append(rec)
    return pd.DataFrame(records)


def scatter_plots(df: pd.DataFrame, ykey: str, outdir: pathlib.Path, tag: str):
    pairs = [
        ("x_inv_pbh", "1 / PBH structured margin"),
        ("x_inv_krylov_smin", "σ_min(K_n)"),
        ("x_inv_mu", "1 / mu_min (left-eig overlap)"),
    ]
    for x, xlabel in pairs:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        ax.scatter(df[x].to_numpy(), df[ykey].to_numpy(), s=18)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"REE({ykey.replace('errA_', '')})")
        ax.set_title(f"{xlabel} vs. {ykey}")
        fig.savefig(outdir / f"{x}_vs_{ykey}_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--trials", type=int, default=64)
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=31415)
    ap.add_argument("--outdir", type=str, default="ident_vs_error_out")
    ap.add_argument("--tag", type=str, default="demo")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    plotdir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)

    df, sysinfo = run_trials(n=args.n, m=args.m, T=args.T, dt=args.dt,
                             trials=args.trials, noise_std=args.noise_std, seed=args.seed)
    df = add_transforms(df)

    # Save trial-wise results
    df.to_csv(outdir / f"results_{args.tag}.csv", index=False)

    # Spearman table
    stab = spearman_table(df)
    stab.to_csv(outdir / f"spearman_{args.tag}.csv", index=False)

    # Plots for both algos
    scatter_plots(df, "errA_dmdc", plotdir, args.tag)
    scatter_plots(df, "errA_moesp", plotdir, args.tag)

    # Minimal metadata
    meta_path = outdir / f"system_{args.tag}.npz"
    np.savez(meta_path, A=sysinfo["A"], B=sysinfo["B"], Ad=sysinfo["Ad"], Bd=sysinfo["Bd"])

    print("Saved:")
    print("  ", outdir / f"results_{args.tag}.csv")
    print("  ", outdir / f"spearman_{args.tag}.csv")
    print("  ", meta_path)
    print("  plots ->", plotdir)


if __name__ == "__main__":
    main()
