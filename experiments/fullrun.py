#!/usr/bin/env python3
"""
Underactuation experiment — single-file CLI script
=================================================
"""

from __future__ import annotations

import os
import sys
import math
import json
import csv
import time
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import numpy.linalg as npl

# Use a non-interactive backend if DISPLAY is not available
import matplotlib
if os.environ.get("DISPLAY", "") == "" and os.environ.get("MPLBACKEND", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import linalg as sla
from scipy.optimize import minimize

np.set_printoptions(precision=4, suppress=True)

# --------------------------- Core metrics and helpers ------------------------

@dataclass
class SolverOpts:
    maxit: int = 150
    tol_grad: float = 1e-7
    num_seeds: int = 8
    random_state: Optional[int] = 0


def build_Q(x0: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Build an orthonormal basis Q for the subspace orthogonal to x0."""
    n = x0.size
    if npl.norm(x0) <= tol:
        return np.eye(n, dtype=x0.dtype)
    row = x0.conj().reshape(1, -1)
    Q = sla.null_space(row)
    if Q.shape[1] < n - 1:
        # Fallback construction
        q0 = x0 / npl.norm(x0)
        M = np.eye(n, dtype=x0.dtype) - np.outer(q0, q0.conj())
        Q, _ = npl.qr(M)
        Q = Q[:, : n - 1]
    return Q


def concat_lambda_block(A: np.ndarray, B: np.ndarray, lam: complex) -> np.ndarray:
    n = A.shape[0]
    return np.concatenate((lam * np.eye(n) - A, B), axis=1)


def _phi_and_grad(ab: np.ndarray, Q: np.ndarray, A: np.ndarray, B: np.ndarray):
    n = A.shape[0]
    alpha, beta = float(ab[0]), float(ab[1])
    lam = alpha + 1j * beta
    H = concat_lambda_block(A, B, lam)
    M = Q.conj().T @ H
    U, s, Vh = sla.svd(M, full_matrices=False, lapack_driver="gesvd")
    sigma = s[-1]
    u = U[:, -1]
    v = Vh.conj().T[:, -1]
    v_first = v[:n]
    c = (Q @ u).conj().T @ v_first
    grad = np.array([np.real(c), -np.imag(c)], dtype=float)
    return float(sigma), grad, u, v


def init_lambda_seeds(A: np.ndarray, opts: SolverOpts) -> np.ndarray:
    rng = np.random.default_rng(opts.random_state)
    eigs = sla.eigvals(A)
    seeds = list(eigs)
    for _ in range(max(0, opts.num_seeds - len(seeds))):
        z = rng.normal(size=A.shape[0]) + 1j * rng.normal(size=A.shape[0])
        z /= npl.norm(z)
        seeds.append(complex(np.vdot(z, A @ z)))
    return np.array(seeds)


def get_d_frob(A: np.ndarray, B: np.ndarray, x0: np.ndarray, opts: Optional[SolverOpts] = None):
    if opts is None:
        opts = SolverOpts()
    Q = build_Q(x0)
    best_sigma = np.inf
    best_lam = 0.0 + 0.0j
    for lam0 in init_lambda_seeds(A, opts):
        ab0 = np.array([np.real(lam0), np.imag(lam0)], dtype=float)
        res = minimize(
            lambda ab: _phi_and_grad(ab, Q, A, B)[0],
            ab0,
            jac=lambda ab: _phi_and_grad(ab, Q, A, B)[1],
            method="BFGS",
            options=dict(maxiter=opts.maxit, gtol=opts.tol_grad),
        )
        sigma = float(res.fun)
        if sigma < best_sigma:
            best_sigma = sigma
            best_lam = complex(res.x[0], res.x[1])
    M = Q.conj().T @ concat_lambda_block(A, B, best_lam)
    U, s, Vh = sla.svd(M, full_matrices=False, lapack_driver="gesvd")
    u_min = U[:, -1]
    w_star = Q @ u_min
    w_star = np.real_if_close(w_star)
    return float(best_sigma), complex(best_lam), w_star


def krylov_metrics(A: np.ndarray, B: np.ndarray, x0: np.ndarray) -> Dict[str, Any]:
    n = A.shape[0]
    cols0 = np.concatenate((x0.reshape(-1, 1), B), axis=1)
    K_blocks = []
    Ak = np.eye(n)
    for k in range(n):
        if k == 0:
            K_blocks.append(cols0)
        else:
            Ak = Ak @ A
            K_blocks.append(Ak @ cols0)
    K = np.concatenate(K_blocks, axis=1)
    s = sla.svdvals(K)
    sigma_min = float(s[-1])
    rank = int((s > 1e-10).sum())
    return dict(K=K, sigma_min=sigma_min, rank=rank)


def controllable_subspace_basis(A: np.ndarray, B: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    n = A.shape[0]
    C = B.copy()
    M = C.copy()
    for _ in range(1, n):
        C = A @ C
        M = np.concatenate((M, C), axis=1)
    U, s, _ = sla.svd(M, full_matrices=False)
    r = (s > tol).sum()
    return U[:, :r]


def subspace_angle_metrics(A: np.ndarray, B: np.ndarray, x0: np.ndarray) -> Dict[str, Any]:
    R = controllable_subspace_basis(A, B)
    if R.size == 0:
        return dict(eta0=0.0, theta0=float(np.pi / 2), R_basis=R)
    PR = R @ (R.T @ x0)
    eta0 = float(np.linalg.norm(PR) / max(np.linalg.norm(x0), 1e-16))
    eta0 = min(max(eta0, 0.0), 1.0)
    theta0 = float(np.arccos(eta0))
    return dict(eta0=eta0, theta0=theta0, R_basis=R)


def modewise_overlaps(
    A: np.ndarray, B: np.ndarray, x0: np.ndarray, real_only: bool = True
) -> Dict[str, Any]:
    wvals, W = sla.eig(A.conj().T)
    alphas, betas, mus = [], [], []
    for i in range(W.shape[1]):
        w = W[:, i]
        if real_only:
            w = np.real_if_close(w)
        denom = np.sqrt(np.real((w.conj().T @ w))) + 1e-12
        wt = w.conj().T
        alpha = float(np.abs(wt @ x0) / denom)
        beta = float(np.linalg.norm(wt @ B) / denom)
        mu = float(
            np.linalg.norm(np.concatenate(((wt @ x0).reshape(1,), wt @ B)))
        ) / denom
        alphas.append(alpha)
        betas.append(beta)
        mus.append(mu)
    mu_min = float(np.min(mus)) if mus else 0.0
    return dict(alpha=np.array(alphas), beta=np.array(betas), mu_min=mu_min, eigenvalues=wvals)


# --------------------------- Ensembles (dense) -------------------------------

def ginibre(n: int, m: int, rng: np.random.Generator):
    A = rng.normal(size=(n, n)) / np.sqrt(n)
    B = rng.normal(size=(n, m)) / np.sqrt(n)
    return A, B


def c2d(A: np.ndarray, B: np.ndarray, dt: float):
    n, m = B.shape
    M = np.block([[A, B], [np.zeros((m, n + m))]])
    E = sla.expm(M * dt)
    Ad = E[:n, :n]
    Bd = E[:n, n : n + m]
    return Ad, Bd


# --------------------------- Estimators --------------------------------------

class DMDC:
    def fit(self, X: np.ndarray, U: np.ndarray):
        X0, X1 = X[:, :-1], X[:, 1:]
        Theta = np.vstack([X0, U])
        AB = X1 @ np.linalg.pinv(Theta)
        n = X0.shape[0]
        self.A_hat = AB[:, :n]
        self.B_hat = AB[:, n:]
        return self


class MOESP:
    def __init__(self, i: int, s: int):
        self.i, self.s = i, s

    def fit(self, X: np.ndarray, U: np.ndarray):
        n, T1 = X.shape
        T = T1 - 1
        i, s = self.i, self.s
        L = T - (i + s) + 1
        if L <= max(2 * n, 5):
            return DMDC().fit(X, U)
        Yf = []
        Up = []
        for k in range(i):
            Yf.append(X[:, s + k : s + k + L])
        for k in range(s):
            Up.append(U[:, k : k + L])
        Yf = np.vstack(Yf)
        Up = np.vstack(Up)
        Q, _ = npl.qr(Up.T, mode="reduced")
        Pf = Yf @ Q @ Q.T
        Uo, S, Vh = npl.svd(Pf, full_matrices=False)
        Un = Uo[:, :n]
        Ob = Un.reshape(i, n, -1).transpose(0, 2, 1)
        O1 = Ob[:-1].reshape(-1, n)
        O2 = Ob[1:].reshape(-1, n)
        A_hat, *_ = npl.lstsq(O1, O2, rcond=None)
        self.A_hat = A_hat.T
        X0, X1 = X[:, :T], X[:, 1 : T + 1]
        Theta = np.vstack([X0, U[:, :T]])
        AB = X1 @ npl.pinv(Theta)
        self.B_hat = AB[:, n:]
        return self


# --------------------------- Input signal ------------------------------------

def prbs(
    m: int, T: int, levels: int = 2, period: int = 7, rng: Optional[np.random.Generator] = None
):
    rng = rng or np.random.default_rng()
    U = np.zeros((m, T))
    for i in range(m):
        bits = rng.integers(0, levels, size=T) * 2 - (levels - 1)
        for t in range(0, T, period):
            bits[t : t + period] = bits[t]
        U[i, :] = bits
    return U


# --------------------------- Utilities ---------------------------------------

def ci95(arr: List[float] | np.ndarray):
    x = np.asarray(arr)
    mu = float(np.mean(x))
    s = float(np.std(x, ddof=1) if len(x) > 1 else 0.0)
    half = 1.96 * s / np.sqrt(max(1, len(x)))
    return mu, mu - half, mu + half


def parse_m_sweeps(specs: List[str]) -> Dict[int, List[int]]:
    """Parse m-sweeps like ["3:1,2,3", "10:1,3,5,10"]."""
    out: Dict[int, List[int]] = {}
    for s in specs:
        if ":" not in s:
            raise ValueError(f"Invalid m-sweep '{s}'. Expected format 'n:a,b,c'.")
        ns, ms = s.split(":", 1)
        n = int(ns.strip())
        m_list = [int(x.strip()) for x in ms.split(",") if x.strip()]
        if not m_list:
            raise ValueError(f"No m values provided for n={n}.")
        out[n] = m_list
    return out


# --------------------------- Experiment runner -------------------------------

def run_underactuation(
    include_moesp: bool,
    n_list: List[int],
    m_sweeps: Dict[int, List[int]],
    num_instances: int,
    repeats: int,
    T: int,
    dt: float,
    x0_mode: str,
    seed: int,
    sopts: SolverOpts,
    prbs_levels: int,
    prbs_period: int,
    save_plots: bool,
    output_dir: Optional[str],
) -> Dict[str, Any]:
    # RNG for instances
    rng0 = np.random.default_rng(seed)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    summary_all: Dict[str, Any] = dict(meta=dict(
        include_moesp=include_moesp,
        n_list=n_list,
        m_sweeps=m_sweeps,
        num_instances=num_instances,
        repeats=repeats,
        T=T,
        dt=dt,
        x0_mode=x0_mode,
        seed=seed,
        solver_opts=vars(sopts),
        prbs_levels=prbs_levels,
        prbs_period=prbs_period,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    ))

    for n in n_list:
        if n not in m_sweeps:
            raise ValueError(f"No m-sweep specified for n={n}.")
        m_list = m_sweeps[n]

        metrics_per_m = {"d_frob": [], "sigma_min_K": [], "mu_min": []}
        var_per_algo = {"dmdc": []}
        if include_moesp:
            var_per_algo["moesp"] = []
        krylov_rank_m: List[List[int]] = []
        contr_rank_m: List[List[int]] = []

        print(f"\n=== Running experiments for n={n} ===")
        for m in m_list:
            print(f"- m={m}")
            dfix_all: List[float] = []
            sigK_all: List[float] = []
            mu_all: List[float] = []
            ranksK: List[int] = []
            ranksC: List[int] = []
            err_alg = {a: [] for a in var_per_algo.keys()}

            for inst in range(num_instances):
                rng = np.random.default_rng(rng0.integers(0, 2**31 - 1))
                A, B = ginibre(n, m, rng)
                if x0_mode == "gaussian":
                    x0 = rng.normal(size=(n,))
                else:
                    raise ValueError(f"Unsupported x0_mode '{x0_mode}'. Only 'gaussian' is implemented.")

                # Metrics
                dfix, _, _ = get_d_frob(A, B, x0, sopts)
                K = krylov_metrics(A, B, x0)
                sigK = K["sigma_min"]
                rankK = K["rank"]
                mu_min = modewise_overlaps(A, B, x0)["mu_min"]
                R = controllable_subspace_basis(A, B)
                rankC = R.shape[1]

                dfix_all.append(dfix)
                sigK_all.append(sigK)
                mu_all.append(mu_min)
                ranksK.append(rankK)
                ranksC.append(rankC)

                # Identification repeats
                Ad, Bd = c2d(A, B, dt)
                for rep in range(repeats):
                    U = prbs(m, T, levels=prbs_levels, period=prbs_period, rng=rng)
                    X = np.zeros((n, T + 1))
                    X[:, 0] = x0
                    for k in range(T):
                        X[:, k + 1] = Ad @ X[:, k] + Bd @ U[:, k]
                    # DMDC
                    est = DMDC().fit(X, U)
                    eA = npl.norm(est.A_hat - Ad, "fro")
                    eB = npl.norm(est.B_hat - Bd, "fro")
                    err_alg["dmdc"].append(float((eA * eA + eB * eB) ** 0.5))
                    # MOESP (optional)
                    if include_moesp:
                        est2 = MOESP(i=2 * n, s=2 * n).fit(X, U)
                        eA2 = npl.norm(est2.A_hat - Ad, "fro")
                        eB2 = npl.norm(est2.B_hat - Bd, "fro")
                        err_alg["moesp"].append(float((eA2 * eA2 + eB2 * eB2) ** 0.5))

            metrics_per_m["d_frob"].append(dfix_all)
            metrics_per_m["sigma_min_K"].append(sigK_all)
            metrics_per_m["mu_min"].append(mu_all)
            krylov_rank_m.append(ranksK)
            contr_rank_m.append(ranksC)
            for a in var_per_algo.keys():
                var_per_algo[a].append(err_alg[a])

        # ---- Plot (two rows) ----
        # For headless environments, plots are only saved when requested
        if save_plots:
            m_axis = np.array(m_list)
            ncols = max(3, len(var_per_algo))
            fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 7), sharex="col")
            metric_keys = ["d_frob", "sigma_min_K", "mu_min"]
            titles = [r"$\\d{\\mathrm{Frob}}$", r"$\\sigma_{\\min}(K)$", r"$\\mu_{\\min}$"]
            for j, (k, title) in enumerate(zip(metric_keys, titles)):
                means, lo, hi = [], [], []
                for arr in metrics_per_m[k]:
                    mu, l, h = ci95(arr)
                    means.append(mu)
                    lo.append(l)
                    hi.append(h)
                ax = axes[0, j]
                ax.plot(m_axis, means, marker="o")
                ax.fill_between(m_axis, lo, hi, alpha=0.2)
                ax.set_title(title)
                ax.set_xlabel("m")
                ax.set_ylabel("mean ± 95% CI")
            for j, a in enumerate(var_per_algo.keys()):
                stds = [float(np.std(v, ddof=1)) for v in var_per_algo[a]]
                ax = axes[1, j]
                ax.plot(m_axis, stds, marker="o")
                ax.set_title(f"{a} variability (std of joint err)")
                ax.set_xlabel("m")
            fig.suptitle(f"Underactuation experiment (n={n})")
            fig.tight_layout()
            if output_dir:
                fig_path = os.path.join(output_dir, f"underactuation_n{n}.png")
                fig.savefig(fig_path, dpi=150)
                print(f"Saved figure: {fig_path}")
            plt.close(fig)

        # ---- Summaries to JSON ----
        def summarize_metric_block(block: List[List[float]]):
            out = []
            for arr in block:
                mu, l, h = ci95(arr)
                out.append(dict(mean=mu, ci95_lo=l, ci95_hi=h, n=len(arr)))
            return out

        def summarize_var_block(block: List[List[float]]):
            out = []
            for arr in block:
                out.append(dict(std=float(np.std(arr, ddof=1)), n=len(arr)))
            return out

        n_key = f"n={n}"
        summary_all[n_key] = dict(
            m_list=m_list,
            metrics=dict(
                d_frob=summarize_metric_block(metrics_per_m["d_frob"]),
                sigma_min_K=summarize_metric_block(metrics_per_m["sigma_min_K"]),
                mu_min=summarize_metric_block(metrics_per_m["mu_min"]),
            ),
            variability={a: summarize_var_block(var_per_algo[a]) for a in var_per_algo},
            krylov_rank=krylov_rank_m,
            controllable_rank=contr_rank_m,
        )

        if output_dir:
            json_path = os.path.join(output_dir, f"summary_n{n}.json")
            with open(json_path, "w") as f:
                json.dump(summary_all[n_key], f, indent=2)
            print(f"Saved JSON summary: {json_path}")

    # Save a combined meta+all summary if requested
    if output_dir:
        all_path = os.path.join(output_dir, "summary_all.json")
        with open(all_path, "w") as f:
            json.dump(summary_all, f, indent=2)
        print(f"Saved combined JSON: {all_path}")

    return summary_all


# --------------------------- CLI --------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run underactuation experiments with CLI-configurable parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-list", type=int, nargs="+", default=[3, 10],
                   help="List of state sizes n to run.")
    p.add_argument(
        "--m-sweeps",
        type=str,
        nargs="*",
        default=["3:1,2,3,4,5,6", "10:1,3,5,10,15,20"],
        help="Per-n sweep of m values as 'n:a,b,c'. Provide multiple entries.",
    )

    moesp_group = p.add_mutually_exclusive_group()
    moesp_group.add_argument("--moesp", dest="include_moesp", action="store_true",
                             help="Include MOESP estimator.")
    moesp_group.add_argument("--no-moesp", dest="include_moesp", action="store_false",
                             help="Disable MOESP estimator.")
    p.set_defaults(include_moesp=True)

    p.add_argument("--num-instances", type=int, default=10,
                   help="Number of random (A,B,x0) instances per (n,m).")
    p.add_argument("--repeats", type=int, default=10,
                   help="Identification repeats per instance (PRBS draws).")
    p.add_argument("--T", type=int, default=400, help="Horizon (timesteps).")
    p.add_argument("--dt", type=float, default=0.02, help="Sampling period for c2d().")
    p.add_argument("--x0-mode", type=str, default="gaussian",
                   choices=["gaussian"], help="Initial state distribution.")
    p.add_argument("--seed", type=int, default=0, help="Top-level RNG seed.")

    # Solver options
    p.add_argument("--solver-num-seeds", type=int, default=8,
                   help="Number of lambda seeds for optimization.")
    p.add_argument("--solver-maxit", type=int, default=120,
                   help="Max iterations for BFGS.")
    p.add_argument("--solver-tol-grad", type=float, default=1e-7,
                   help="Gradient tolerance for BFGS.")
    p.add_argument("--solver-random-state", type=int, default=None, nargs="?",
                   help="Random state for lambda seeding (None => random).")

    # PRBS options
    p.add_argument("--prbs-levels", type=int, default=2, help="PRBS amplitude levels.")
    p.add_argument("--prbs-period", type=int, default=7, help="PRBS hold period.")

    # Output / plotting
    p.add_argument("--output-dir", type=str, default="out",
                   help="Directory to save JSON (and plots, if enabled).")
    p.add_argument("--save-plots", action="store_true",
                   help="Generate and save plots (PNG) in output-dir.")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    m_sweeps = parse_m_sweeps(args.m_sweeps)

    sopts = SolverOpts(
        maxit=args.solver_maxit,
        tol_grad=args.solver_tol_grad,
        num_seeds=args.solver_num_seeds,
        random_state=args.solver_random_state,
    )

    _ = run_underactuation(
        include_moesp=args.include_moesp,
        n_list=args.n_list,
        m_sweeps=m_sweeps,
        num_instances=args.num_instances,
        repeats=args.repeats,
        T=args.T,
        dt=args.dt,
        x0_mode=args.x0_mode,
        seed=args.seed,
        sopts=sopts,
        prbs_levels=args.prbs_levels,
        prbs_period=args.prbs_period,
        save_plots=args.save_plots,
        output_dir=args.output_dir,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
