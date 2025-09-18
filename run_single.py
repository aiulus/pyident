# pyident/run_single.py
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Sequence
from .config import ExpConfig, SolverOpts, RunMeta
from .ensembles import ginibre, sparse_continuous, stable, binary, draw_initial_state
from .signals import prbs, multisine, restrict_pointwise, pe_order_estimate
from .metrics import cont2discrete_zoh, krylov_generator, subspace_dimension, gramian_ct, projected_errors
from .pbh import unified_generator, pbh_margin
from .estimators import dmdc_fit, moesp_fit
from .sys_utils import simulate

def _select_ensemble(cfg: ExpConfig, rng: np.random.Generator):
    if cfg.ensemble == "ginibre":
        return ginibre(cfg.n, cfg.m, rng)
    if cfg.ensemble == "sparse":
        return sparse_continuous(cfg.n, cfg.m, cfg.sparsity_p, rng)
    if cfg.ensemble == "stable":
        return stable(cfg.n, cfg.m, rng)
    if cfg.ensemble == "binary":
        return binary(cfg.n, cfg.m, rng)
    raise ValueError(cfg.ensemble)

def _gen_signal(cfg: ExpConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.signal == "prbs":
        u = prbs(cfg.horizon, cfg.m, rng, period=max(31, cfg.pe_order_target))
    elif cfg.signal == "multisine":
        k_lines = max(4, min(cfg.pe_order_target, cfg.horizon//4))
        u = multisine(cfg.horizon, cfg.m, rng, k_lines=k_lines)
    else:
        raise ValueError(cfg.signal)
    return u

def run_single(cfg: ExpConfig, seed: int, sopts: SolverOpts, estimators: Sequence[str] = ("dmdc","moesp")) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    A, B = _select_ensemble(cfg, rng)
    x0 = draw_initial_state(cfg.n, cfg.x0_mode, rng)

    # --- build input signal ---
    u = _gen_signal(cfg, rng)
    if cfg.U_restr is not None:
        u = restrict_pointwise(u, cfg.U_restr)
    
    # --- ZOH discretization ---
    Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)

    X = simulate(cfg.T, x0, Ad, Bd, u)
    Xtrain = X[:, :-1]
    Xp = X[:, 1:]
    Utrain = u[:-1, :].T

    pe_hat = pe_order_estimate(Xtrain, Utrain, cfg.m, max_order=cfg.T//2)

    # --- build the unified generator K(U; x0) ---
    mode = "unrestricted"
    Wmat = None
    r = None
    if cfg.PE_r is not None:
        mode = "pointwise"
        m = B.shape[1]
        Wmat = np.eye(m)[:, :cfg.PE_r]
    if cfg.momPE is not None:
        mode = "moment-pe"
        r = cfg.momPE
        
    K = unified_generator(A, B, x0, mode=mode, W=Wmat, r=r)
    K_rank = int(np.linalg.matrix_rank(K, tol=1e-8))

    # --- compute metrics ---
    # --- 1. Frobenius distance to PBH failure ---
    d_frob = pbh_margin(A, B, x0, K)
    # 2. --- Continuous Gramian condition number ---
    W = None
    gram_min = None
    try:
        W = gramian_ct(A, K)
        if W is not None:
            gram_min = float(np.linalg.eigvalsh(W).min())
    except Exception:
        pass

    # --- Run identification algorithms and evaluate ---
    results_est = {}
    if "dmdc" in estimators:
        Ahat, Bhat = dmdc_fit(Xtrain, Xp, Utrain)
        errA, errB = projected_errors(Ahat, Bhat, A, B, Vbasis=_basis_from_K(A, B, x0, mode, Wmat, r))
        results_est["dmdc"] = {"A_err_PV": errA, "B_err_PV": errB}
    if "moesp" in estimators:
        Ahat, Bhat = moesp_fit(Xtrain, Xp, Utrain, s=min(10, T//4))
        errA, errB = projected_errors(Ahat, Bhat, A, B, Vbasis=_basis_from_K(A, B, x0, mode, Wmat, r))
        results_est["moesp"] = {"A_err_PV": errA, "B_err_PV": errB}

    return {
        "seed": seed,
        "n": cfg.n, "m": cfg.m, "T": cfg.horizon, "dt": cfg.dt,
        "ensemble": cfg.ensemble, "signal": cfg.signal,
        "sigPE": cfg.sigPE, "pe_order_hat": pe_hat,
        "K_rank": K_rank, "delta_pbh": float(d_frob),
        "gram_min": gram_min,
        "estimators": results_est,
    }

def _basis_from_K(A: np.ndarray, B: np.ndarray, x0: np.ndarray, 
                  mode: str, W: np.ndarray | None, r: int | None) -> np.ndarray:
    K = unified_generator(A, B, x0, mode=mode, W=W, r=r)
    # thin basis via SVD
    U, S, Vt = np.linalg.svd(K, full_matrices=False)
    rank = np.linalg.matrix_rank(K, tol=1e-8)
    return U[:, :rank]
                 
