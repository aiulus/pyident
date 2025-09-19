# pyident/run_single.py
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Sequence
import inspect

from .config import ExpConfig, SolverOpts
from .ensembles import ginibre, sparse_continuous, stable, binary, draw_initial_state
from .signals import prbs, multisine, restrict_pointwise, estimate_pe_order
from .metrics import (
    cont2discrete_zoh,
    gramian_ct_infinite as gramian_ct,
    gramian_dt_finite as gramian_dt,
    projected_errors,
    unified_generator as unified_generator_np,  # keep explicit alias
)
from .loggers.runtime_banner import runtime_banner
from .loggers.tolerances import TolerancePolicy
from .loggers.ledger import start_ledger, attach_tolerances, log_approx, log_warning


# Estimators (DMDc directly; MOESP imported lazily below)
from .estimators.dmdc import dmdc_fit

# Optional JAX accelerator (fully graceful fallback)
try:
    from . import jax_accel as jxa
    _JAX_AVAILABLE = True
except Exception:
    jxa = None
    _JAX_AVAILABLE = False


# --------------------------
# internal small utilities
# --------------------------
def _select_ensemble(cfg: ExpConfig, rng: np.random.Generator):
    if cfg.ensemble == "ginibre":
        return ginibre(cfg.n, cfg.m, rng)
    if cfg.ensemble == "sparse":
        return sparse_continuous(
            n=cfg.n,
            m=cfg.m,
            rng=rng,
            which=cfg.sparse_which,
            p_density=cfg.p_density,
            p_density_A=cfg._density_A if cfg.sparse_which in ("A", "both") else None,
            p_density_B=cfg._density_B if cfg.sparse_which in ("B", "both") else None,
        )
    if cfg.ensemble == "stable":
        return stable(cfg.n, cfg.m, rng)
    if cfg.ensemble == "binary":
        return binary(cfg.n, cfg.m, rng)
    raise ValueError(f"unknown ensemble: {cfg.ensemble}")


def _gen_signal(cfg: ExpConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.signal == "prbs":
        u = prbs(cfg.T, cfg.m, rng, period=max(31, cfg.sigPE))
    elif cfg.signal == "multisine":
        k_lines = max(4, min(cfg.sigPE, cfg.T // 4))
        u = multisine(cfg.T, cfg.m, rng, k_lines=k_lines)
    else:
        raise ValueError(f"unknown signal: {cfg.signal}")

    # Pointwise restriction (project onto span(W))
    if cfg.U_restr is not None:
        u = restrict_pointwise(u, cfg.U_restr)  # (T, m)
    return u


def _simulate_numpy(Ad: np.ndarray, Bd: np.ndarray, u: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """
    Deterministic discrete-time simulation:
        x_{k+1} = Ad x_k + Bd u_k
    Returns X of shape (n, T+1).
    """
    n = Ad.shape[0]
    T = u.shape[0]
    X = np.empty((n, T + 1), dtype=float)
    X[:, 0] = x0
    for k in range(T):
        X[:, k + 1] = Ad @ X[:, k] + Bd @ u[k, :]
    return X


def _basis_from_K_np(K: np.ndarray) -> np.ndarray:
    """
    Thin orthonormal basis of span(K) using SVD (stable).
    """
    if K.size == 0:
        # empty basis, zero-rank
        return np.zeros((K.shape[0], 0))
    U, S, _ = np.linalg.svd(K, full_matrices=False)
    from .loggers.tolerances import TolerancePolicy
    tol = TolerancePolicy()
    r = tol.rank_from_singulars(S)
    return U[:, :r]


def _compute_unified_generator(A: np.ndarray,
                               B: np.ndarray,
                               x0: np.ndarray,
                               mode: str,
                               W: np.ndarray | None,
                               r: int | None,
                               use_jax: bool) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Build the unified generator K(U; x0)
    return (K, rank, Vbasis) with Vbasis a thin basis for span(K).
    """
    if use_jax and _JAX_AVAILABLE and hasattr(jxa, "unified_generator"):
        K = np.asarray(jxa.unified_generator(A, B, x0, mode=mode, W=W, r=r))
        K_rank = int(np.linalg.matrix_rank(K, tol=1e-8))
        # Basis (use NumPy SVD to avoid requiring jnp here)
        Vbasis = _basis_from_K_np(K)
        return K, K_rank, Vbasis

    # NumPy path
    if mode == "unrestricted":
        K = unified_generator_np(A, B, x0, mode="unrestricted")
    elif mode == "pointwise":
        K = unified_generator_np(A, B, x0, mode="pointwise", W=W)
    elif mode == "moment-pe":
        K = unified_generator_np(A, B, x0, mode="moment-pe", r=r)
    else:
        raise ValueError(f"unknown unified_generator mode: {mode}")

    K_rank = int(np.linalg.matrix_rank(K, tol=1e-8))
    Vbasis = _basis_from_K_np(K)
    return K, K_rank, Vbasis


def _pbh_margin_min_sigma(A: np.ndarray,
                          K: np.ndarray,
                          use_jax: bool) -> float:
    """
    Compute min_λ σ_min([λI - A, K]) (unstructured Frobenius distance to PBH failure).
    JAX path uses jax_accelerator if available; otherwise uses NumPy.
    """
    if use_jax and _JAX_AVAILABLE and hasattr(jxa, "pbh_margin_min_sigma"):
        return float(jxa.pbh_margin_min_sigma(A, K))

    # NumPy fallback
    lam = np.linalg.eigvals(A)
    from .loggers.tolerances import TolerancePolicy
    tol = TolerancePolicy()
    lam = tol.cluster_eigs(lam)
    n = A.shape[0]
    margin = np.inf
    for l in lam:
        M = np.concatenate([l * np.eye(n) - A, K], axis=1).astype(np.complex128)
        smin = np.linalg.svd(M, compute_uv=False).min().real
        margin = min(margin, float(smin))
    return float(margin)


# --------------------------
# public entry point
# --------------------------
def run_single(cfg: ExpConfig,
               seed: int,
               sopts: SolverOpts,
               algs: Sequence[str] = ("dmdc", "moesp"),
               use_jax: bool = False) -> Dict[str, Any]:
    """
    Run one experiment instance:
      1) draw (A,B), x0, generate u(t)
      2) ZOH discretize
      3) simulate DT trajectory
      4) compute core objects & metrics (K, PBH margin, Gramian eigen-min, PE order)
      5) run estimators and report errors projected onto span(K)

    Disclaimer:
      - Gramian (CT infinite-horizon) is computed only if A is Hurwitz; otherwise we report None.
      - Projected errors are evaluated in the discrete-time domain (Ad, Bd), but the projection
        uses V = span K(U; x0) as defined in the *continuous* model.
      - PBH “distance to failure” is reported as min_λ σ_min([λI-A, K]); this matches the structured
        test when K is the unified generator (with x0 fixed).
    """
    # --- ledger & tolerances
    _ledger = start_ledger()
    _tol = TolerancePolicy()
    attach_tolerances(_ledger, _tol)
    rng = np.random.default_rng(seed)

    # --- (A,B) and x0
    A, B = _select_ensemble(cfg, rng)
    x0 = draw_initial_state(cfg.n, cfg.x0_mode, rng)

    # --- input u(t)
    u = _gen_signal(cfg, rng)  # (T, m)

    # --- ZOH discretization (exact mapping to DT)
    Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)

    # --- simulate in DT
    if use_jax:
        if not _JAX_AVAILABLE:
            raise RuntimeError("JAX requested but not available. Install jax/jaxlib or omit --use-jax.")
        if hasattr(jxa, "enable_x64"):
            # Use 64-bit for reproducible & stable linear algebra
            jxa.enable_x64(True)
        if hasattr(jxa, "simulate_discrete"):
            X = np.asarray(jxa.simulate_discrete(Ad, Bd, u, x0))  # (n, T+1)
        else:
            # Graceful fallback
            X = _simulate_numpy(Ad, Bd, u, x0)
    else:
        X = _simulate_numpy(Ad, Bd, u, x0)

    # --- regression blocks
    Xtrain = X[:, :-1] # (n, T)
    Xp = X[:,  1:] # (n, T)
    Utrain = u.T # (m, T)


    # --- convenience PE estimate (input-based)
    pe_hat = int(estimate_pe_order(u, s_max=cfg.T // 2, tol=1e-8))

    # --- unified generator mode for analysis 
    mode = "unrestricted"
    Wmat = None
    r = None
    if cfg.U_restr is not None:
        mode = "pointwise"
        Wmat = cfg.U_restr  # basis of allowed directions (m×q)
    if cfg.PE_r is not None:
        mode = "moment-pe"
        r = int(cfg.PE_r)

    # --- build K and span(K) basis
    K, K_rank, Vbasis = _compute_unified_generator(A, B, x0, mode=mode, W=Wmat, r=r, use_jax=use_jax)

    # --- PBH margin min σ_min([λI-A, K]) (structured wrt fixed x0 via K)
    delta_pbh = _pbh_margin_min_sigma(A, K, use_jax=use_jax)

    # --- CT Gramian W (only if A is Hurwitz); returns None otherwise
    gram_min = None
    try:
        Kcore_ct = np.concatenate([x0.reshape(-1, 1), B], axis=1)
        Wct = gramian_ct(A, Kcore_ct)
        if Wct is not None:
            gram_min = float(np.linalg.eigvalsh(Wct).min())
        else:
            from .metrics import gramian_dt_infinite
            Kcore_dt = np.concatenate([x0.reshape(-1, 1), Bd], axis=1)
            Wdt = gramian_dt_infinite(Ad, Kcore_dt)
            if Wdt is not None:
                gram_min = float(np.linalg.eigvalsh(Wdt).min())
    except Exception:
        gram_min = None

    # --- Gramian (CT if Hurwitz; else DT if rho(Ad) < 1)
    Kcore_ct = np.concatenate([x0.reshape(-1, 1), B], axis=1)
    Kcore_dt = np.concatenate([x0.reshape(-1, 1), Bd], axis=1)

    gram_min = None
    gram_mode = "none"
    spec = {}

    Wct = gramian_ct(A, Kcore_ct)
    if Wct is not None:
        try:
            gram_min = float(np.linalg.eigvalsh(Wct).min())
            gram_mode = "CT"
            spec["ct_max_real"] = float(np.max(np.real(np.linalg.eigvals(A))))
        except Exception:
            pass
        try:
            spec["dt_rho"] = float(np.max(np.abs(np.linalg.eigvals(Ad))))
        except Exception:
            pass
    else:
        Wdt = None
        used_finite = False
        try:
            # Try infinite-horizon: signature (Ad, K)
            Wdt = gramian_dt(Ad, Kcore_dt)
            gram_mode = "DT-infinite"
        except TypeError:
            # Fall back to finite-horizon: signature (Ad, K, T)
            Wdt = gramian_dt(Ad, Kcore_dt, int(cfg.T))
            gram_mode = "DT-finite"
            spec["dt_T"] = int(cfg.T)
            used_finite = True
        if Wdt is not None:
            gram_min = float(np.linalg.eigvalsh(Wdt).min())
            # Always report spectral radius for context
            try:
                dt_rho = float(np.max(np.abs(np.linalg.eigvals(Ad))))
                spec["dt_rho"] = dt_rho
                if dt_rho >= 1.0 and used_finite:
                    spec["warning"] = "DT unstable (rho>=1); reported Gramian is finite-horizon."
                    log_warning(_ledger, spec["warning"])
            except Exception:
                pass
    # --- Estimators & projected errors on span(K)
    results_est: Dict[str, Any] = {}

    if "dmdc" in algs:
        try:
            if use_jax and _JAX_AVAILABLE and hasattr(jxa, "dmdc_fit_jax"):
                Ahat, Bhat = jxa.dmdc_fit_jax(Xtrain, Xp, Utrain, rcond=1e-8, ridge=None)
                Ahat, Bhat = np.asarray(Ahat), np.asarray(Bhat)
            else:
                Ahat, Bhat = dmdc_fit(Xtrain, Xp, Utrain)
            # Evaluate in DT (Ad, Bd) projected onto span(K)
            errA, errB = projected_errors(Ahat, Bhat, Ad, Bd, Vbasis=Vbasis)
            results_est["dmdc"] = {"A_err_PV": errA, "B_err_PV": errB}
        except Exception as e:
            results_est["dmdc"] = {"error": str(e)}

    if "moesp" in algs:
        try:
            # Simple PI-MOESP assuming full-state output y = x
            from .estimators.moesp import moesp_fit as moesp_core
            y = X.T  # (T+1, n)
            s_use = max(cfg.n, min(10, cfg.T // 4))  # keep s >= n (heuristic)
            Ahat, Bhat = moesp_core(Xtrain, Xp, Utrain, s=s_use, n=cfg.n)
            errA, errB = projected_errors(Ahat, Bhat, Ad, Bd, Vbasis=Vbasis)
            results_est["moesp"] = {"A_err_PV": errA, "B_err_PV": errB}
        except Exception as e:
            results_est["moesp"] = {"error": str(e)}
    if "gd_dmdc" in algs:
        from .estimators.gradient_based import dmdc_gd_fit as gd_fit
        Ahat, Bhat, diag = gd_fit(
            Xtrain, Xp, Utrain,
            steps=getattr(sopts, "gd_steps", 200),
            rcond=getattr(sopts, "rcond", 1e-10),
            lr=getattr(sopts, "gd_lr", None),
            optimizer=getattr(sopts, "gd_opt", "adam"),
            ridge=getattr(sopts, "gd_ridge", None),
            project_stable=getattr(sopts, "gd_project", None),
            project_params=getattr(sopts, "gd_proj_params", None),
            seed=seed,
            use_jax=use_jax,
            jax_x64=getattr(sopts, "jax_x64", False),
        )
        errA, errB = projected_errors(Ahat, Bhat, Ad, Bd, Vbasis=Vbasis)
        results_est["gd_dmdc"] = {
            "A_err_PV": float(errA),  
            "B_err_PV": float(errB),
            "diag": diag,
         }
        
    _env = runtime_banner()
    # record some approximations 
    log_approx(_ledger, "PBH-margin",
               "computed over clustered eigenvalues; complex SVD; Frobenius-distance proxy")

    return {
        "seed": seed,
        "accelerator": "jax" if (use_jax and _JAX_AVAILABLE) else "numpy",
        "n": cfg.n, "m": cfg.m, "T": cfg.T, "dt": cfg.dt,
        "ensemble": cfg.ensemble, "signal": cfg.signal,
        "sigPE": cfg.sigPE, "pe_order_hat": pe_hat,
        "K_rank": K_rank, "delta_pbh": float(delta_pbh),
        "gram_min": gram_min,
        "gram_mode": gram_mode,   
        "spec": spec,
        "estimators": results_est,
        "env": _env,
        "notes": {"ledger": _ledger},
        # Optional: flags documenting the analytical mode used for K
        "K_mode": mode,
        "K_pointwise_q": None if Wmat is None else int(Wmat.shape[1]),
        "K_PE_r": r,
    }
