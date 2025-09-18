# pyident/run_single.py
from __future__ import annotations
from typing import Dict, Any, Sequence, Optional, Tuple
import numpy as np

from .config import ExpConfig, SolverOpts
from .ensembles import ginibre, sparse_continuous, stable, binary, draw_initial_state
from .signals import (
    prbs,
    multisine,
    restrict_pointwise,
    estimate_pe_order_block,
    estimate_moment_pe_order,
)
from .metrics import (
    cont2discrete_zoh,
    unified_generator,
    visible_subspace,
    gramian_ct_infinite,
    gramian_dt_finite,
    pbh_margin_structured,
    pbh_margin_unstructured,
    projected_errors,
)


# -------------------------- helpers ----------------------------------

def _select_ensemble(cfg: ExpConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if cfg.ensemble == "ginibre":
        return ginibre(cfg.n, cfg.m, rng)
    if cfg.ensemble == "sparse":
        densA = cfg.p_density if cfg.p_density_A is None else cfg.p_density_A
        densB = cfg.p_density if cfg.p_density_B is None else cfg.p_density_B
        return sparse_continuous(
            cfg.n,
            cfg.m,
            p_density=densA,
            rng=rng,
            which=cfg.sparse_which,
            b_density=densB if cfg.sparse_which in ("B", "both") else None,
        )
    if cfg.ensemble == "stable":
        return stable(cfg.n, cfg.m, rng)
    if cfg.ensemble == "binary":
        return binary(cfg.n, cfg.m, rng)
    raise ValueError(f"Unknown ensemble={cfg.ensemble!r}")


def _gen_signal(cfg: ExpConfig, rng: np.random.Generator) -> np.ndarray:
    # Try to “aim” for the requested PE order by choosing period/band richness.
    if cfg.signal == "prbs":
        period = max(31, 2 * int(cfg.pe_order_target) + 1)
        u = prbs(cfg.T, cfg.m, rng, period=period)
    elif cfg.signal == "multisine":
        k_lines = max(4, min(int(cfg.pe_order_target), max(1, cfg.T // 4)))
        u = multisine(cfg.T, cfg.m, rng, k_lines=k_lines)
    else:
        raise ValueError(f"Unknown signal={cfg.signal!r}")
    return u


def _simulate_dt(T: int, x0: np.ndarray, Ad: np.ndarray, Bd: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Simulate x_{k+1} = Ad x_k + Bd u_k for k=0..T-2; return X \in R^{n×T}."""
    n = Ad.shape[0]
    assert u.shape == (T, Bd.shape[1]), "u must be (T×m)"
    X = np.zeros((n, T))
    X[:, 0] = x0
    for k in range(T - 1):
        X[:, k + 1] = Ad @ X[:, k] + Bd @ u[k, :]
    return X


# -------------------------- main runner -------------------------------

def run_single(
    cfg: ExpConfig,
    seed: int,
    sopts: SolverOpts,
    estimators: Sequence[str] = ("dmdc", "moesp"),
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    # System draw + initial condition
    A, B = _select_ensemble(cfg, rng)
    x0 = draw_initial_state(cfg.n, cfg.x0_mode, rng)

    # Input (optionally pointwise-restricted)
    u = _gen_signal(cfg, rng)
    if cfg.U_restr is not None:
        # Project onto span(W)
        u = restrict_pointwise(u, cfg.U_restr)

    # ZOH discretization and simulation
    Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)
    X = _simulate_dt(cfg.T, x0, Ad, Bd, u)
    Xtrain = X[:, :-1]        # n × (T-1)
    Xp     = X[:, 1:]         # n × (T-1)
    Utrain = u[:-1, :].T      # m × (T-1)

    # PE diagnostics on the inputs (achieved, not just target)
    pe_block_hat  = estimate_pe_order_block(u, s_max=min(cfg.pe_order_target, cfg.T // 2))
    pe_moment_hat = estimate_moment_pe_order(u, r_max=min(cfg.pe_order_target, cfg.T), t0=cfg.T - 1, dt=cfg.dt)

    # Choose analysis mode for K(\calU; x0) and V
    # Priority: explicit pointwise → explicit PE_r → unrestricted
    mode = "unrestricted"
    Wmat: Optional[np.ndarray] = None
    r: Optional[int] = None
    if cfg.U_restr is not None:
        mode = "pointwise"
        Wmat = cfg.U_restr
    elif cfg.PE_r is not None:
        mode = "moment-pe"
        r = int(cfg.PE_r)

    # Unified generator and visible subspace
    K = unified_generator(A, B, x0, mode=mode, W=Wmat, r=r)
    K_rank = int(np.linalg.matrix_rank(K, tol=1e-8))
    Vbasis, V_dim = visible_subspace(A, B, x0, mode=mode, W=Wmat, r=r, tol=1e-10)

    # Frobenius-distance proxies to PBH failure
    d_frob_struct = pbh_margin_structured(A, B, x0)       
    d_frob_unstr  = pbh_margin_unstructured(A, K)         

    # Gramian metrics
    gram_min_ct: Optional[float] = None
    gram_min_dt: Optional[float] = None

    # CT infinite-horizon (only if A Hurwitz) with K_core = [x0 B]
    Kcore = np.concatenate([x0.reshape(-1, 1), B], axis=1)
    W_ct = gramian_ct_infinite(A, Kcore)
    if W_ct is not None:
        ev = np.linalg.eigvalsh(W_ct)
        gram_min_ct = float(ev.min())

    # DT finite-horizon fallback (always exists)
    # Disclaimer: This is a *finite-horizon* surrogate
    # for unstable A. We include x0 as a fictitious input by augmenting K with x0.
    K_dt = np.concatenate([x0.reshape(-1, 1), Bd], axis=1)
    W_dt = gramian_dt_finite(Ad, K_dt, T=cfg.T)
    ev_dt = np.linalg.eigvalsh(W_dt)
    gram_min_dt = float(ev_dt.min())

    # Identification algorithms (projected errors on V)
    results_est: Dict[str, Any] = {}

    if "dmdc" in estimators:
        try:
            from .estimators.dmdc import dmdc_fit  # explicit import to avoid circulars
            Ahat, Bhat = dmdc_fit(Xtrain, Xp, Utrain)
            errA, errB = projected_errors(Ahat, Bhat, A, B, Vbasis)
            results_est["dmdc"] = {"A_err_PV": errA, "B_err_PV": errB}
        except Exception as ex:
            results_est["dmdc"] = {"error": f"{type(ex).__name__}: {ex}"}

    if "moesp" in estimators:
        try:
            from .estimators.moesp import moesp_fit
            s_depth = max(2, min(10, cfg.T // 4))
            Ahat, Bhat = moesp_fit(Xtrain, Xp, Utrain, s=s_depth)
            errA, errB = projected_errors(Ahat, Bhat, A, B, Vbasis)
            results_est["moesp"] = {"A_err_PV": errA, "B_err_PV": errB, "s": s_depth}
        except Exception as ex:
            results_est["moesp"] = {"error": f"{type(ex).__name__}: {ex}"}

    # Assemble JSON-able output
    out: Dict[str, Any] = {
        "seed": seed,
        "n": cfg.n,
        "m": cfg.m,
        "T": cfg.T,
        "dt": cfg.dt,
        "ensemble": cfg.ensemble,
        "signal": cfg.signal,
        # input/PE diagnostics
        "pe_order_target": int(cfg.pe_order_target),
        "pe_block_hat": int(pe_block_hat),
        "pe_moment_hat": int(pe_moment_hat),
        # analysis mode and visible component
        "analysis_mode": mode,
        "W_dim": None if Wmat is None else int(Wmat.shape[1]),
        "PE_r": r,
        "K_rank": int(K_rank),
        "V_dim": int(V_dim),
        # PBH margins
        "pbh_struct": float(d_frob_struct),
        "pbh_unstruct": float(d_frob_unstr),
        # Gramians
        "gram_min_ct": gram_min_ct,  # None if A not Hurwitz
        "gram_min_dt": gram_min_dt,
        # Estimator results
        "estimators": results_est,
    }
    return out
