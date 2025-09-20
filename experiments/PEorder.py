from __future__ import annotations
from typing import Iterable, Dict, Any, List, Tuple
import numpy as np

from ..config import ExpConfig, SolverOpts
from ..run_single import run_single
from ..io_utils import save_csv
from ..ensembles import ginibre, sparse_continuous, stable, binary
from ..metrics import unified_generator

def _flatten_result(result: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for k in [
        "seed","n","m","T","dt","ensemble","signal",
        "sigPE","PE_r","analysis_mode",
        "pe_block_hat","pe_moment_hat",
        "K_rank","V_dim","pbh_struct","pbh_unstruct",
        "gram_min_ct","gram_min_dt",
    ]:
        row[k] = result.get(k, None)
    # estimators
    est = result.get("estimators", {})
    for name, payload in est.items():
        if isinstance(payload, dict):
            for kk, vv in payload.items():
                row[f"est.{name}.{kk}"] = vv
        else:
            row[f"est.{name}"] = payload
    return row

# --- Controllability / uncontrollability testers (algebraic)
def _ctrb_rank(A: np.ndarray, B: np.ndarray, tol: float = 1e-10) -> int:
    n = A.shape[0]
    K = B
    M = B.copy()
    for _ in range(1, n):
        M = A @ M
        K = np.concatenate([K, M], axis=1)
    return int(np.linalg.matrix_rank(K, tol=tol))

def _draw_until(ensemble: str, n: int, m: int, pA: float, pB: float, which: str,
                mode: str, rng: np.random.Generator,
                max_tries: int = 200) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    mode \in {'controllable','uncontrollable'}.
    Cap tries and report the selection outcome.
    """
    note = ""
    for t in range(max_tries):
        if ensemble == "ginibre":
            A, B = ginibre(n, m, rng)
        elif ensemble == "stable":
            A, B = stable(n, m, rng)
        elif ensemble == "sparse":
            A, B = sparse_continuous(n, m, rng, which=which, p_density_A=pA, p_density_B=pB)
        else:
            A, B = binary(n, m, rng)

        r = _ctrb_rank(A, B)
        want = (mode == "controllable")
        if (want and r == n) or ((not want) and r < n):
            note = f"{mode} achieved at try {t+1} (ctrb_rank={r})"
            return A, B, note

    # If this fails, return last draw with a warning note.
    note = f"WARNING: could not enforce {mode} within {max_tries} tries (last ctrb_rank={r})."
    return A, B, note

def sweep_pe_order_controllable(
    *,
    n: int = 10,
    m: int = 3,
    T: int = 400,
    dt: float = 0.05,
    ensemble: str = "ginibre",
    p_density: float = 0.8,
    sparse_which: str = "both",
    signal: str = "prbs",
    sigPE: int = 12,
    r_values: Iterable[int] = (1,2,3,4,6,8,10,12,16),
    seeds: Iterable[int] = range(30),
    algs=("dmdc","moesp"),
    out_csv: str = "results_pe_controllable.csv",
) -> None:
    rows: List[Dict[str, Any]] = []
    sopts = SolverOpts()

    for seed in seeds:
        rng = np.random.default_rng(seed)
        # Preselect a controllable pair (shared across r to remove system variance)
        A, B, note = _draw_until(
            ensemble=ensemble, n=n, m=m,
            pA=p_density, pB=p_density, which=sparse_which,
            mode="controllable", rng=rng
        )
        # pack a per-seed base config
        base_cfg = ExpConfig(
            n=n, m=m, T=T, dt=dt,
            ensemble=ensemble,
            p_density=p_density, sparse_which=sparse_which,
            signal=signal, sigPE=sigPE,
            U_restr=None, PE_r=None,
            algs=algs, light=True
        )

        for r in r_values:
            cfg = base_cfg
            cfg.PE_r = int(r)  # analysis on V_PE^r
            # run
            res = run_single(cfg, seed=seed, sopts=sopts, algs=algs)
            row = _flatten_result(res)
            row["tag"] = "pe_controllable"
            row["note_system"] = note
            rows.append(row)

    save_csv(rows, out_csv)

def sweep_pe_order_uncontrollable(
    *,
    n: int = 10,
    m: int = 3,
    T: int = 400,
    dt: float = 0.05,
    ensemble: str = "sparse",
    p_density_A: float = 0.5,
    p_density_B: float = 0.2,
    sparse_which: str = "both",
    signal: str = "prbs",
    sigPE: int = 12,
    r_values: Iterable[int] = (1,2,3,4,6,8,10,12,16),
    seeds: Iterable[int] = range(30),
    algs=("dmdc","moesp"),
    out_csv: str = "results_pe_uncontrollable.csv",
) -> None:
    """
    Bias towards uncontrollability by sparser B (p_density_B) and/or A pattern.
    Check for controllability and resample if needed.
    """
    rows: List[Dict[str, Any]] = []
    sopts = SolverOpts()

    for seed in seeds:
        rng = np.random.default_rng(seed)
        A, B, note = _draw_until(
            ensemble=ensemble, n=n, m=m,
            pA=p_density_A, pB=p_density_B, which=sparse_which,
            mode="uncontrollable", rng=rng
        )
        base_cfg = ExpConfig(
            n=n, m=m, T=T, dt=dt,
            ensemble=ensemble,
            p_density=p_density_A, sparse_which=sparse_which,
            p_density_A=p_density_A, p_density_B=p_density_B,
            signal=signal, sigPE=sigPE,
            U_restr=None, PE_r=None,
            algs=algs, light=True
        )

        for r in r_values:
            cfg = base_cfg
            cfg.PE_r = int(r)
            res = run_single(cfg, seed=seed, sopts=sopts, algs=algs)
            row = _flatten_result(res)
            row["tag"] = "pe_uncontrollable"
            row["note_system"] = note
            rows.append(row)

    save_csv(rows, out_csv)
