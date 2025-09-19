from __future__ import annotations
from typing import Iterable, Dict, Any, List
import numpy as np

from ..config import ExpConfig, SolverOpts
from ..run_single import run_single
from ..io_utils import save_csv
from ..jax_accel import enable_x64            

def _flatten(prefix: str, d: Dict[str, Any], out: Dict[str, Any]) -> None:
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten(key, v, out)
        else:
            out[key] = v

def _rowify(result: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    keep = [
        "seed","n","m","T","dt","ensemble","signal",
        "sigPE","pe_block_hat","pe_moment_hat",
        "analysis_mode","W_dim","PE_r","K_rank","V_dim",
        "pbh_struct","pbh_unstruct","gram_min_ct","gram_min_dt",
    ]
    for k in keep:
        row[k] = result.get(k, None)

    row["analysis_mode"] = result.get("analysis_mode", result.get("K_mode"))
    row["W_dim"]         = result.get("W_dim",         result.get("K_pointwise_q"))
    row["V_dim"]         = result.get("V_dim",         result.get("K_rank"))
    row["pbh_struct"]    = result.get("pbh_struct",    result.get("delta_pbh"))
    row["gram_min_ct"]   = result.get("gram_min_ct",   result.get("gram_min"))

    env = result.get("env", {})
    row["env.accelerator"] = env.get("accelerator")
    row["env.jax_x64"]     = env.get("jax_x64")

    led = result.get("notes", {}).get("ledger", {})
    tol = led.get("tolerances", {}) if isinstance(led, dict) else {}
    row["tol.svd_rtol"]    = tol.get("svd_rtol")
    row["tol.svd_atol"]    = tol.get("svd_atol")
    row["tol.pbh_cluster"] = tol.get("pbh_cluster_tol")

    # Estimator metrics flattened
    est = result.get("estimators", result.get("algs", {}))
    for name, payload in est.items():
        if isinstance(payload, dict):
            for kk, vv in payload.items():
                row[f"est.{name}.{kk}"] = vv
        else:
            row[f"est.{name}"] = payload
    return row


def sweep_sparsity(
    *,
    n: int = 10,
    m: int = 3,
    p_values: Iterable[float] = (1.0, 0.9, 0.8, 0.6, 0.4, 0.2),
    T: int = 400,
    dt: float = 0.05,
    sparse_which: str = "both",
    signal: str = "prbs",
    sigPE: int = 12,
    seeds: Iterable[int] = range(50),
    algs=("dmdc","moesp","dmdc_tls","dmdc_iv"),
    out_csv: str = "results_sparsity.csv",
    use_jax: bool = False, jax_x64: bool = True,
    x0_mode: str | None = None,
    U_restr_dim: int | None = None,
) -> None:

    if use_jax:
        enable_x64(bool(jax_x64))   # (idempotent; safe if already set)

    rows: List[Dict[str, Any]] = []
    sopts = SolverOpts()
    for p in p_values:
        for seed in seeds:
            # --- build U_restr (canonical q-dim subspace) if requested ---
            if U_restr_dim is not None:
                if U_restr_dim < 1 or U_restr_dim > m:
                    raise ValueError(f"--U_restr_dim must be in [1, m]={m}, got {U_restr_dim}.")
                U_restr = np.eye(m, dtype=float)[:, :U_restr_dim]
            else:
                U_restr = None
            cfg = ExpConfig(
                n=n, m=m, T=T, dt=dt,
                ensemble="sparse",
                sparse_which=sparse_which,   # "A","B","both"
                p_density=(p if sparse_which in ("A","both") else 1.0),
                p_density_B=p if sparse_which in ("B","both") else None,
                x0_mode=(x0_mode or "gaussian"),
                signal=signal,
                sigPE=sigPE,
                U_restr=U_restr,  
                algs=algs,
                light=True,
            )

            res = run_single(cfg, seed=seed, sopts=sopts, algs=algs,
                             use_jax=use_jax, jax_x64=jax_x64)
            row = _rowify(res)
            row["tag"] = "sparsity"
            row["p_density"] = p
            row["sparse_which"] = sparse_which
            rows.append(row)
    save_csv(rows, out_csv)
