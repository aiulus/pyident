from __future__ import annotations
from typing import Iterable, Dict, Any, List
import numpy as np

from ..config import ExpConfig, SolverOpts
from ..run_single import run_single
from ..io_utils import save_csv

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
    
    # Estimator metrics flattened
    est = result.get("estimators", result.get("algs", {}))
    for name, payload in est.items():
        if isinstance(payload, dict):
            for kk, vv in payload.items():
                row[f"est.{name}.{kk}"] = vv
        else:
            row[f"est.{name}"] = payload
    return row

def sweep_underactuation(
    *,
    n: int = 10,
    m_values: Iterable[int] = (1,2,3,4,5,6,8,10),
    T: int = 400,
    dt: float = 0.05,
    ensemble: str = "ginibre",
    signal: str = "prbs",
    sigPE: int = 12,
    seeds: Iterable[int] = range(50),
    algs=("dmdc","moesp","dmdc_tls","dmdc_iv"),
    out_csv: str = "results_underactuation.csv",
    use_jax: bool = False, jax_x64: bool = True,
) -> None:
    if use_jax:
        import jax_accel as jxa
        jxa.enable_x64(bool(jax_x64))
    rows: List[Dict[str, Any]] = []
    sopts = SolverOpts()
    for m in m_values:
        for seed in seeds:
            cfg = ExpConfig(
                n=n, m=m, T=T, dt=dt,
                ensemble=ensemble,
                signal=signal,
                sigPE=sigPE,
                U_restr=None, PE_r=None,
                algs=algs,
                light=True,
            )
            res = run_single(cfg, seed=seed, sopts=sopts, algs=algs)
            row = _rowify(res)
            row["tag"] = "underactuation"
            rows.append(row)
    save_csv(rows, out_csv)
