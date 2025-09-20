from __future__ import annotations
from typing import Iterable, Dict, Any, List
import numpy as np

from ..config import ExpConfig, SolverOpts
from ..run_single import run_single
from ..io_utils import save_csv
from ..jax_accel import enable_x64            
from ..metrics import eta0, left_eig_overlaps, ctrl_growth_metrics
from .analysis import est_success, summarize_array, wilson_ci 

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


def sweep_sparsity_plus(n: int, m: int, p_values, T: int, dt: float, *,
                        trials: int = 100, signal: str = "prbs", sigPE: int = 31,
                        sparse_which: str = "both", seed: int = 0,
                        use_jax: bool = False, algs=("dmdc","moesp")) -> dict:
    """
    Extends baseline sparsity sweep with complementary metrics and success CIs.
    """
    from ..config import ExpConfig, SolverOpts
    from ..run_single import run_single
    rng = np.random.default_rng(seed)
    out = {"n": n, "m": m, "T": T, "dt": dt, "p_values": list(p_values), "trials": trials, "rows": []}

    for p in p_values:
        rows = []
        succ_dmdc = []; succ_moesp = []
        for t in range(trials):
            cfg = ExpConfig(n=n, m=m, T=T, dt=dt, ensemble="sparse", p_density=p,
                            sparse_which=sparse_which, signal=signal, sigPE=sigPE)
            rs = run_single(cfg, seed=int(rng.integers(0, 2**31-1)),
                            sopts=SolverOpts(), algs=algs, use_jax=use_jax)
            rows.append({"K_rank": rs["K_rank"], "delta_pbh": rs["delta_pbh"]})
            succ_dmdc.append(est_success(rs["estimators"].get("dmdc", {})))
            succ_moesp.append(est_success(rs["estimators"].get("moesp", {})))
            try:
                A = rs["notes"]["ledger"]["A"]; B = rs["notes"]["ledger"]["B"]; x0 = rs["notes"]["ledger"]["x0"]
                _, alpha, beta = left_eig_overlaps(A, x0, B)
                nu_max, nu_gap = ctrl_growth_metrics(A, B)
                rows[-1].update({
                    "alpha_med": float(np.median(alpha)),
                    "beta_med": float(np.median(beta)),
                    "eta0": float(eta0(A, B, x0)),
                    "nu_max": int(nu_max), "nu_gap": int(nu_gap),
                })
            except Exception:
                pass

        p_d = float(np.mean(succ_dmdc)); p_m = float(np.mean(succ_moesp))
        out["rows"].append({
            "p": float(p),
            "K_rank": summarize_array(r["K_rank"] for r in rows),
            "delta_pbh": summarize_array(r["delta_pbh"] for r in rows),
            "success": {
                "dmdc": {"p": p_d, "ci": wilson_ci(p_d, trials)},
                "moesp": {"p": p_m, "ci": wilson_ci(p_m, trials)},
            },
            "extras": {
                "eta0": summarize_array(r.get("eta0", np.nan) for r in rows),
                "alpha_med": summarize_array(r.get("alpha_med", np.nan) for r in rows),
                "beta_med": summarize_array(r.get("beta_med", np.nan) for r in rows),
                "nu_max": summarize_array(r.get("nu_max", np.nan) for r in rows),
                "nu_gap": summarize_array(r.get("nu_gap", np.nan) for r in rows),
            }
        })
    return out
