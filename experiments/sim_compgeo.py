from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple
import pathlib

import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..ensembles import draw_with_ctrb_rank
from ..estimators import dmdc_tls, moesp_fit
from ..metrics import (
    cont2discrete_zoh,
    data_equivalence_residual,
    regressor_stats,
    regressor_stats_on_V,
    visible_basis_dt,
)
from ..projectors import principal_angles, projector_from_basis, projector_gap, visible_from_traj
from ..signals import estimate_pe_order_block, multisine
from ..simulation import prbs as prbs_dt
from ..simulation import simulate_dt
from .visible_sampling import (
    construct_x0_with_dim_visible,
    reachable_basis,
    sample_visible_initial_state,
)


@dataclass
class CompGeoConfig(ExperimentConfig):
    """
    Geometry-aware vs Fisher-information adaptive comparison.
    """

    k_values: Tuple[int, ...] = (2, 3)
    T_max: int = 60
    L: int = 5

    u_max: float = 1.0
    E_max: float = 10.0
    candidate_bank_size: int = 12
    dwell_options: Tuple[int, ...] = (1, 2, 4)
    multisine_k_lines: int = 6

    noise_std_proc: float = 0.0
    noise_std_meas: float = 0.0

    n_systems: int = 3
    n_x0_per_system: int = 2

    policies: Tuple[str, ...] = ("GA", "OED")
    oed_criteria: Tuple[str, ...] = ("D", "E")
    estimator: str = "MOESP"  # or "DMDc"

    rtol_rank: float = 1e-12
    fisher_ridge: float = 1e-6

    stop_on_v_sat: bool = False
    v_sat_eps_proj: float = 1e-2
    v_sat_min_windows: int = 2

    save_dir: pathlib.Path = field(default_factory=lambda: pathlib.Path("out_compgeo"))

    def __post_init__(self) -> None:
        super().__post_init__()
        self.save_dir.mkdir(parents=True, exist_ok=True)


def _scale_to_budget(U: np.ndarray, u_max: float, e_max: float) -> Tuple[np.ndarray, float]:
    """Clip to amplitude and rescale for energy, returning (U_scaled, energy)."""
    U = np.asarray(U, dtype=float)
    U = np.clip(U, -u_max, u_max)
    energy = float(np.sum(U ** 2))
    if energy > e_max and energy > 0:
        U = U * np.sqrt(e_max / energy)
        energy = float(np.sum(U ** 2))
    return U, energy


def _candidate_bank(cfg: CompGeoConfig, rng: np.random.Generator, L: int) -> List[Dict]:
    """Generate a shared candidate bank of admissible windows."""
    out: List[Dict] = []
    # Half PRBS, half multisine (rounded)
    num_prbs = max(1, cfg.candidate_bank_size // 2)
    num_ms = cfg.candidate_bank_size - num_prbs

    for _ in range(num_prbs):
        dwell = int(rng.choice(cfg.dwell_options))
        U = prbs_dt(L, cfg.m, scale=cfg.u_max, dwell=dwell, rng=rng)
        U, energy = _scale_to_budget(U, cfg.u_max, cfg.E_max)
        pe_est = int(estimate_pe_order_block(U, s_max=min(L // 2 + 1, 8)))
        out.append(
            {
                "U": U,
                "energy": energy,
                "kind": "prbs",
                "dwell": dwell,
                "k_lines": 0,
                "pe_est": pe_est,
            }
        )

    for _ in range(num_ms):
        U = multisine(L, cfg.m, rng=rng, k_lines=cfg.multisine_k_lines)
        U, energy = _scale_to_budget(U, cfg.u_max, cfg.E_max)
        pe_est = int(estimate_pe_order_block(U, s_max=min(L // 2 + 1, 8)))
        out.append(
            {
                "U": U,
                "energy": energy,
                "kind": "multisine",
                "dwell": 0,
                "k_lines": cfg.multisine_k_lines,
                "pe_est": pe_est,
            }
        )

    rng.shuffle(out)
    return out


def _build_bank_sequence(cfg: CompGeoConfig, rng: np.random.Generator) -> List[Dict]:
    """
    Precompute a candidate bank for each time window so policies share the same menu.
    Each element has keys {"L": L_eff, "bank": [...] }.
    """
    seq: List[Dict] = []
    steps = 0
    while steps < cfg.T_max:
        L_eff = min(cfg.L, cfg.T_max - steps)
        seq.append({"L": L_eff, "bank": _candidate_bank(cfg, rng, L_eff)})
        steps += L_eff
    return seq


def _fit_model(X_hist: np.ndarray, U_hist: np.ndarray, estimator: str, dt: float, fallback: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Refit (A,B) from data; fall back to provided pair on failure."""
    if X_hist.shape[1] < 2 or U_hist.shape[0] == 0:
        return fallback
    X0 = X_hist[:, :-1]
    X1 = X_hist[:, 1:]
    U_cm = U_hist.T
    n = X_hist.shape[0]
    try:
        if estimator.lower() == "moesp":
            u_ts = U_cm.T
            x_ts = X0.T
            s = max(5, min(n + 1, X0.shape[1] // 2))
            Ahat, Bhat, _, _, _ = moesp_fit(u_ts, x_ts, s=s, n=n, return_states=False)
        else:
            Ahat, Bhat = dmdc_tls(X0, X1, U_cm)
        return Ahat, Bhat
    except Exception:
        return fallback


def _fisher_info(X0: np.ndarray, U_cm: np.ndarray) -> np.ndarray:
    """Fisher-like info using regressor Z=[X0; U_cm] (channel-major inputs)."""
    Z = np.vstack([X0, U_cm])
    return Z @ Z.T


def _rank_with_tol(M: np.ndarray, rtol: float) -> int:
    if M.size == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    if s.size == 0:
        return 0
    return int(np.sum(s > rtol * s[0]))


def _score_oed(F_base: np.ndarray, Delta: np.ndarray, criterion: str, ridge: float) -> float:
    d = F_base.shape[0]
    F = F_base + Delta + ridge * np.eye(d)
    if criterion == "D":
        sign, logdet = np.linalg.slogdet(F)
        return float(logdet if sign > 0 else -np.inf)
    if criterion == "E":
        lam_min = np.linalg.eigvalsh(F).min()
        return float(lam_min)
    if criterion == "A":
        try:
            tr_inv = float(np.trace(np.linalg.inv(F)))
        except np.linalg.LinAlgError:
            return -np.inf
        return -tr_inv
    return -np.inf


def _window_stats(
    PV: np.ndarray,
    X_hist: np.ndarray,
    U_hist: np.ndarray,
    rtol_rank: float,
) -> Tuple[Dict, Dict]:
    """Compute regressor stats global and on V."""
    if U_hist.shape[0] == 0:
        empty = {"rank": 0, "cond": np.inf, "smin": 0.0}
        return empty, empty
    X0 = X_hist[:, :-1]
    U_cm = U_hist.T
    return regressor_stats(X0, U_cm, rtol_rank=rtol_rank), regressor_stats_on_V(PV, X0, U_cm, rtol=rtol_rank)


def _run_single_policy(
    cfg: CompGeoConfig,
    Ad: np.ndarray,
    Bd: np.ndarray,
    x0: np.ndarray,
    PV: np.ndarray,
    policy: str,
    criterion: str,
    rng: np.random.Generator,
    bank_sequence: List[Dict],
) -> List[Dict]:
    """Run one adaptive trajectory for a fixed policy/criterion."""
    n, m = Bd.shape
    X_hist = [np.asarray(x0, dtype=float)]
    U_hist_rows: List[np.ndarray] = []

    Ahat, Bhat = Ad.copy(), Bd.copy()  # cold-start with oracle for fairness
    hitting_time: int | None = None

    V_prev: np.ndarray | None = None
    k_prev: int = 0
    sat_counter: int = 0
    v_saturated: bool = False

    records: List[Dict] = []
    steps = 0
    for win in bank_sequence:
        L_eff = int(win["L"])
        bank = win["bank"]
        # Refresh model from accumulated data
        if U_hist_rows:
            U_arr = np.vstack(U_hist_rows)
            X_arr = np.column_stack(X_hist)
            Ahat, Bhat = _fit_model(X_arr, U_arr, cfg.estimator, cfg.dt, (Ahat, Bhat))

        # Prepare GA/OED ingredients
        X_arr = np.column_stack(X_hist)
        V_hat = visible_from_traj(X_arr, tol=cfg.rtol_rank)
        P_perp = np.eye(n) - projector_from_basis(V_hat)

        if U_hist_rows:
            U_arr = np.vstack(U_hist_rows)
            X0_data = X_arr[:, :-1]
            U_cm = U_arr.T
            F_base = _fisher_info(X0_data, U_cm)
        else:
            U_arr = np.zeros((0, m))
            X0_data = np.zeros((n, 0))
            U_cm = np.zeros((m, 0))
            F_base = np.zeros((n + m, n + m))

        scores: List[float] = []
        for cand in bank:
            U_win = cand["U"]
            X_pred = simulate_dt(X_hist[-1], Ahat, Bhat, U_win, noise_std=0.0, rng=rng)
            X_pred_seg = X_pred[:, 1:]  # exclude starting state
            if policy == "GA":
                score = float(np.linalg.norm(P_perp @ X_pred_seg, "fro") ** 2)
            else:
                Xf0 = X_pred_seg[:, :-1] if X_pred_seg.shape[1] > 0 else np.zeros((n, 0))
                Uf = U_win[:-1, :].T if U_win.shape[0] > 1 else np.zeros((m, 0))
                Delta = _fisher_info(Xf0, Uf)
                score = _score_oed(F_base, Delta, criterion, cfg.fisher_ridge)
            scores.append(score)

        best_idx = int(np.argmax(scores)) if scores else 0
        chosen = bank[best_idx]
        U_exec = chosen["U"]

        # Execute on true system
        X_seg = simulate_dt(X_hist[-1], Ad, Bd, U_exec, noise_std=cfg.noise_std_proc, rng=rng)
        if cfg.noise_std_meas > 0.0:
            meas_noise = rng.normal(0.0, cfg.noise_std_meas, size=X_seg[:, 1:].shape)
            X_seg[:, 1:] += meas_noise

        # Append trajectories
        for j in range(1, X_seg.shape[1]):
            X_hist.append(X_seg[:, j])
        U_hist_rows.append(U_exec)
        steps += U_exec.shape[0]

        X_arr = np.column_stack(X_hist)
        V_hat_post = visible_from_traj(X_arr, tol=cfg.rtol_rank)
        r_emp = _rank_with_tol(X_arr, cfg.rtol_rank)
        if hitting_time is None and r_emp >= PV.shape[1]:
            hitting_time = steps

        reg_all, reg_on_v = _window_stats(
            PV, X_arr, np.vstack(U_hist_rows), cfg.rtol_rank
        )
        gap = projector_gap(V_hat_post, PV)
        angs = principal_angles(V_hat_post, PV)
        max_ang = float(np.rad2deg(angs.max())) if angs.size else 0.0

        k_emp = V_hat_post.shape[1]
        if V_prev is not None:
            delta_proj_emp = float(
                np.linalg.norm(
                    projector_from_basis(V_hat_post) - projector_from_basis(V_prev),
                    ord="fro",
                )
            )
        else:
            delta_proj_emp = np.nan

        if (
            V_prev is not None
            and k_emp == k_prev
            and not np.isnan(delta_proj_emp)
            and delta_proj_emp <= cfg.v_sat_eps_proj
        ):
            sat_counter += 1
        else:
            sat_counter = 0

        V_prev = V_hat_post
        k_prev = k_emp
        v_saturated = sat_counter >= cfg.v_sat_min_windows

        try:
            fit_ok, fit_info = data_equivalence_residual(
                X_arr[:, :-1], X_arr[:, 1:], np.vstack(U_hist_rows).T, Ahat, Bhat, rtol=1e-10
            )
            fit_resid = fit_info.get("resid_rel", np.nan)
        except Exception:
            fit_resid = np.nan

        records.append(
            {
                "policy": policy,
                "criterion": criterion if policy == "OED" else "-",
                "t": steps,
                "T_star": hitting_time if hitting_time is not None else -1,
                "rank_emp": r_emp,
                "proj_gap": gap,
                "theta_max_deg": max_ang,
                "smin_on_V": reg_on_v.get("smin", np.nan),
                "rank_on_V": reg_on_v.get("rank", np.nan),
                "cond_on_V": reg_on_v.get("cond", np.nan),
                "smin_global": reg_all.get("smin", np.nan),
                "rank_global": reg_all.get("rank", np.nan),
                "cond_global": reg_all.get("cond", np.nan),
                "energy": chosen["energy"],
                "pe_est": chosen["pe_est"],
                "dwell": chosen["dwell"],
                "k_lines": chosen["k_lines"],
                "fit_resid": fit_resid,
                "k_emp": k_emp,
                "delta_proj_emp": delta_proj_emp,
                "sat_counter": sat_counter,
                "v_saturated": v_saturated,
            }
        )

        if steps >= cfg.T_max:
            break

        if cfg.stop_on_v_sat and v_saturated:
            break

    return records


def run_compgeo(cfg: CompGeoConfig) -> pd.DataFrame:
    """
    Public entry: run comparisons and return tidy DataFrame.
    """
    rng = np.random.default_rng(cfg.seed)
    all_records: List[Dict] = []

    for k in cfg.k_values:
        for sys_idx in range(cfg.n_systems):
            sys_rng = np.random.default_rng(rng.integers(1_000_000_000))
            A, B, meta = draw_with_ctrb_rank(n=cfg.n, m=cfg.m, r=k, rng=sys_rng, ensemble_type="ginibre")
            Ad, Bd = cont2discrete_zoh(A, B, cfg.dt)
            Rbasis = meta.get("R_basis", reachable_basis(Ad, Bd, tol=cfg.rtol_rank))

            for x_idx in range(cfg.n_x0_per_system):
                x_rng = np.random.default_rng(sys_rng.integers(1_000_000_000))
                try:
                    x0, _ = construct_x0_with_dim_visible(Ad, Bd, Rbasis, k, rng=x_rng, tol=cfg.rtol_rank)
                except Exception:
                    x0, _ = sample_visible_initial_state(Ad, Bd, Rbasis, k, rng=x_rng, tol=cfg.rtol_rank)

                PV = visible_basis_dt(Ad, Bd, x0, tol_rank=cfg.rtol_rank)
                bank_seq = _build_bank_sequence(cfg, np.random.default_rng(x_rng.integers(1_000_000_000)))
                for policy in cfg.policies:
                    crits: Sequence[str] = cfg.oed_criteria if policy == "OED" else ("-",)
                    for crit in crits:
                        run_rng = np.random.default_rng(x_rng.integers(1_000_000_000))
                        recs = _run_single_policy(cfg, Ad, Bd, x0, PV, policy, crit, run_rng, bank_seq)
                        for r in recs:
                            r.update(
                                {
                                    "n": cfg.n,
                                    "m": cfg.m,
                                    "k": k,
                                    "sys_idx": sys_idx,
                                    "x0_idx": x_idx,
                                    "dt": cfg.dt,
                                    "L": cfg.L,
                                    "u_max": cfg.u_max,
                                    "E_max": cfg.E_max,
                                    "noise_proc": cfg.noise_std_proc,
                                    "noise_meas": cfg.noise_std_meas,
                                    "estimator": cfg.estimator,
                                    "seed": cfg.seed,
                                }
                            )
                        all_records.extend(recs)

    df = pd.DataFrame.from_records(all_records)
    csv_path = cfg.save_dir / "sim_compgeo.csv"
    df.to_csv(csv_path, index=False)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Geometry-aware vs OED adaptive comparison.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out_compgeo"))
    parser.add_argument("--n_systems", type=int, default=3)
    parser.add_argument("--n_x0_per_system", type=int, default=2)
    parser.add_argument("--n", type=int, default=8, help="state dimension")
    parser.add_argument("--m", type=int, default=2, help="input dimension")
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[2, 3],
        help="visible subspace dimensions to target",
    )
    parser.add_argument("--T_max", type=int, default=60)
    parser.add_argument("--L", type=int, default=5)
    parser.add_argument("--u_max", type=float, default=1.0)
    parser.add_argument("--E_max", type=float, default=10.0)
    parser.add_argument("--policies", type=str, nargs="+", default=["GA", "OED"])
    parser.add_argument("--oed_criteria", type=str, nargs="+", default=["D", "E"])
    parser.add_argument("--estimator", type=str, default="MOESP")
    parser.add_argument(
        "--stop-on-v-sat",
        action="store_true",
        help="Stop a trajectory once the empirical visible subspace saturates.",
    )
    parser.add_argument(
        "--v-sat-eps-proj",
        type=float,
        default=1e-2,
        help="Projector gap tolerance for saturation (Frobenius norm).",
    )
    parser.add_argument(
        "--v-sat-min-windows",
        type=int,
        default=2,
        help="Number of consecutive windows required to declare saturation.",
    )
    args = parser.parse_args()

    cfg = CompGeoConfig(
        seed=args.seed,
        save_dir=args.out,
        n=args.n,
        m=args.m,
        k_values=tuple(args.k_values),
        n_systems=args.n_systems,
        n_x0_per_system=args.n_x0_per_system,
        T_max=args.T_max,
        L=args.L,
        u_max=args.u_max,
        E_max=args.E_max,
        policies=tuple(args.policies),
        oed_criteria=tuple(args.oed_criteria),
        estimator=args.estimator,
        stop_on_v_sat=args.stop_on_v_sat,
        v_sat_eps_proj=args.v_sat_eps_proj,
        v_sat_min_windows=args.v_sat_min_windows,
    )
    run_compgeo(cfg)
