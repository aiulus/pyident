
from __future__ import annotations
import numpy as np
from typing import Sequence
from scipy import linalg as sla
from config import SolverOpts, ExpConfig
from ensembles import sample_instance
from metrics import (delta_fix_lambda, krylov_metrics, subspace_angle_metrics,
                      modewise_overlaps, controllable_subspace_basis, gramian_augmented, bounds_pbh)
from sys_utils import c2d


def run_single(cfg: ExpConfig, seed: int, sopts: SolverOpts, estimators: Sequence[str] = ()):
    rng = np.random.default_rng(seed)
    A,B,x0,meta = sample_instance(cfg, rng)
    d_fix, lam_star, w_star = delta_fix_lambda(A,B,x0,sopts)
    bounds = bounds_pbh(A,B,x0,sopts)
    gram   = gramian_augmented(A,B,x0, t=cfg.horizon_t, N=int(max(4, cfg.horizon_t/cfg.dt)))
    kry    = krylov_metrics(A,B,x0)
    ang    = subspace_angle_metrics(A,B,x0)
    mode   = modewise_overlaps(A,B,x0)
    # optional estimators
    est_out = {}
    if estimators:
        T = int(cfg.horizon_t / cfg.dt)
        # simulate discrete-time trajectory
        Ad, Bd = c2d(A,B,cfg.dt)
        from .signals import prbs
        if cfg.u_type == "prbs":
            U = prbs(B.shape[1], T, rng=rng)
        else:
            U = rng.normal(size=(B.shape[1], T))
        X = np.zeros((A.shape[0], T+1)); X[:,0]=x0
        for k in range(T):
            X[:,k+1] = Ad @ X[:,k] + Bd @ U[:,k]
        # DMDC
        if "dmdc" in estimators:
            from .estimators.dmdc import DMDC
            est = DMDC().fit(X,U)
            est_out["dmdc"] = est.metrics(Ad,Bd)
        if "moesp" in estimators:
            from .estimators.moesp import MOESP
            est = MOESP(i=2*A.shape[0], s=2*A.shape[0]).fit(X,U)
            # compute metrics against Ad,Bd
            import numpy.linalg as npl
            eA = npl.norm(est.A_hat-Ad,'fro'); eB = npl.norm(est.B_hat-Bd,'fro')
            est_out["moesp"] = dict(errA_fro=float(eA), errB_fro=float(eB), err_joint=float((eA*eA+eB*eB)**0.5))

    return dict(
        meta=dict(seed=seed, n=A.shape[0], m=B.shape[1], ensemble=meta.get("ensemble"), density=meta.get("density")),
        A=A, B=B, x0=x0,
        delta_fix=d_fix, lambda_star=lam_star, w_star=w_star,
        bounds=bounds, gramian=gram, krylov=kry, angles=ang, modewise=mode,
        estimators=est_out, warnings=[]
    )
