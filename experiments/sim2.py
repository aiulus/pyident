#!/usr/bin/env python3
"""
ROC from *estimation errors* (prediction-data–driven labels).

Pipeline (mirrors your earlier experiment, now for ROC):
  1) Fix (A,B) with controllability rank n-1 (barely uncontrollable).
  2) For each trial: draw x0 ~ Unif(S^{n-1}); simulate DT with PRBS; optionally add noise.
  3) Estimate (A,B) via DMDc (pinv) and MOESP (full-state wrapper).
  4) Compute "scores" from TRUE (A,B,x0) *before* estimation (predictors):
        s_pbh  = 1 / PBH_structured_margin(A,B,x0)         (↑ = more likely UNidentifiable)
        s_kry  = σ_min(K_n(A,B,x0))                         (↑ = more likely UNidentifiable)  [per your spec]
        s_mu   = 1 / μ_min(A,[x0 B])                        (↑ = more likely UNidentifiable)
  5) Compute error labels from estimation outcomes (ground truth from data):
        errA_* = ||Â - A_d||_F / ||A_d||_F
        errB_* = ||\hat B - B_d||_F / ||B_d||_F
        err_mean_* = 0.5*(errA_* + errB_*)     # default target for ROC
     Then set y=1 (UNidentifiable) if err_target > τ  (τ can be absolute or quantile-based).
  6) ROC: vary the threshold over score s and compute TPR/FPR vs the fixed y labels.
     AUC via trapezoidal rule. Save CSVs and PNGs.

CLI options let you choose algorithm (dmdc|moesp|either), the error target (A,B,mean),
and the label thresholding mode (absolute τ or quantile q).
"""
from __future__ import annotations
import argparse, math, sys, pathlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from ..metrics import(
    pbh_margin_structured,
    unified_generator,
    left_eigvec_overlap,
    cont2discrete_zoh,
)

from ..estimators import(
    dmdc_pinv,
    moesp_fit,
    sindy_fit,
    node_fit
)

from ..ensembles import draw_with_ctrb_rank

def unit(v, eps=0.0):
    n = np.linalg.norm(v)
    return v / (n if n>eps else 1.0)

def prbs(T, m, rng, dwell=1):
    steps = math.ceil(T/dwell)
    seq = rng.choice([-1.0, 1.0], size=(steps, m))
    return np.repeat(seq, dwell, axis=0)[:T,:]

def simulate_dt(Ad, Bd, U, x0):
    n, T = Ad.shape[0], U.shape[0]
    X = np.empty((n, T+1)); X[:,0]=x0
    for k in range(T):
        X[:,k+1] = Ad@X[:,k] + Bd@U[k,:]
    return X

def rel_fro(Mhat, Mtrue):
    den = np.linalg.norm(Mtrue, 'fro')
    return float(np.linalg.norm(Mhat-Mtrue,'fro')/(den if den>0 else 1.0))

def build_scores(A,B,x0, eps=1e-12):
    # 1) PBH margin (structured) inverted
    pbh = float(pbh_margin_structured(A,B,x0))
    s_pbh = 1.0/max(pbh, eps)
    # 2) Krylov σmin
    K = unified_generator(A,B,x0, mode='unrestricted')
    svals = np.linalg.svd(K, compute_uv=False)
    krylov_smin = float(svals.min()) if svals.size else 0.0
    s_kry = 1.0 / max(krylov_smin, eps)
    # 3) Left-eig inverse μ_min
    Xaug = np.concatenate([x0.reshape(-1,1), B], axis=1)
    mu = left_eigvec_overlap(A, Xaug)
    mu_min = float(np.min(mu)) if mu.size else 0.0
    s_mu = 1.0/max(mu_min, eps)
    return s_pbh, s_kry, s_mu

def build_labels(df, algo='either', target='mean', mode='q', thr=0.7):
    """
    df has per-trial columns: errA_dmdc, errB_dmdc, errA_moesp, errB_moesp.
    'algo': 'dmdc'|'moesp'|'either' (either means max over algos)
    'target': 'A'|'B'|'mean' which error to threshold
    'mode': 'abs' for absolute threshold, or 'q' for quantile (e.g., 0.7 means top 30% → 1)
    Returns y∈{0,1} where 1 = UNidentifiable (error above threshold).
    """
    if algo == 'dmdc':
        eA = df['errA_dmdc'].to_numpy(); eB = df['errB_dmdc'].to_numpy()
    elif algo == 'moesp':
        eA = df['errA_moesp'].to_numpy(); eB = df['errB_moesp'].to_numpy()
    else:
        # max across algorithms → pessimistic "either fails"
        eA = np.maximum(df['errA_dmdc'].to_numpy(), df['errA_moesp'].to_numpy())
        eB = np.maximum(df['errB_dmdc'].to_numpy(), df['errB_moesp'].to_numpy())
    if target == 'A':
        e = eA
    elif target == 'B':
        e = eB
    else:
        e = 0.5*(eA+eB)
    if mode == 'abs':
        tau = float(thr)
    else:
        tau = float(np.quantile(e, thr))
    y = (e > tau).astype(int)
    return y, tau

def roc_curve(scores, labels, roc_step=0.2):
    min_thr = scores.min()
    max_thr = scores.max()
    thresholds = np.arange(min_thr, max_thr + roc_step, roc_step)
    tpr = []
    fpr = []
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        tp = np.sum((y_pred == 1) & (labels == 1))
        fp = np.sum((y_pred == 1) & (labels == 0))
        fn = np.sum((y_pred == 0) & (labels == 1))
        tn = np.sum((y_pred == 0) & (labels == 0))
        tpr.append(tp / max(tp + fn, 1))
        fpr.append(fp / max(fp + tn, 1))
    return np.array(fpr), np.array(tpr), thresholds


def auc_trapz(fpr,tpr):
    return float(np.trapz(tpr,fpr))

def run(n=6, m=2, T=200, dt=0.05, trials=400, noise_std=0.0, seed=123,
        algo='either', target='mean', label_mode='q', label_thr=0.7,
        outdir='roc_from_estimation', roc_step=0.2):
    rng = np.random.default_rng(seed)
    outdir = pathlib.Path(outdir); (outdir/'plots').mkdir(parents=True, exist_ok=True)

    # Fix (A,B) with controllability rank n-1
    A, B, meta = draw_with_ctrb_rank(n, m, r=n-1, rng=rng, base_c='ginibre', base_u='ginibre', embed_random_basis=True)
    Ad, Bd = cont2discrete_zoh(A,B,dt)

    rows = []
    for t in range(trials):
        x0 = unit(rng.standard_normal(n))
        U = prbs(T, m, rng, dwell=1)
        X = simulate_dt(Ad, Bd, U, x0)
        if noise_std>0:
            X = X + noise_std*rng.standard_normal(X.shape)
        X0, X1, Utr = X[:,:-1], X[:,1:], U.T

        # estimates
        A_d, B_d = dmdc_pinv(X0, X1, Utr)
        A_m, B_m = moesp_fit(X0, X1, Utr, n=n)

        # errors vs DT true
        errA_d = rel_fro(A_d, Ad)
        errB_d = rel_fro(B_d, Bd)
        errA_m = rel_fro(A_m, Ad)
        errB_m = rel_fro(B_m, Bd)

        # scores from TRUE system
        s_pbh, s_kry, s_mu = build_scores(A,B,x0)

        rows.append(dict(trial=t,
                         s_pbh=s_pbh, s_kry=s_kry, s_mu=s_mu,
                         errA_dmdc=errA_d, errB_dmdc=errB_d,
                         errA_moesp=errA_m, errB_moesp=errB_m))

    df = pd.DataFrame(rows)
    df.to_csv(outdir/'results.csv', index=False)

    # labels from estimation errors
    y, tau = build_labels(df, algo=algo, target=target, mode=label_mode, thr=label_thr)

    # ROC per metric
    auc_tbl = []
    for key, nice in [('s_pbh','1/PBH'), ('s_kry','1/σ_min(K_n)'), ('s_mu','1/μ_min')]:
        fpr, tpr, thr = roc_curve(df[key].to_numpy(), y, roc_step=roc_step)
        auc = auc_trapz(fpr,tpr)
        auc_tbl.append({'metric': nice, 'AUC': auc, 'label_tau': tau})
        pd.DataFrame({'fpr':fpr,'tpr':tpr,'threshold':thr}).to_csv(outdir/f'roc_points_{key}.csv', index=False)

        # plot
        plt.figure(figsize=(6.0,5.2))
        plt.plot(fpr,tpr)
        plt.plot([0,1],[0,1],'--',linewidth=1)
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(f'ROC — {nice} (AUC={auc:.3f})\nlabel: {algo}/{target}, {label_mode}={label_thr:.3f}, τ={tau:.3g}')
        plt.tight_layout()
        plt.savefig(outdir/f'plots/roc_{key}.png', dpi=150); plt.close()

    # combined plot
    plt.figure(figsize=(6.4,5.2))
    for key, nice in [('s_pbh','1/PBH'), ('s_kry','σ_min(K_n)'), ('s_mu','1/μ_min')]:
        pts = pd.read_csv(outdir/f'roc_points_{key}.csv')
        plt.plot(pts['fpr'], pts['tpr'], label=nice)
    plt.plot([0,1],[0,1],'--',linewidth=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC — label: {algo}/{target}, {label_mode}={label_thr:.3f} (τ={tau:.3g})')
    plt.legend(loc='lower right'); plt.tight_layout()
    plt.savefig(outdir/'plots/roc_all.png', dpi=150); plt.close()

    pd.DataFrame(auc_tbl).to_csv(outdir/'roc_summary.csv', index=False)
    # also save system for provenance
    np.savez(outdir/'system.npz', A=A, B=B, Ad=Ad, Bd=Bd)

    return outdir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--m', type=int, default=2)
    ap.add_argument('--T', type=int, default=200)
    ap.add_argument('--dt', type=float, default=0.05)
    ap.add_argument('--trials', type=int, default=400)
    ap.add_argument('--noise-std', type=float, default=0.00)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--algo', type=str, default='either', choices=['dmdc','moesp','either'])
    ap.add_argument('--target', type=str, default='mean', choices=['A','B','mean'])
    ap.add_argument('--label-mode', type=str, default='q', choices=['abs','q'])
    ap.add_argument('--label-thr', type=float, default=0.70, help='abs: threshold value; q: quantile in [0,1]')
    ap.add_argument('--outdir', type=str, default='/mnt/data/roc_from_estimation_demo')
    ap.add_argument('--roc-step', type=float, default=0.2, help='Step size for ROC threshold sweep')
    args = ap.parse_args()
    out = run(n=args.n, m=args.m, T=args.T, dt=args.dt, trials=args.trials,
              noise_std=args.noise_std, seed=args.seed,
              algo=args.algo, target=args.target,
              label_mode=args.label_mode, label_thr=args.label_thr,
              outdir=args.outdir, roc_step=args.roc_step)
    print("Saved artifacts under:", out)

if __name__ == '__main__':
    main()
