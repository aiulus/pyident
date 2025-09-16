import numpy as np
import numpy.linalg as npl
from dataclasses import dataclass
from typing import Tuple, Dict, Any
from scipy import linalg as sla
from config import SolverOpts


def is_zero(v: np.ndarray, tol: float = 1e-14) -> bool:
    return npl.norm(v) <= tol

def build_Q(x0: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    n = x0.size
    if npl.norm(x0) <= tol:
        return np.eye(n, dtype=x0.dtype)
    row = x0.conj().reshape(1, -1)
    Q = sla.null_space(row)
    if Q.shape[1] < n - 1:
        q0 = x0 / npl.norm(x0)
        M = np.eye(n, dtype=x0.dtype) - np.outer(q0, q0.conj())
        Q, _ = npl.qr(M)
        Q = Q[:, : n - 1]
    return Q

def concat_lambda_block(A: np.ndarray, B: np.ndarray, lam: complex) -> np.ndarray:
    n = A.shape[0]
    return np.concatenate((lam * np.eye(n) - A, B), axis=1)

def _phi_and_grad(ab: np.ndarray, Q: np.ndarray, A: np.ndarray, B: np.ndarray):
    n = A.shape[0]
    alpha, beta = float(ab[0]), float(ab[1])
    lam = alpha + 1j * beta
    H = concat_lambda_block(A, B, lam)
    M = Q.conj().T @ H
    U, s, Vh = sla.svd(M, full_matrices=False, lapack_driver='gesvd')
    sigma = s[-1]; u = U[:, -1]; v = Vh.conj().T[:, -1]
    v_first = v[:n]
    c = (Q @ u).conj().T @ v_first
    grad = np.array([np.real(c), -np.imag(c)], dtype=float)
    return float(sigma), grad, u, v

def init_lambda_seeds(A: np.ndarray, opts: SolverOpts) -> np.ndarray:
    rng = np.random.default_rng(opts.random_state)
    eigs = sla.eigvals(A)
    seeds = list(eigs)
    for _ in range(max(0, opts.num_seeds - len(seeds))):
        z = rng.normal(size=A.shape[0]) + 1j * rng.normal(size=A.shape[0])
        z /= npl.norm(z)
        seeds.append(complex(np.vdot(z, A @ z)))
    return np.array(seeds)

def get_d_frob(A: np.ndarray, B: np.ndarray, x0: np.ndarray, opts: SolverOpts|None=None):
    from scipy.optimize import minimize
    if opts is None:
        opts = SolverOpts()
    Q = build_Q(x0)
    best_sigma = np.inf; best_lam = 0.0 + 0.0j
    for lam0 in init_lambda_seeds(A, opts):
        ab0 = np.array([np.real(lam0), np.imag(lam0)], dtype=float)
        res = minimize(lambda ab: _phi_and_grad(ab, Q, A, B)[0],
                       ab0, jac=lambda ab: _phi_and_grad(ab, Q, A, B)[1], method="BFGS",
                       options=dict(maxiter=opts.maxit, gtol=opts.tol_grad))
        sigma = float(res.fun)
        if sigma < best_sigma:
            best_sigma = sigma
            best_lam = complex(res.x[0], res.x[1])
    # witness vector
    M = Q.conj().T @ concat_lambda_block(A, B, best_lam)
    U, s, Vh = sla.svd(M, full_matrices=False, lapack_driver='gesvd')
    u_min = U[:, -1]
    w_star = Q @ u_min
    w_star = np.real_if_close(w_star)
    return float(best_sigma), complex(best_lam), w_star

def gramian_augmented(A: np.ndarray, B: np.ndarray, x0: np.ndarray, t: float = 10.0, N: int = 200):
    n = A.shape[0]
    X = np.concatenate((x0.reshape(-1,1), B), axis=1)
    s_grid = np.linspace(0.0, t, N)
    W = np.zeros((n,n), dtype=A.dtype)
    for k, s in enumerate(s_grid):
        Es = sla.expm(A * s)
        G = Es @ X
        w = 0.5 if (k==0 or k==N-1) else 1.0
        W += w * (G @ G.T.conj())
    W *= t / (N - 1)
    evals = np.real_if_close(npl.eigvalsh((W + W.T.conj())/2.0))
    kappa = float(evals[-1] / max(evals[0], 1e-16))
    return dict(W=W, lambda_min=float(evals[0]), kappa=kappa)

def krylov_metrics(A: np.ndarray, B: np.ndarray, x0: np.ndarray):
    n = A.shape[0]
    cols0 = np.concatenate((x0.reshape(-1,1), B), axis=1)
    K_blocks = []
    Ak = np.eye(n)
    for k in range(n):
        if k == 0:
            K_blocks.append(cols0)
        else:
            Ak = Ak @ A
            K_blocks.append(Ak @ cols0)
    K = np.concatenate(K_blocks, axis=1)
    s = sla.svdvals(K)
    sigma_min = float(s[-1]); rank = int((s > 1e-10).sum())
    return dict(K=K, sigma_min=sigma_min, rank=rank)

def controllable_subspace_basis(A: np.ndarray, B: np.ndarray, tol: float=1e-10):
    n = A.shape[0]
    C = B.copy(); M = C.copy()
    for _ in range(1,n):
        C = A @ C
        M = np.concatenate((M, C), axis=1)
    U, s, _ = sla.svd(M, full_matrices=False)
    r = (s > tol).sum()
    return U[:, :r]

def subspace_angle_metrics(A: np.ndarray, B: np.ndarray, x0: np.ndarray):
    R = controllable_subspace_basis(A,B)
    if R.size == 0:
        return dict(eta0=0.0, theta0=float(np.pi/2), R_basis=R)
    PR = R @ (R.T @ x0)
    eta0 = float(np.linalg.norm(PR) / max(np.linalg.norm(x0), 1e-16))
    eta0 = min(max(eta0, 0.0), 1.0)
    theta0 = float(np.arccos(eta0))
    return dict(eta0=eta0, theta0=theta0, R_basis=R)

def modewise_overlaps(A: np.ndarray, B: np.ndarray, x0: np.ndarray, real_only: bool=True):
    wvals, W = sla.eig(A.conj().T)
    alphas, betas, mus = [], [], []
    for i in range(W.shape[1]):
        w = W[:, i]
        if real_only:
            w = np.real_if_close(w)
        denom = np.sqrt(np.real((w.conj().T @ w)))
        if denom < 1e-14:
            continue
        wt = w.conj().T
        alpha = float(np.abs(wt @ x0) / denom)
        beta  = float(np.linalg.norm(wt @ B) / denom)
        mu    = float(np.linalg.norm(np.concatenate(((wt @ x0).reshape(1,), wt @ B)))) / denom
        alphas.append(alpha); betas.append(beta); mus.append(mu)
    mu_min = float(np.min(mus)) if mus else 0.0
    return dict(alpha=np.array(alphas), beta=np.array(betas), mu_min=mu_min, eigenvalues=wvals)

def bounds_pbh(A: np.ndarray, B: np.ndarray, x0: np.ndarray, opts: SolverOpts|None=None):
    if opts is None:
        opts = SolverOpts()
    Q = build_Q(x0)
    def fH(lam: complex) -> float:
        M = concat_lambda_block(A,B,lam)
        return float(sla.svdvals(M)[-1])
    def fHaug(lam: complex) -> float:
        M = np.concatenate((lam*np.eye(A.shape[0])-A, x0.reshape(-1,1), B), axis=1)
        return float(sla.svdvals(M)[-1])
    # imag-axis proxy
    omega_max = opts.omega_max_factor * np.linalg.norm(A,2)
    omegas = np.linspace(-omega_max, omega_max, opts.grid_omega)
    def fQ(lam: complex) -> float:
        M = Q.conj().T @ concat_lambda_block(A,B,lam)
        return float(sla.svdvals(M)[-1])
    # seeds
    from .metrics import init_lambda_seeds
    seeds = init_lambda_seeds(A, opts)
    upper = min(fH(l) for l in seeds)
    lower = min(fHaug(l) for l in seeds)
    imag  = min(fQ(1j*w) for w in omegas)
    out = dict(lower=lower, upper=upper, imag=imag)
    if lower - upper > 1e-8:
        out["warning"] = "Lower bound exceeded upper; search likely under-resolved."
    return out
