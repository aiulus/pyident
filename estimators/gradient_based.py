# Gradient-based DMDc (batch) with ridge, Adam/SGD, and optional stability projection.
# Works in NumPy; optionally uses JAX when available.

from __future__ import annotations
import math
from typing import Dict, Tuple, Optional

import numpy as np

# --- Optional JAX support (falls back to NumPy gracefully) ---
try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except Exception:
    jax = None
    jnp = None
    _JAX_AVAILABLE = False


# ----------------------------
# Utilities
# ----------------------------
def _spectral_radius(M: np.ndarray) -> float:
    try:
        return float(np.max(np.abs(np.linalg.eigvals(M))))
    except Exception:
        return float("inf")

def _max_real_eig(M: np.ndarray) -> float:
    try:
        return float(np.max(np.real(np.linalg.eigvals(M))))
    except Exception:
        return float("inf")

def _ridge_auto(ZZt: np.ndarray, rcond: float) -> float:
    if ZZt.size == 0:
        return 0.0
    smax = np.linalg.svd(ZZt, compute_uv=False)
    smax = float(smax[0]) if smax.size else 0.0
    return rcond * smax

def _closed_form_ridge(X: np.ndarray, Xp: np.ndarray, U: np.ndarray,
                       rcond: float, ridge: Optional[float]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Warm-start via ridge-regularized DMDc."""
    n = X.shape[0]
    Z = np.vstack([X, U])                    # (n+m, T-1)
    ZZt = Z @ Z.T
    lam = _ridge_auto(ZZt, rcond) if ridge is None else float(ridge)
    I = np.eye(ZZt.shape[0])
    Theta = (Xp @ Z.T) @ np.linalg.pinv(ZZt + lam * I, rcond=max(rcond, 1e-12))
    A0 = Theta[:, :n]
    B0 = Theta[:, n:]
    try:
        cond_ZZt = float(np.linalg.cond(ZZt + lam * I))
    except Exception:
        cond_ZZt = None
    diag = {
        "ridge_lambda": lam,
        "cond_ZZt_init": cond_ZZt,
        "smax_Z": float(np.linalg.svd(Z, compute_uv=False)[0]) if Z.size else 0.0,
    }
    return A0, B0, diag

def _loss_and_grads_np(A: np.ndarray, B: np.ndarray, X: np.ndarray, Xp: np.ndarray, U: np.ndarray, lam: float):
    Tm1 = X.shape[1]
    R = A @ X + B @ U - Xp                             # (n, T-1)
    scale = 1.0 / max(Tm1, 1)
    loss = scale * float(np.sum(R * R)) + lam * (float(np.sum(A*A)) + float(np.sum(B*B)))
    # grads
    gA = 2.0 * scale * (R @ X.T) + 2.0 * lam * A
    gB = 2.0 * scale * (R @ U.T) + 2.0 * lam * B
    return loss, gA, gB

def _project_ct_stable_inplace(A: np.ndarray, margin: float = 1e-3) -> bool:
    """Ensure max Re(lambda(A)) <= -margin by shifting left if needed."""
    mr = _max_real_eig(A)
    if not np.isfinite(mr):
        return False
    if mr > -margin:
        A -= (mr + margin) * np.eye(A.shape[0])
        return True
    return False

def _project_dt_stable_inplace(A: np.ndarray, rho_max: float = 0.999) -> bool:
    """Ensure spectral radius <= rho_max by uniform scaling if needed."""
    rho = _spectral_radius(A)
    if not np.isfinite(rho) or rho == 0.0:
        return False
    if rho > rho_max:
        A *= (rho_max / rho)
        return True
    return False


# ----------------------------
# Main API (NumPy path by default; optional JAX)
# ----------------------------
def dmdc_gd_fit(
    X: np.ndarray,             # (n, T-1)
    Xp: np.ndarray,            # (n, T-1)
    U: np.ndarray,             # (m, T-1)
    steps: int = 200,
    rcond: float = 1e-10,
    lr: Optional[float] = None,
    optimizer: str = "adam",   # "adam" or "sgd"
    ridge: Optional[float] = None,     # if None: auto (scale-aware)
    tsvd_energy: Optional[float] = None,  # reserved (not used here, but kept for parity)
    project_stable: Optional[str] = None, # None | "ct" | "dt"
    project_params: Optional[Dict] = None, # {"ct_margin": ..., "dt_rho": ...}
    seed: int = 0,
    use_jax: bool = False,
    jax_x64: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Gradient-based DMDc on the one-step ridge loss:
        min_{A,B} (1/(T-1)) || Xp - A X - B U ||_F^2 + lambda (||A||_F^2 + ||B||_F^2)

    Returns
    -------
    Ahat, Bhat, diag
      diag includes {"final_loss","init_loss","ridge_lambda","lr","optimizer",
                     "steps","lipschitz_L","used_jax",
                     "stable_proj_count","stable_proj_mode", ... }
    """
    # JAX dtype policy
    if use_jax and _JAX_AVAILABLE and jax_x64:
        try:
            from jax import config as _jax_config  # type: ignore
            _jax_config.update("jax_enable_x64", True)
        except Exception:
            pass
        
    rng = np.random.default_rng(seed)
    n, Tm1 = X.shape
    m = U.shape[0]

    # Warm start from closed-form ridge solution
    A, B, d0 = _closed_form_ridge(X, Xp, U, rcond=rcond, ridge=ridge)
    lam = d0["ridge_lambda"]

    # Lipschitz estimate for gradient -> safe step size if lr not given
    Z = np.vstack([X, U])                           # (n+m, T-1)
    smaxZ = float(np.linalg.svd(Z, compute_uv=False)[0]) if Z.size else 0.0
    L = (2.0 / max(Tm1, 1)) * (smaxZ ** 2) + 2.0 * lam
    if lr is None:
        lr = 0.9 / L if L > 0 else 1e-2

    # Initial loss
    # init_loss, _, _ = _loss_and_grads_np(A, B, X, Xp, U, lam)
    init_loss, gA0, gB0 = _loss_and_grads_np(A, B, X, Xp, U, lam)
    best_A, best_B = A.copy(), B.copy()
    best_loss = float(init_loss)
    best_t = 0
    early_stop = False
    final_loss = float(init_loss)

    # --- NumPy optimizer (default) ---
    used_jax = False
    if use_jax and _JAX_AVAILABLE:
        # JAX path (pure-jax Adam/SGD); safe defaults; convert inputs to jnp
        if jax_x64:
            try:
                jax.config.update("jax_enable_x64", True)
            except Exception:
                pass
        used_jax = True

        Xj = jnp.asarray(X)
        Uj = jnp.asarray(U)
        Xpj = jnp.asarray(Xp)
        lamj = jnp.asarray(lam)

        def loss_fn(params):
            Aj, Bj = params
            Rj = Aj @ Xj + Bj @ Uj - Xpj
            scale = 1.0 / max(Tm1, 1)
            return scale * jnp.sum(Rj * Rj) + lamj * (jnp.sum(Aj * Aj) + jnp.sum(Bj * Bj))

        grad_fn = jax.grad(loss_fn, argnums=(0, 1))
        Aj = jnp.asarray(A)
        Bj = jnp.asarray(B)

        # Adam buffers
        if optimizer.lower() == "adam":
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            mA = jnp.zeros_like(Aj); vA = jnp.zeros_like(Aj)
            mB = jnp.zeros_like(Bj); vB = jnp.zeros_like(Bj)

        # Stability settings
        proj_mode = (project_stable or "none").lower()
        ct_margin = (project_params or {}).get("ct_margin", 1e-3)
        dt_rho = (project_params or {}).get("dt_rho", 0.999)
        proj_count = 0

        for t in range(1, steps + 1):
            gA, gB = grad_fn((Aj, Bj))
            gnorm = float(jnp.maximum(jnp.linalg.norm(gA), jnp.linalg.norm(gB)))
            if gnorm < 1e-12 * (1.0 + float(jnp.linalg.norm(Aj)) + float(jnp.linalg.norm(Bj))):
                early_stop = True
                break
            if optimizer.lower() == "adam":
                mA = 0.9 * mA + 0.1 * gA
                vA = 0.999 * vA + 0.001 * (gA * gA)
                mB = 0.9 * mB + 0.1 * gB
                vB = 0.999 * vB + 0.001 * (gB * gB)
                mA_hat = mA / (1.0 - 0.9 ** t)
                vA_hat = vA / (1.0 - 0.999 ** t)
                mB_hat = mB / (1.0 - 0.9 ** t)
                vB_hat = vB / (1.0 - 0.999 ** t)
                Aj = Aj - lr * mA_hat / (jnp.sqrt(vA_hat) + 1e-8)
                Bj = Bj - lr * mB_hat / (jnp.sqrt(vB_hat) + 1e-8)
            else:
                Aj = Aj - lr * gA
                Bj = Bj - lr * gB

            # Stability projection (JAX: we implement via host callback -> convert to NumPy)
            if proj_mode in ("ct", "dt"):
                A_host = np.array(Aj)
                if proj_mode == "ct":
                    did = _project_ct_stable_inplace(A_host, margin=ct_margin)
                else:
                    did = _project_dt_stable_inplace(A_host, rho_max=dt_rho)
                if did:
                    proj_count += 1
                    Aj = jnp.asarray(A_host)
            def _loss_only(Ak, Bk):
                Rj = Ak @ Xj + Bk @ Uj - Xpj
                scale = 1.0 / max(Tm1, 1)
                return float(scale * jnp.sum(Rj * Rj) + lamj * (jnp.sum(Ak * Ak) + jnp.sum(Bk * Bk)))

            cur_loss = _loss_only(Aj, Bj)
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_t = t
                best_A = np.array(Aj)
                best_B = np.array(Bj)

                A = np.array(Aj)
        B = np.array(Bj)
        final_loss, _, _ = _loss_and_grads_np(A, B, X, Xp, U, lam)

        # Safeguard: if we got worse than our best, revert
        reverted_to_best = False
        if best_loss < final_loss * (1.0 - 1e-12):
            A, B = best_A, best_B
            final_loss = best_loss
            reverted_to_best = True

        final_loss, _, _ = _loss_and_grads_np(A, B, X, Xp, U, lam)

    else:
        # NumPy path
        optimizer = optimizer.lower()
        if optimizer not in ("adam", "sgd"):
            optimizer = "adam"

        beta1, beta2, eps = 0.9, 0.999, 1e-8
        mA = np.zeros_like(A); vA = np.zeros_like(A)
        mB = np.zeros_like(B); vB = np.zeros_like(B)

        proj_mode = (project_stable or "none").lower()
        ct_margin = (project_params or {}).get("ct_margin", 1e-3)
        dt_rho = (project_params or {}).get("dt_rho", 0.999)
        proj_count = 0

        for t in range(1, steps + 1):
            loss, gA, gB = _loss_and_grads_np(A, B, X, Xp, U, lam)
            gnorm = max(np.linalg.norm(gA), np.linalg.norm(gB))
            if gnorm < 1e-12 * (1.0 + np.linalg.norm(A) + np.linalg.norm(B)):
                early_stop = True
                break
            if optimizer == "adam":
                # Adam update A
                mA = beta1 * mA + (1.0 - beta1) * gA
                vA = beta2 * vA + (1.0 - beta2) * (gA * gA)
                mA_hat = mA / (1.0 - beta1 ** t)
                vA_hat = vA / (1.0 - beta2 ** t)
                A = A - lr * mA_hat / (np.sqrt(vA_hat) + eps)
                # Adam update B
                mB = beta1 * mB + (1.0 - beta1) * gB
                vB = beta2 * vB + (1.0 - beta2) * (gB * gB)
                mB_hat = mB / (1.0 - beta1 ** t)
                vB_hat = vB / (1.0 - beta2 ** t)
                B = B - lr * mB_hat / (np.sqrt(vB_hat) + eps)
            else:
                # SGD
                A = A - lr * gA
                B = B - lr * gB

            # Stability projection (cheap and safe)
            if proj_mode == "ct":
                if _project_ct_stable_inplace(A, margin=ct_margin):
                    proj_count += 1
            elif proj_mode == "dt":
                if _project_dt_stable_inplace(A, rho_max=dt_rho):
                    proj_count += 1

            # ---- Track best iterate by loss
            cur_loss, _, _ = _loss_and_grads_np(A, B, X, Xp, U, lam)
            if cur_loss < best_loss:
                best_loss = float(cur_loss)
                best_t = t
                best_A = A.copy()
                best_B = B.copy()
                final_loss, _, _ = _loss_and_grads_np(A, B, X, Xp, U, lam)

                reverted_to_best = False
                if best_loss < final_loss * (1.0 - 1e-12):
                    A, B = best_A, best_B
                    final_loss = best_loss
                    reverted_to_best = True


    diag = {
        "init_loss": float(init_loss),
        "final_loss": float(final_loss),
        "ridge_lambda": float(lam),
        "lr": float(lr),
        "optimizer": optimizer.lower(),
        "steps": int(steps),
        "lipschitz_L": float(L),
        "used_jax": bool(used_jax),
        "stable_proj_mode": None if project_stable is None else project_stable.lower(),
        "stable_proj_count": int(proj_count),
        "rcond": float(rcond),
        "smaxZ": float(smaxZ),
        "grad_norm_init": float(max(np.linalg.norm(gA0), np.linalg.norm(gB0))),
        "early_stop": bool(early_stop),
        "reverted_to_best": bool(reverted_to_best),
        "best_iter": int(best_t),

    }
    return A, B, diag


# Back-compat export name
fit = dmdc_gd_fit
__all__ = ["dmdc_gd_fit", "fit"]
