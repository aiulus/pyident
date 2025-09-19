from __future__ import annotations
from typing import Optional, Tuple
from functools import partial


import os
if os.environ.get("JAX_PLATFORM_NAME","").lower() == "metal":
    os.environ.setdefault("JAX_ENABLE_X64", "0")
else:
    os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
from jax import lax

# ---------------------------
# Global toggles / utilities
# ---------------------------

def enable_x64(flag: bool = True) -> None:
    try:
        from jax import config as _jax_config
        _jax_config.update("jax_enable_x64", bool(flag))
    except Exception:
        pass


def _to_f64(*xs):
    return tuple(jnp.asarray(x, dtype=jnp.float64) for x in xs)

# ---------------------------
# Discrete-time simulation
# ---------------------------

@jax.jit
def simulate_discrete(Ad: jnp.ndarray,
                      Bd: jnp.ndarray,
                      u: jnp.ndarray,
                      x0: jnp.ndarray) -> jnp.ndarray:
    """
    Simulate x_{k+1} = Ad x_k + Bd u_k over k=0..T-1.

    Shapes:
      Ad: (n,n), Bd: (n,m)
      u : (T, m)      -- note time-major for JAX-friendly scan
      x0: (n,)

    Returns:
      X: (n, T+1) with X[:,0]=x0
    """
    Ad, Bd, u, x0 = _to_f64(Ad, Bd, u, x0)

    def step(x, uk):
        x_next = Ad @ x + Bd @ uk
        return x_next, x_next

    x_last, xs = lax.scan(step, x0, u)  # xs: (T, n)
    X = jnp.concatenate([x0[None, :], xs], axis=0).T  # (n, T+1)
    return X


@jax.jit
def simulate_discrete_batch(Ad: jnp.ndarray,
                            Bd: jnp.ndarray,
                            U: jnp.ndarray,
                            X0: jnp.ndarray) -> jnp.ndarray:
    """
    Batched simulation.

    Shapes:
      Ad: (B, n, n), Bd: (B, n, m)
      U : (B, T, m)
      X0: (B, n)

    Returns:
      X: (B, n, T+1)
    """
    def one(Ad_i, Bd_i, U_i, X0_i):
        return simulate_discrete(Ad_i, Bd_i, U_i, X0_i)
    return jax.vmap(one)(Ad, Bd, U, X0)

# ---------------------------
# Krylov / unified generator
# ---------------------------

def _krylov(A: jnp.ndarray, X: jnp.ndarray, depth: int) -> jnp.ndarray:
    """[X, AX, ..., A^{depth-1} X] with depth>=0."""
    n = A.shape[0]
    if depth <= 0:
        return jnp.zeros((n, 0), dtype=A.dtype)

    def body(carry, _):
        M = carry
        return A @ M, M  # yield previous M
    M0 = X
    _, Ms = lax.scan(body, M0, jnp.arange(depth))
    return jnp.concatenate(Ms, axis=1)

def krylov_generator(A: jnp.ndarray, X: jnp.ndarray, depth: Optional[int] = None) -> jnp.ndarray:
    n = A.shape[0]
    d = n if depth is None else int(depth)
    return _krylov(A, X, d)

@partial(jax.jit, static_argnames=("mode","r"))
def unified_generator(A: jnp.ndarray,
                      B: jnp.ndarray,
                      x0: jnp.ndarray,
                      mode: str = "unrestricted",
                      W: Optional[jnp.ndarray] = None,
                      r: Optional[int] = None) -> jnp.ndarray:
    """
    JAX version of the unified generator K(U; x0).

    - mode="unrestricted": K = Krylov(A, [x0 B], n)
    - mode="pointwise":    K = Krylov(A, [x0 BW], n) where BW projects B onto span(W)
    - mode="moment-pe":    K = [Krylov(A,[x0 B], r-1), A^r x0, ..., A^{n-1} x0]
    """
    A, B, x0 = _to_f64(A, B, x0)
    x0 = x0.reshape(-1, 1)

    if mode == "unrestricted":
        Kcore = jnp.concatenate([x0, B], axis=1)
        K = _krylov(A, Kcore, A.shape[0])

    elif mode == "pointwise":
        assert W is not None, "W required for pointwise mode"
        # Project onto span(W) via QR projector (more stable than pinv)
        QW, _ = jnp.linalg.qr(jnp.asarray(W, dtype=jnp.float64), mode="reduced")
        BW = B @ (QW @ QW.T)
        Kcore = jnp.concatenate([x0, BW], axis=1)
        K = _krylov(A, Kcore, A.shape[0])

    elif mode == "moment-pe":
        assert r is not None and r >= 1
        Kcore = jnp.concatenate([x0, B], axis=1)
        K1 = _krylov(A, Kcore, max(0, r - 1))
        x_r = jnp.linalg.matrix_power(A, int(r)) @ x0
        K2 = _krylov(A, x_r, int(A.shape[0]) - int(r)) if r < A.shape[0] else jnp.zeros((A.shape[0], 0), dtype=A.dtype)
        K = jnp.concatenate([K1, K2], axis=1)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return K


def _apowers_on_x0(A, x0, start, stop):
    # returns [A^start x0, ..., A^(stop-1) x0] as a list-like stack
    n = A.shape[0]
    def body(carry, _):
        v, cols = carry
        cols = cols.append(v)  # use an explicit concatenation pattern below
        return (A @ v, cols)
    # start by advancing to A^start x0
    v0 = x0
    v0 = lax.fori_loop(0, start, lambda i, v: A @ v, v0)
    # now collect stop-start columns
    def push(acc, v): return jnp.concatenate([acc, v], axis=1)
    cols0 = jnp.zeros((n,0), dtype=A.dtype)
    v_end, cols = lax.fori_loop(0, stop-start,
                                lambda i, c: (A @ c[0], (A @ c[0], push(c[1], c[0]))),
                                (v0, cols0))
    return cols  # (n, stop-start)


# ---------------------------
# Thin basis from a generator
# ---------------------------

def thin_basis(K: jnp.ndarray, rtol: float = 1e-8, atol: float = 0.0) -> jnp.ndarray:
    """
    Return a thin orthonormal basis for span(K) using JAX SVD.
    """
    U, S, _ = jnp.linalg.svd(K, full_matrices=False)
    thresh = rtol * (S[0] if S.size > 0 else 0.0) + atol
    r = jnp.sum(S > thresh)
    return U[:, : r]

# ---------------------------
# Structured PBH margin (x0 fixed)
# ---------------------------

def _householder_orth_complement(v: jnp.ndarray) -> jnp.ndarray:
    """
    Build Q ∈ R^{n×(n-1)} whose columns form an orthonormal basis of v^⊥.
    Uses a Householder reflection that maps v to ±||v|| e1.
    """
    v = v.reshape(-1)
    n = v.shape[0]
    nv = jnp.linalg.norm(v)
    # Handle zero vector robustly: fall back to e1
    e1 = jnp.zeros_like(v).at[0].set(1.0)
    v_unit = jnp.where(nv > 0, v / nv, e1)
    sign = jnp.where(v_unit[0] >= 0, 1.0, -1.0)
    u = v_unit + sign * e1
    nu = jnp.linalg.norm(u)
    # If v ~ -e1, u ~ 0; handle with a fallback axis
    def _fallback():
        # choose e2 as axis if available
        e2 = jnp.zeros_like(v).at[jnp.minimum(1, n-1)].set(1.0)
        u2 = v_unit + e2
        return u2 / jnp.linalg.norm(u2)
    w = jax.lax.cond(nu > 1e-15, lambda _: u / nu, lambda _: _fallback(), operand=None)
    H = jnp.eye(n) - 2.0 * jnp.outer(w, w)  # Householder
    # Columns 1..n-1 span e1^⊥ in the reflected space; map back = same H
    Q = H[:, 1:]
    return Q


@partial(jax.jit, static_argnames=("mode_for_K","r"))
def pbh_margin_structured(A, B, x0, eigvals=None, mode_for_K="unrestricted", W=None, r=None):
    A, B, x0 = _to_f64(A, B, x0)
    n = A.shape[0]
    eigvals = jnp.linalg.eigvals(A) if eigvals is None else eigvals

    # Build [x0 B*]
    if mode_for_K == "unrestricted" or mode_for_K == "moment-pe":
        Bstar = B
    elif mode_for_K == "pointwise":
        QW, _ = jnp.linalg.qr(jnp.asarray(W, dtype=A.dtype), mode="reduced")
        Bstar = B @ (QW @ QW.T)
    else:
        raise ValueError

    aug = jnp.concatenate([x0.reshape(-1,1), Bstar], axis=1)
    Q = _householder_orth_complement(x0)

    def _smin_realfield(QM):
        H = QM.conj().T @ QM
        Hr = jnp.block([[jnp.real(H), -jnp.imag(H)],
    +                    [jnp.imag(H),  jnp.real(H)]])
        lam_min = jnp.linalg.eigvalsh(Hr)[0]
        return jnp.sqrt(jnp.maximum(lam_min, 0.0))
    
    def smin_for_lambda(lam):
        M  = jnp.concatenate([lam * jnp.eye(n) - A, aug], axis=1).astype(jnp.complex64)
        QM = Q.T @ M
        return _smin_realfield(QM)

    return jax.vmap(smin_for_lambda)(eigvals).min()


# ---------------------------
# JAX DMDc (optional)
# ---------------------------

def dmdc_fit_jax(X: jnp.ndarray,
                 Xp: jnp.ndarray,
                 U: jnp.ndarray,
                 rcond: float = 1e-10,
                 ridge: Optional[float] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solve Xp ≈ [A B][X;U] in least squares.
    Shapes: X,Xp (n,T-1), U (m,T-1). Returns Ahat(n,n), Bhat(n,m).
    """
    X = jnp.asarray(X, dtype=jnp.float64)
    Xp = jnp.asarray(Xp, dtype=jnp.float64)
    U = jnp.asarray(U, dtype=jnp.float64)
    n = X.shape[0]
    Z = jnp.concatenate([X, U], axis=0)  # (n+m, T-1)

    if ridge is not None and ridge > 0:
        ZZt = Z @ Z.T
        G = ZZt + ridge * jnp.eye(ZZt.shape[0], dtype=ZZt.dtype)
        AB = (Xp @ Z.T) @ jnp.linalg.solve(G, jnp.eye(G.shape[0], dtype=G.dtype))
    else:
        # JAX lstsq is stable and compiled
        AB_T, _, _, _ = jnp.linalg.lstsq(Z.T, Xp.T, rcond=rcond)
        AB = AB_T.T

    Ahat = AB[:, :n]
    Bhat = AB[:, n:]
    return Ahat, Bhat

# ---------------------------
# JAX utils
# ---------------------------

def _to_policy_dtype(*xs):
    use64 = bool(jax.config.read("jax_enable_x64"))
    dt = jnp.float64 if use64 else jnp.float32
    return tuple(jnp.asarray(x, dtype=dt) for x in xs)

def backend_info() -> dict:
    try:
        import jax
        return {
            "backend": jax.default_backend(),
            "version": getattr(jax, "__version__", None),
        }
    except Exception:
        return {"backend": "none", "version": None}
