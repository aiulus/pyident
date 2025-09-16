import numpy as np


from __future__ import annotations
import numpy as np

def available():
    try:
        import jax  # noqa: F401
        return True
    except Exception:
        return False

def batch_sigma_min(Q, A, B, lam_grid):
    try:
        import jax
        import jax.numpy as jnp
    except Exception:
        raise RuntimeError("JAX not available")

    def smin_block(M):
        s = jnp.linalg.svd(M, compute_uv=False)
        return s[-1]

    def concat_lambda_block(A,B,lam):
        n = A.shape[0]
        return jnp.concatenate((lam*jnp.eye(n, dtype=A.dtype)-A, B), axis=1)

    def phi(lam):
        M = Q.conj().T @ concat_lambda_block(A,B,lam)
        return smin_block(M)

    vphi = jax.vmap(phi)
    return jax.jit(vphi)(lam_grid)

def coarse_lambda_grid(A, n_alpha=64, n_beta=64, radius_scale=2.0):
    try:
        import jax.numpy as jnp
    except Exception:
        import numpy as jnp
    w = jnp.linalg.norm(A, ord=2)
    r = radius_scale * w
    alpha = jnp.linspace(-r, r, n_alpha)
    beta  = jnp.linspace(-r, r, n_beta)
    AA, BB = jnp.meshgrid(alpha, beta, indexing="ij")
    return (AA + 1j*BB).reshape(-1)
