from __future__ import annotations

class SeedPolicy:
    """Shared RNGs for NumPy/JAX with a single seed."""
    def __init__(self, seed: int):
        import numpy as np
        self.seed = int(seed)
        self.np_rng = np.random.default_rng(self.seed)
        try:
            import jax  # type: ignore
            self.jax_key = jax.random.PRNGKey(self.seed)  # type: ignore
        except Exception:
            self.jax_key = None

