from __future__ import annotations

def runtime_banner():
    """Lightweight runtime environment capture for reproducibility logs."""
    import sys
    import numpy as np
    try:
        import jax  # type: ignore
        accelerator = getattr(jax, "default_backend", lambda: "unknown")()
        jax_ver = getattr(jax, "__version__", None)
        try:
            from jax import config as _jax_config  # type: ignore
            x64_enabled = bool(getattr(_jax_config, "read", lambda *a, **k: False)("jax_enable_x64"))
        except Exception:
            x64_enabled = None
    except Exception:
        accelerator = "none"
        jax_ver = None
        x64_enabled = None

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "jax": jax_ver,
        "accelerator": accelerator,
        "jax_x64": x64_enabled,
        "dtype_default": str(np.dtype(float)),
    }
