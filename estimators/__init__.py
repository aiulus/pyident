# pyident/estimators/__init__.py

from .dmdc import dmdc_fit

# Lazy wrapper to avoid importing moesp on package import
def moesp_core(u, y, s, n, rcond: float = 1e-10):
    from .moesp import moesp  # imported only when needed
    return moesp(u, y, s, n, rcond=rcond)

__all__ = ["dmdc_fit", "moesp_core"]
