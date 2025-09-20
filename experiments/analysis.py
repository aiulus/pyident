from __future__ import annotations
import numpy as np
from typing import Dict, Any, Iterable, Tuple

def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    denom = 1.0 + z * z / n
    center = (p_hat + (z*z)/(2*n)) / denom
    half = z * np.sqrt(p_hat*(1-p_hat)/n + (z*z)/(4*n*n)) / denom
    return float(center - half), float(center + half)

def est_success(di: Dict[str, Any], tol: float = 1e-6) -> bool:
    # success definition on projected errors (available in run_single outputs)
    if not isinstance(di, dict):
        return False
    return ("A_err_PV" in di and "B_err_PV" in di
            and di["A_err_PV"] <= tol and di["B_err_PV"] <= tol)

def summarize_array(xs: Iterable[float]) -> Dict[str, float]:
    xs = np.asarray(list(xs), dtype=float)
    if xs.size == 0:
        return {"median": np.nan, "q10": np.nan, "q90": np.nan, "mean": np.nan}
    return {
        "median": float(np.median(xs)),
        "q10": float(np.quantile(xs, 0.10)),
        "q90": float(np.quantile(xs, 0.90)),
        "mean": float(np.mean(xs)),
    }
