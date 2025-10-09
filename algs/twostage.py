"""
Two-stage least-squares–like identification for linear ODEs
===========================================================

Implements the "two-stage" estimators from the R snippets you shared:
  • twostage1: finite-difference variant (simple two-stage)
  • twostage2: smoothing-spline / functional variant (functional two-stage)

Both return Â = (Y L Yᵀ) (Y S Yᵀ)^(-1) with appropriate S, L.

Input conventions
-----------------
Y : (dxn) array — rows are state dimensions, columns are time samples
T : (n,) array — strictly increasing time stamps (for twostage2)
dt: float       — constant step size (for twostage1)

Dependencies
------------
NumPy is required. SciPy is required only for twostage2 (BSpline).

References
----------
Stanhope, Rubin & Swigon (2014) — Identifiability from a single trajectory.
Qiu et al. (2020+) — Identifiability Analysis of Linear ODE Systems with a Single Trajectory.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike

try:
    from scipy.interpolate import BSpline  # type: ignore
except Exception:  # pragma: no cover
    BSpline = None  # twostage2 will raise if SciPy is unavailable


# ------------------------------
# Numerical helpers
# ------------------------------
def rsolve(mat: ArrayLike, d_prop: float = 1e-6, dmin: float = 1e-9, dmax: float = 1e9) -> np.ndarray:
    """Numerically robust pseudo-inverse via SVD with adaptive threshold.

    Parameters
    ----------
    mat : array_like, shape (p, q)
        Matrix to pseudo-invert.
    d_prop, dmin, dmax : float
        Threshold = clip(sum(singular_values) * d_prop, dmin, dmax).

    Returns
    -------
    pinv : ndarray, shape (q, p)
    """
    U, s, Vt = np.linalg.svd(np.asarray(mat), full_matrices=False)
    thresh = float(np.clip(s.sum() * d_prop, dmin, dmax))
    sinv = np.where(s >= thresh, 1.0 / s, 0.0)
    return (Vt.T * sinv) @ U.T


def _first_diff_matrix(n: int) -> np.ndarray:
    """Forward finite-difference operator L (nxn) with a trailing zero column.

    For 1≤k≤n−1, column k encodes (x_{k+1} − x_k); last column is zeros.
    """
    if n < 2:
        raise ValueError("n must be ≥ 2")
    E = np.eye(n - 1)
    lower = np.vstack([E, np.zeros((1, n - 1))])  # diag at rows 1..n-1
    upper = np.vstack([np.zeros((1, n - 1)), E])  # diag at rows 2..n
    D = -lower + upper  # nx(n-1)
    L = np.column_stack([D, np.zeros((n, 1))])    # nxn
    return L


def _trapezoid_weights(T: np.ndarray) -> np.ndarray:
    """Trapezoidal quadrature weights over a monotone grid T (n,)."""
    T = np.asarray(T, dtype=float)
    if T.ndim != 1 or T.size < 2:
        raise ValueError("T must be a 1D array with length ≥ 2")
    w = np.empty_like(T)
    w[1:-1] = 0.5 * (T[2:] - T[:-2])
    w[0] = 0.5 * (T[1] - T[0])
    w[-1] = 0.5 * (T[-1] - T[-2])
    return w


# ------------------------------
# Two-stage (finite-difference)
# ------------------------------
def twostage1(Y: ArrayLike, dt: float) -> dict:
    """Simple two-stage estimator (finite differences).

    Parameters
    ----------
    Y : array_like, shape (d, n)
        Discrete state observations (rows: dimensions, cols: time).
    dt : float
        Constant time step between consecutive samples.

    Returns
    -------
    out : dict
        {
          'Ahat': (d, d) estimated system matrix,
          'S':    (n, n) identity (inner product on x),
          'L':    (n, n) finite-difference inner-product mapping,
          'x0':   (d,)   initial state estimate (Y[:, 0])
        }
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be a 2D array of shape (d, n)")
    d, n = Y.shape
    if n < 2:
        raise ValueError("Need at least two time samples")
    if dt <= 0:
        raise ValueError("dt must be positive")

    L = _first_diff_matrix(n) / float(dt)
    S = np.eye(n)
    YYt = Y @ Y.T  # (dxd)
    Ahat = (Y @ L @ Y.T) @ rsolve(YYt)
    x0 = Y[:, 0].copy()
    return {"Ahat": Ahat, "S": S, "L": L, "x0": x0}


# ------------------------------
# Two-stage (functional / smoothing-spline)
# ------------------------------
# We construct a shared B-spline basis across dimensions, derive the linear
# map y -> c (smoothing coefficients), and form S = y2cᵀ * Jb * y2c and
# L = y2cᵀ * C * y2c where Jb = ∫ Bᵀ B dt and C = ∫ B'ᵀ B dt
# approximated by trapezoidal quadrature on a dense grid.

def _bspline_knots(T: np.ndarray, degree: int) -> np.ndarray:
    """Augmented knot vector with interior knots at T[1:-1] and degree+1 end repeats."""
    t0, tN = float(T[0]), float(T[-1])
    interior = np.array(T[1:-1], dtype=float)
    t = np.concatenate([
        np.full(degree + 1, t0),
        interior,
        np.full(degree + 1, tN),
    ])
    return t


def _bspline_design(T_eval: np.ndarray, T_knots: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (B, Bdot, Bpp, t) evaluated at T_eval for a basis with knots from T_knots.

    B(i,j) = basis_j(T_eval[i]) with degree `degree`.
    """
    if BSpline is None:  # pragma: no cover
        raise ImportError("scipy is required for twostage2 (scipy.interpolate.BSpline)")
    t = _bspline_knots(T_knots, degree)
    nb = len(t) - degree - 1
    m = len(T_eval)
    B = np.empty((m, nb))
    Bdot = np.empty_like(B)
    Bpp = np.empty_like(B)
    for j in range(nb):
        coeff = np.zeros(nb)
        coeff[j] = 1.0
        spl = BSpline(t, coeff, degree, extrapolate=False)
        spl1 = spl.derivative(1)
        spl2 = spl.derivative(2)
        B[:, j] = spl(T_eval)
        Bdot[:, j] = spl1(T_eval)
        Bpp[:, j] = spl2(T_eval)
    return B, Bdot, Bpp, t


def _dense_grid(T: np.ndarray, factor: int = 3) -> np.ndarray:
    M = max(200, factor * (len(T) - 1))
    return np.linspace(float(T[0]), float(T[-1]), M)


def _y2c_map_and_B(T: np.ndarray, degree: int, rough_pen: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute y→c linear map and the design matrix B at observation grid T."""
    # Design at observation grid
    B, _, Bpp, _ = _bspline_design(T, T, degree)
    # Roughness penalty matrix R ≈ ∫ (B''ᵀ B'') dt on a dense grid
    Td = _dense_grid(T)
    Bd, _, Bpp_d, _ = _bspline_design(Td, T, degree)
    w_int = _trapezoid_weights(Td)
    R = (Bpp_d.T * w_int) @ Bpp_d  # (nbxnb)
    # Observation weighting (identity)
    G = B.T @ B  # (nbxnb)
    M = G + float(rough_pen) * R
    y2c = rsolve(M) @ B.T  # (nbxn)
    return y2c, B


def twostage2(Y: ArrayLike, T: ArrayLike, degree: int = 3, rough_pen: float = 1e-3) -> dict:
    """Functional two-stage estimator (smoothing-spline variant).

    Parameters
    ----------
    Y : array_like, shape (d, n)
        Discrete state observations.
    T : array_like, shape (n,)
        Strictly increasing sample times (will be cast to float).
    degree : int, optional
        B-spline polynomial degree (default: 3 → cubic splines).
    rough_pen : float, optional
        Roughness penalty λ multiplying ∫ |x''(t)|² dt.

    Returns
    -------
    out : dict
        {
          'Ahat':  (d, d) estimated system matrix,
          'S':     (n, n) inner-product matrix for x,
          'L':     (n, n) cross inner-product matrix for (ẋ, x),
          'xt_hat':(d, n) smoothed trajectories at times T,
          'x0':    (d,)   initial state estimate from smoothed curves
        }
    """
    if BSpline is None:  # pragma: no cover
        raise ImportError("twostage2 requires SciPy (scipy.interpolate.BSpline)")

    Y = np.asarray(Y, dtype=float)
    T = np.asarray(T, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be a 2D array of shape (d, n)")
    d, n = Y.shape
    if T.ndim != 1 or T.size != n:
        raise ValueError("T must be a 1D array with the same number of samples as Y columns")
    if not np.all(np.diff(T) > 0):
        raise ValueError("T must be strictly increasing")

    # Linear map from observations to spline coefficients (shared across dims)
    y2c, B = _y2c_map_and_B(T, degree, rough_pen)

    # Integral matrices on a dense grid
    Td = _dense_grid(T)
    Bd, Bdot_d, _, _ = _bspline_design(Td, T, degree)
    w_int = _trapezoid_weights(Td)
    Jb = (Bd.T * w_int) @ Bd          # ∫ Bᵀ B dt
    C = (Bdot_d.T * w_int) @ Bd       # ∫ (B')ᵀ B dt

    # Build S and L in the R sense so that Y L Yᵀ ≈ ∫ ẋ xᵀ dt, Y S Yᵀ ≈ ∫ x xᵀ dt
    S = y2c.T @ Jb @ y2c              # (nxn)
    L = y2c.T @ C  @ y2c              # (nxn)

    # Two-stage estimate
    YSYt = Y @ S @ Y.T
    YLYt = Y @ L @ Y.T
    Ahat = YLYt @ rsolve(YSYt)

    # Smoothed trajectories at observation times
    C_all = y2c @ Y.T                 # (nbxd)
    xt_hat = (B @ C_all).T            # (dxn)
    x0 = xt_hat[:, 0].copy()

    return {"Ahat": Ahat, "S": S, "L": L, "xt_hat": xt_hat, "x0": x0}


# ------------------------------
# Minimal usage example (doctest-style, not executed here)
# ------------------------------
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(0)
    d, n = 3, 200
    A_true = np.array([[0.0, 1.0, 0.0],
                       [-2.0, -0.5, 0.0],
                       [0.0, 0.0, -0.2]])
    T = np.linspace(0.0, 10.0, n)
    dt = T[1] - T[0]
    # simulate noiseless linear ODE x' = A x by Euler for demo
    x = np.zeros((d, n))
    x[:, 0] = rng.standard_normal(d)
    for k in range(n - 1):
        x[:, k + 1] = x[:, k] + dt * (A_true @ x[:, k])
    Y = x + 0.01 * rng.standard_normal(x.shape)

    out1 = twostage1(Y, dt)
    print("twostage1 ||Ahat - A_true||_F:", np.linalg.norm(out1["Ahat"] - A_true))

    if BSpline is not None:
        out2 = twostage2(Y, T, degree=3, rough_pen=1e-2)
        print("twostage2 ||Ahat - A_true||_F:", np.linalg.norm(out2["Ahat"] - A_true))
