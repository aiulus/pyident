# pyident/estimators.py
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pysindy
import torch


class NODE(torch.nn.Module):
    def __init__(self, n, m, bias: bool = False):
        super().__init__()
        self.fc = torch.nn.Linear(n + m, n, bias=bias)

    def forward(self, t, x_and_u):
        return self.fc(x_and_u)
    
def _ensure_channel_major(U: np.ndarray, expected_T: int) -> np.ndarray:
    """Return ``U`` with shape ``(m, T)``; accept time-major ``(T, m)`` inputs."""

    if U.ndim != 2:
        raise ValueError(f"U must be 2-D, got shape {U.shape}.")

    if U.shape[1] == expected_T:
        return U
    if U.shape[0] == expected_T:
        return U.T

    raise ValueError(
        f"Unable to align U of shape {U.shape} with expected T={expected_T}."
    )


def node_fit_old(Xtrain: np.ndarray,
             Xp: np.ndarray,
             Utrain: np.ndarray,
             dt: float,
             epochs: int = 100):
    n, T = Xtrain.shape
    #m = Utrain.shape[0]
    #device = 'cpu'
    U_cm = _ensure_channel_major(Utrain, T)
    m = U_cm.shape[0]
    device = "cpu"

    model = NODE(n, m).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    Xtrain_torch = torch.tensor(Xtrain.T, dtype=torch.float32, device=device)
    Utrain_torch = torch.tensor(U_cm.T, dtype=torch.float32, device=device)
    Xp_torch = torch.tensor(Xp.T, dtype=torch.float32, device=device)

    for _ in range(epochs):
        optimizer.zero_grad()
        xu = torch.cat([Xtrain_torch, Utrain_torch], dim=1)
        pred = model.fc(xu)
        loss = torch.nn.functional.mse_loss(pred, Xp_torch)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        W = model.fc.weight.detach().cpu().numpy()

    A_node = W[:, :n]
    B_node = W[:, n:]
    return A_node, B_node


def sindy_fit_old(X, Xp, U, dt):
    n, T = X.shape
    U_cm = _ensure_channel_major(U, T)
    X_all = np.hstack([X[:, :1], Xp])
    Xdot = np.empty_like(X)
    if T >= 2:
        Xdot[:, :-1] = (X_all[:, 2:] - X_all[:, :-2]) / (2.0 * dt)
        Xdot[:, 0] = (X_all[:, 1] - X_all[:, 0]) / dt
        Xdot[:, -1] = (X_all[:, -1] - X_all[:, -2]) / dt
    else:
        Xdot[:, 0] = (X_all[:, 1] - X_all[:, -2]) / dt 

    Z = np.vstack([X, U_cm])
    Theta = Xdot @ np.linalg.pinv(Z, rcond=1e-12)
    return Theta[:, :n], Theta[:, n:]

def sindy_fit_dt(X, U, dt, degree=1):
    from pysindy import SINDy
    from pysindy.feature_library import PolynomialLibrary
    lib = PolynomialLibrary(degree=degree, include_bias=False)
    model = SINDy(discrete_time=True, feature_library=lib)
    U_cm = _ensure_channel_major(U, X.shape[1])
    model.fit(X.T, u=U_cm.T, t=dt)
    coef = model.coefficients()  # (n, n+m)
    n = X.shape[0]
    return coef[:, :n], coef[:, n:]

class LinearContinuousNODE(torch.nn.Module):
    """Continuous-time linear NODE with control treated as constant input."""

    def __init__(self, n: int, m: int):
        super().__init__()
        self.A = torch.nn.Parameter(torch.zeros(n, n))
        self.B = torch.nn.Parameter(torch.zeros(n, m))

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        n = self.A.shape[0]
        x = z[..., :n]
        u = z[..., n:]
        dxdt = x @ self.A.T + u @ self.B.T
        dudt = torch.zeros_like(u)
        return torch.cat([dxdt, dudt], dim=-1)


def node_fit(
    Xtrain: np.ndarray,
    Xp: np.ndarray,
    Utrain: np.ndarray,
    dt: float,
    epochs: int = 200,
    lr: float = 1e-2,
    verbose: bool = True,
    log_every: int = 10,
    return_history: bool = False,
):
    """Train a linear continuous-time NODE and return discretized dynamics.

    The model parameterizes continuous-time matrices :math:`(A_c, B_c)` and uses
    the matrix exponential of the block matrix to obtain discrete-time dynamics
    during training. This keeps the NODE interpretation intact while avoiding the
    heavy per-epoch ODE solves that made the original implementation slow.
    """

    n, T = Xtrain.shape
    U_cm = _ensure_channel_major(Utrain, T)
    m = U_cm.shape[0]

    device = torch.device("cpu")
    dtype = torch.float32

    Xk = torch.tensor(Xtrain.T, dtype=dtype, device=device)  # (T, n)
    Uk = torch.tensor(U_cm.T, dtype=dtype, device=device)    # (T, m)
    Xnext = torch.tensor(Xp.T, dtype=dtype, device=device)   # (T, n)

    A_ct = torch.nn.Parameter(torch.zeros((n, n), dtype=dtype, device=device))
    B_ct = torch.nn.Parameter(torch.zeros((n, m), dtype=dtype, device=device))
    params = [A_ct, B_ct]
    optimizer = torch.optim.Adam(params, lr=lr)

    history: list[float] = []

    def _block_matrix() -> torch.Tensor:
        block = torch.zeros((n + m, n + m), dtype=dtype, device=device)
        block[:n, :n] = A_ct
        block[:n, n:] = B_ct
        return block

    for epoch in range(int(epochs)):
        optimizer.zero_grad()

        block = _block_matrix()
        exp_block = torch.matrix_exp(block * float(dt))
        Ad = exp_block[:n, :n]
        Bd = exp_block[:n, n:]

        preds = Xk @ Ad.T + Uk @ Bd.T
        loss = torch.nn.functional.mse_loss(preds, Xnext)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        if return_history:
            history.append(loss_value)
        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"[NODE] epoch {epoch:4d}  mse={loss_value:.6e}")

    with torch.no_grad():
        block = _block_matrix()
        exp_block = torch.matrix_exp(block * float(dt))
        Ad = exp_block[:n, :n].cpu().numpy()
        Bd = exp_block[:n, n:].cpu().numpy()

    if return_history:
        return Ad, Bd, np.array(history)
    return Ad, Bd


def sindy_fit(
    X: np.ndarray,
    Xp: np.ndarray,
    U: np.ndarray,
    dt: float,
    degree: int = 1,
    optimizer: Optional[object] = None,
    feature_library: Optional[object] = None,
    verbose: bool = True,
    return_diagnostics: bool = False,
):
    """Fit a discrete-time SINDy model using PySINDy and log reconstruction error."""

    n, T = X.shape
    U_cm = _ensure_channel_major(U, T)
    # Ensure the samples provided to PySINDy line up with the control inputs.
    # Previously we stacked ``[x_0, Xp]`` and passed an ``(n, T+1)`` array to
    # ``model.fit`` while providing only ``T`` control samples.  Newer releases
    # of PySINDy keep trying to reconcile this mismatch internally, which makes
    # the optimizer spin forever without producing output.  Feeding the ``T``
    # state samples directly and supplying the next-step targets via ``x_dot``
    # keeps the data aligned and avoids the hang.
    X_samples = X.T      # shape (T, n)
    X_targets = Xp.T     # shape (T, n)

    if feature_library is None:
        feature_library = pysindy.feature_library.PolynomialLibrary(
            degree=degree,
            include_bias=False,
        )
    if optimizer is None:
        optimizer = pysindy.optimizers.STLSQ(
            threshold=1e-4,
            max_iter=10,
            normalize_columns=True,
        )

    model = pysindy.SINDy(
        discrete_time=True,
        feature_library=feature_library,
        optimizer=optimizer,
    )
    model.fit(X_samples, u=U_cm.T, x_dot=X_targets, t=dt)

    coef = model.coefficients()
    Ahat = coef[:, :n]
    Bhat = coef[:, n:]

    X_pred = model.predict(X_samples, u=U_cm.T)
    recon_mse = float(np.mean((X_pred - Xp.T) ** 2))
    sparsity = float(np.mean(np.abs(coef) > 0))

    if verbose:
        print(
            f"[SINDy] recon_mse={recon_mse:.6e}  coeff_density={sparsity:.3f}"
        )

    if return_diagnostics:
        diagnostics = {
            "reconstruction_mse": recon_mse,
            "coefficient_density": sparsity,
        }
        return Ahat, Bhat, diagnostics

    return Ahat, Bhat


# -----------------------------
# 1) DMDc (minimum-norm pinv)
# -----------------------------
def dmdc_pinv(
    X: np.ndarray,   # (n, T)
    Xp: np.ndarray,  # (n, T)
    U: np.ndarray,   # (m, T)
    rcond: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimum-norm DMDc via pseudoinverse.
    Solves Xp ≈ [A B] [X; U] with Θ = Xp Z^+ , Z=[X;U].
    Exact in noiseless data; returns a solution even if Z is rank-deficient.
    """
    n, T = X.shape
    U_cm = _ensure_channel_major(U, T)
    Z = np.vstack([X, U_cm])  
    Z_pinv = np.linalg.pinv(Z, rcond=rcond)
    AB = Xp @ Z_pinv                   # (n, n+m)
    Ahat = AB[:, :n]
    Bhat = AB[:, n:]
    return Ahat, Bhat

# Backward-compat shim used across the repo/tests
def dmdc_fit(X: np.ndarray, Xp: np.ndarray, U: np.ndarray, rcond: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    return dmdc_pinv(X, Xp, U, rcond=rcond)


# -----------------------------
# 2) DMDc (ridge / Tikhonov)
# -----------------------------
def dmdc_ridge(
    X: np.ndarray, Xp: np.ndarray, U: np.ndarray, lam: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ridge-regularized DMDc:
      Θ = (Xp Z^T) (Z Z^T + λ I)^{-1},  Z=[X;U].
    More stable when Z is ill-conditioned or short.
    """
    n, T = X.shape
    U_cm = _ensure_channel_major(U, T)
    Z = np.vstack([X, U_cm])  
    ZZt = Z @ Z.T
    G = ZZt + lam * np.eye(ZZt.shape[0], dtype=ZZt.dtype)
    AB = (Xp @ Z.T) @ np.linalg.solve(G, np.eye(G.shape[0], dtype=G.dtype))
    Ahat = AB[:, :n]
    Bhat = AB[:, n:]
    return Ahat, Bhat


# -----------------------------
# 3) DMDc (truncated SVD)
# -----------------------------
def dmdc_tsvd(
    X: np.ndarray, Xp: np.ndarray, U: np.ndarray,
    rank: Optional[int] = None, svd_tol: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Truncated-SVD DMDc:
      Z = U Σ V^T ; use top-r or tol-based truncation to form Z^+_r.
    Useful to restrict to the excited/identifiable subspace explicitly.
    """
    n, T = X.shape
    U_cm = _ensure_channel_major(U, T)
    Z = np.vstack([X, U_cm]) 
    Uz, Sz, Vtz = np.linalg.svd(Z, full_matrices=False)
    if rank is None:
        if svd_tol is not None:
            r = int(np.sum(Sz > svd_tol))
        else:
            r = int(np.sum(Sz > max(1e-12, Sz.max() * 1e-12)))
    else:
        r = int(min(rank, Sz.size))
    if r <= 0:
        # fall back to pinv if everything is tiny
        return dmdc_pinv(X, Xp, U, rcond=1e-10)
    Z_pinv_r = (Vtz[:r, :].T) @ np.diag(1.0 / Sz[:r]) @ (Uz[:, :r].T)
    AB = Xp @ Z_pinv_r
    Ahat = AB[:, :n]
    Bhat = AB[:, n:]
    return Ahat, Bhat


# ----------------------------------------------------
# 4) MOESP (full-state) + one-step B refit (simplified)
# ----------------------------------------------------
def moesp_fullstate(
    u_ts: np.ndarray,   # (T, m)
    x_ts: np.ndarray,   # (T, n) full state (C=I), same T as u
    n: int,
    i: Optional[int] = None,   # kept for signature parity; not used in simplified flow
    f: Optional[int] = None,   # kept for signature parity; not used in simplified flow
    rcond: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full-state identification in the noiseless case can be done via one-step LS,
    which equals the MOESP A-estimate under ideal conditions. We compute A,B from:
        Xp = A X + B U
    where X = x_ts[:-1].T, U = u_ts[:-1].T, Xp = x_ts[1:].T.
    """
    X  = x_ts[:-1].T   # (n, T-1)
    Xp = x_ts[1:].T    # (n, T-1)
    U  = u_ts[:-1].T   # (m, T-1)
    # use minimum-norm solution; for conditioning, user can switch to ridge/tsvd
    return dmdc_pinv(X, Xp, U, rcond=rcond)


# Backward-compat wrapper: same call sites as old code/tests
def moesp_fit(
    X: np.ndarray,      # (n, T)
    Xp: np.ndarray,     # (n, T)
    U: np.ndarray,      # (m, T)
    s: Optional[int] = None,   # unused in simplified flow
    n: Optional[int] = None,
    rcond: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper to accept (X,Xp,U) blocks (as used in run_single/tests) and call
    moesp_fullstate with time-major sequences.
    """
    T = X.shape[1]
    U_cm = _ensure_channel_major(U, T)
    assert Xp.shape[1] == T, "time lengths must match"
    u_ts = U_cm.T 
    x_ts = X.T                       # (T, n)
    n_use = int(n if n is not None else X.shape[0])
    return moesp_fullstate(u_ts, x_ts, n=n_use, i=s, f=None, rcond=rcond)

# ----------------------------------------------------
# 4) Noise-sensitive DMDc variants (TLS, IV)
# ----------------------------------------------------

def dmdc_tls(X: np.ndarray, Xp: np.ndarray, U: np.ndarray,
             rcond: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """
    Row-wise scaled TLS for Xp ≈ Θ [X;U].
    Column-scale the augmented matrix to improve conditioning; fallback to OLS if needed.
    """
    n, T = X.shape
    U_cm = _ensure_channel_major(U, T)
    Z = np.vstack([X, U_cm]).T 
    Y = Xp.T                                   

    Theta = np.zeros((n, Z.shape[1]), dtype=X.dtype)
    for j in range(n):
        M = np.hstack([Z, Y[:, [j]]])          # (T, n+m+1)
        # column scaling
        scales = np.linalg.norm(M, axis=0)
        scales[scales == 0.0] = 1.0
        Ms = M / scales

        _, _, Vt = np.linalg.svd(Ms, full_matrices=False)
        v = Vt[-1, :] / scales                 # unscale

        denom = v[-1]
        if abs(denom) < 1e-12:
            # fallback: ridge-OLS on this row
            lam = 1e-6
            G = (Z.T @ Z) + lam*np.eye(Z.shape[1])
            theta_j = np.linalg.solve(G, Z.T @ Y[:, j])
        else:
            theta_j = -v[:-1] / denom
        Theta[j, :] = theta_j

    Ahat = Theta[:, :n]
    Bhat = Theta[:, n:]
    return Ahat, Bhat



def _lag_stack(U: np.ndarray, L: int) -> np.ndarray:
    # U: (m, T) → stack [u_{t-1}; …; u_{t-L}] for t=L..T-1 → (mL, T-L)
    m, T = U.shape
    return np.vstack([U[:, L-ell:T-ell] for ell in range(1, L+1)])

def dmdc_iv(X: np.ndarray, Xp: np.ndarray, U: np.ndarray,
            L: int = 1, instruments: Optional[np.ndarray] = None,
            rcond: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """
    2SLS IV-DMDc:
      1) Project regressors Z onto instrument space Π_Φ
      2) OLS of Y on Z_hat
    """
    n, T = X.shape
    U_cm = _ensure_channel_major(U, T)
    assert Xp.shape[1] == T and U_cm.shape[1] == T
    if L < 1 or L >= T:
        raise ValueError("L must be in [1, T-1].")

    Y = Xp[:, L:]                               # (n, T-L)
    Z = np.vstack([X[:, L-1:T-1], U_cm[:, L-1:T-1]])  # ((n+m), T-L)

    if instruments is None:
        Phi = _lag_stack(U_cm, L)                  # (mL, T-L)
    else:
        assert instruments.shape[1] == T, "instruments must have same time length as X"
        Phi = instruments[:, L:]                # (p, T-L)

    # Π_Φ in time domain
    G = Phi @ Phi.T                             # (p, p)
    Gp = np.linalg.pinv(G, rcond=rcond)         # robust if Phi low-rank
    Pi = Phi.T @ Gp @ Phi                       # (T-L, T-L)

    Zhat = Z @ Pi                               # project regressors onto instrument span

    ZZ = Zhat @ Zhat.T                          # ((n+m), (n+m))
    Theta = (Y @ Zhat.T) @ np.linalg.pinv(ZZ, rcond=rcond)

    return Theta[:, :n], Theta[:, n:]


# ----------------------------------------------------
# 5) Identifiable-component projector (utility)
# ----------------------------------------------------
def project_identifiable(theta: np.ndarray, Z: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """
    Project parameter matrix Θ onto the identifiable component given regressors Z=[X;U].
    This right-multiplies by the projector onto col(Z):
        P = Z Z^+   (shape (n+m)x(n+m)),
    so Θ_ident = Θ P only alters directions that were not excited by the data.
    """
    P = Z @ np.linalg.pinv(Z, rcond=rcond)   # projector onto col(Z)
    return theta @ P
