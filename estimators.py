# pyident/estimators.py
from __future__ import annotations
from typing import Optional, Tuple, Union, Dict, Any
import numpy as np
import scipy.linalg
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
    epochs: int = 500,
    lr: float = 1e-2,
    verbose: bool = True,
    log_every: int = 10,
    return_history: bool = False,
    return_diagnostics: bool = False,
    # Modern ML monitoring parameters
    patience: int = 25,
    min_delta: float = 1e-6,  # Relaxed from 1e-8 per feedback
    convergence_tol: float = 1e-6,  # Relaxed from 1e-8 per feedback
    max_grad_norm: float = 10.0,
    early_stopping: bool = True,
    # Log file parameters
    log_file: str | None = None,
    log_append: bool = True,
    # Enhanced numerical parameters per feedback
    device: str | None = None,
    dtype: str | None = None,
    seed: int | None = None,
    weight_decay: float = 1e-6,
    use_scheduler: bool = True,
    warmstart_lstsq: bool = True,
    continuous_residual_weight: float = 1e-2,
    return_ct_params: bool = True,
):
    """Train a linear continuous-time NODE and return discretized dynamics.

    The model parameterizes continuous-time matrices :math:`(A_c, B_c)` and uses
    the matrix exponential of the block matrix to obtain discrete-time dynamics
    during training. This keeps the NODE interpretation intact while avoiding the
    heavy per-epoch ODE solves that made the original implementation slow.
    
    Parameters
    ----------
    Xtrain : np.ndarray
        Input state data, shape (n, T)
    Xp : np.ndarray
        Next state data, shape (n, T)
    Utrain : np.ndarray
        Input control data
    dt : float
        Time step
    epochs : int, default=300
        Maximum number of training epochs (increased from 200)
    lr : float, default=1e-2
        Learning rate for Adam optimizer
    verbose : bool, default=True
        Whether to print training progress
    log_every : int, default=10
        Frequency of progress logging
    return_history : bool, default=False
        Whether to return loss history (legacy parameter)
    return_diagnostics : bool, default=False
        Whether to return comprehensive training diagnostics
    patience : int, default=25
        Early stopping patience (epochs without improvement)
    min_delta : float, default=1e-6
        Minimum improvement threshold for early stopping
    convergence_tol : float, default=1e-5
        Loss threshold for convergence detection
    max_grad_norm : float, default=10.0
        Maximum gradient norm (for gradient clipping)
    early_stopping : bool, default=True
        Whether to use early stopping
    log_file : str, optional
        Path to log file for storing training progress. If None, no file logging.
    log_append : bool, default=True
        Whether to append to existing log file or overwrite
    device : str, optional
        Device to run on ('cpu', 'cuda'). If None, defaults to 'cpu'.
    dtype : str, optional
        Tensor data type ('float32', 'float64'). If None, defaults to 'float64'.
    seed : int, optional
        Random seed for reproducibility
    weight_decay : float, default=1e-6
        L2 regularization weight for Adam optimizer
    use_scheduler : bool, default=True
        Whether to use ReduceLROnPlateau scheduler
    warmstart_lstsq : bool, default=True
        Whether to initialize with discrete least-squares solution
    continuous_residual_weight : float, default=1e-2
        Weight for continuous residual loss term to improve small-dt conditioning
    return_ct_params : bool, default=True
        Whether to return continuous-time parameters in diagnostics
        
    Returns
    -------
    Ad : np.ndarray
        Discrete-time A matrix
    Bd : np.ndarray
        Discrete-time B matrix
    diagnostics : dict, optional
        Training diagnostics if return_diagnostics=True or return_history=True
    """
    import time
    
    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    n, T = Xtrain.shape
    U_cm = _ensure_channel_major(Utrain, T)
    m = U_cm.shape[0]

    # Device and dtype configuration (High-ROI improvement A)
    if device is None:
        device_obj = torch.device("cpu")
    else:
        device_obj = torch.device(device)
        
    if dtype is None:
        dtype_obj = torch.float64  # Default to float64 for better conditioning
    else:
        dtype_obj = getattr(torch, dtype)

    # Adaptive convergence tolerance based on data scale
    if convergence_tol == 1e-6:
        data_scale = float(np.mean(np.square(Xp)))
    else:
        data_scale = 1.0
    adaptive_convergence_tol = convergence_tol * max(data_scale, 1e-6)

    # Create tensors with consistent dtype and device (High-ROI improvement A)
    Xk = torch.tensor(Xtrain.T, dtype=dtype_obj, device=device_obj)  # (T, n)
    Uk = torch.tensor(U_cm.T, dtype=dtype_obj, device=device_obj)    # (T, m)
    Xnext = torch.tensor(Xp.T, dtype=dtype_obj, device=device_obj)   # (T, n)

    eye_n = torch.eye(n, dtype=dtype_obj, device=device_obj)

    # Initialize parameters with warmstart if requested (High-ROI improvement E)
    if warmstart_lstsq:
        # Discrete least-squares warm start: min||Z*Theta - X+||
        Z = torch.cat([Xk, Uk], dim=1)  # (T, n+m)
        try:
            Theta = torch.linalg.lstsq(Z, Xnext).solution  # (n+m, n)
            Ad0 = Theta[:n, :].T  # (n, n)
            Bd0 = Theta[n:, :].T  # (n, m)
            
            # Attempt continuous-time initialization via matrix log (High-ROI improvement F)
            try:
                I = torch.eye(n, dtype=dtype_obj, device=device_obj)
                # Safe matrix log with fallback for singular cases
                if torch.linalg.det(Ad0) > 1e-12:
                    A_ct_init = torch.matrix_log(Ad0) / dt
                    # Solve for B_ct from ZOH relation: (Ad - I) = A_ct * dt * Bd_dt/dt
                    # Bd = A_ct^-1 * (Ad - I) * B_ct, so B_ct = A_ct \ (Ad-I) \ Bd
                    if torch.linalg.det(A_ct_init) > 1e-12:
                        B_ct_init = torch.linalg.solve(A_ct_init, torch.linalg.solve(Ad0 - I, Bd0))
                    else:
                        # Fallback to scaled discrete init if A_ct is singular
                        A_ct_init = (Ad0 - I) / dt
                        B_ct_init = Bd0 / dt
                else:
                    # Fallback for singular Ad0
                    A_ct_init = torch.randn_like(Ad0) * 0.1
                    B_ct_init = torch.randn(n, m, dtype=dtype_obj, device=device_obj) * 0.1
            except Exception:
                # Fallback to simple discrete-time scaling 
                A_ct_init = (Ad0 - torch.eye(n, dtype=dtype_obj, device=device_obj)) / dt
                B_ct_init = Bd0 / dt
        except Exception:
            # Fallback to random initialization if lstsq fails
            A_ct_init = torch.randn((n, n), dtype=dtype_obj, device=device_obj) * 0.1
            B_ct_init = torch.randn((n, m), dtype=dtype_obj, device=device_obj) * 0.1
    else:
        # Random initialization
        A_ct_init = torch.randn((n, n), dtype=dtype_obj, device=device_obj) * 0.1
        B_ct_init = torch.randn((n, m), dtype=dtype_obj, device=device_obj) * 0.1
    
    A_ct = torch.nn.Parameter(A_ct_init)
    B_ct = torch.nn.Parameter(B_ct_init)
    params = [A_ct, B_ct]
    # Add weight decay for regularization (High-ROI improvement I)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    
    # Add learning rate scheduler (High-ROI improvement H)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, 
            threshold=1e-6, min_lr=lr*1e-3
        )

    # Enhanced training monitoring
    start_time = time.time()
    history: list[float] = []
    grad_norms: list[float] = []
    best_loss = float('inf')
    patience_counter = 0
    converged = False
    early_stopped = False
    
    # Initialize file logging
    log_file_handle = None
    if log_file is not None:
        import pathlib
        log_path = pathlib.Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        mode = 'a' if log_append else 'w'
        log_file_handle = open(log_path, mode, encoding='utf-8')
        
        # Write header for new log file
        if not log_append or log_path.stat().st_size == 0:
            log_file_handle.write("# NODE Training Log\n")
            log_file_handle.write(f"# Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file_handle.write(f"# Parameters: n={n}, m={m}, T={T}, dt={dt:.3f}\n")
            log_file_handle.write(f"# Hyperparameters: epochs={epochs}, lr={lr}, patience={patience}, tol={convergence_tol:.1e}\n")
            log_file_handle.write("epoch,mse_loss,grad_norm,best_loss,patience_counter,status\n")
        else:
            log_file_handle.write(f"\n# New session: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    def _compute_discrete_matrices() -> tuple[torch.Tensor, torch.Tensor]:
        """Efficient computation without block matrix exponential (High-ROI improvement B)"""
        # Ad = exp(A_ct * dt)
        Ad = torch.matrix_exp(A_ct * dt)
        
        # Bd = A_ct^-1 * (Ad - I) * B_ct (ZOH formula)
        bd_rhs = (Ad - eye_n) @ B_ct
        try:
            Bd = torch.linalg.solve(A_ct, bd_rhs)
        except RuntimeError:
            # Fallback with Tikhonov regularization if A_ct is singular
            Bd = torch.linalg.solve(A_ct + 1e-10 * eye_n, bd_rhs)
        
        return Ad, Bd

    for epoch in range(int(epochs)):
        optimizer.zero_grad()

        # Efficient discrete matrix computation (High-ROI improvement B)
        Ad, Bd = _compute_discrete_matrices()

        # Primary discrete-time prediction loss
        preds = Xk @ Ad.T + Uk @ Bd.T
        discrete_loss = torch.nn.functional.mse_loss(preds, Xnext)
        
        # Continuous residual term for better small-dt conditioning (High-ROI improvement G)
        continuous_residual = (Xnext - Xk) / dt - (Xk @ A_ct.T + Uk @ B_ct.T)
        continuous_loss = continuous_residual.pow(2).mean()
        
        # Combined loss
        loss = discrete_loss + continuous_residual_weight * continuous_loss
        
        # NaN/Inf guard (High-ROI improvement D)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss became non-finite at epoch {epoch}; check dtype/init/lr.")
        
        loss.backward()
        
        # Gradient monitoring and clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)
        grad_norms.append(float(grad_norm))
        
        optimizer.step()

        loss_value = float(loss.item())
        history.append(loss_value)
        
        # Update learning rate scheduler (High-ROI improvement H)
        if scheduler is not None:
            scheduler.step(loss_value)
        
        # Convergence detection with adaptive tolerance
        if loss_value < adaptive_convergence_tol:
            converged = True
            message = f"[NODE] ✓ Converged at epoch {epoch}: loss={loss_value:.6e} < tol={convergence_tol:.1e}"
            if verbose:
                print(message)
            if log_file_handle:
                log_file_handle.write(f"{epoch},{loss_value:.6e},{grad_norm:.6e},{best_loss:.6e},{patience_counter},CONVERGED\n")
                log_file_handle.write(f"# {message}\n")
                log_file_handle.flush()
            break
            
        # Early stopping logic
        if early_stopping:
            if loss_value < best_loss - min_delta:
                best_loss = loss_value
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                early_stopped = True
                message = f"[NODE] ⏹ Early stopping at epoch {epoch}: no improvement for {patience} epochs"
                if verbose:
                    print(message)
                if log_file_handle:
                    log_file_handle.write(f"{epoch},{loss_value:.6e},{grad_norm:.6e},{best_loss:.6e},{patience_counter},EARLY_STOPPED\n")
                    log_file_handle.write(f"# {message}\n")
                    log_file_handle.flush()
                break
        
        # Gradient explosion detection
        if grad_norm > max_grad_norm:
            if verbose:
                print(f"[NODE] ⚠️ High gradient norm at epoch {epoch}: {grad_norm:.3e}")
        
        # Progress logging
        if epoch % log_every == 0 or epoch == epochs - 1:
            improvement_info = ""
            status = "TRAINING"
            if epoch > 0:
                if early_stopping:
                    improvement_info = f" | best={best_loss:.6e} | patience={patience_counter}/{patience}"
            
            if verbose:
                print(f"[NODE] epoch {epoch:4d}  mse={loss_value:.6e}  grad_norm={grad_norm:.3e}{improvement_info}")
            
            # Log to file every log_every epochs
            if log_file_handle:
                log_file_handle.write(f"{epoch},{loss_value:.6e},{grad_norm:.6e},{best_loss:.6e},{patience_counter},{status}\n")
                if epoch % (log_every * 5) == 0:  # Flush periodically
                    log_file_handle.flush()

    training_time = time.time() - start_time
    epochs_actual = len(history)
    
    # Final logging and cleanup
    if log_file_handle:
        log_file_handle.write(f"# Training completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file_handle.write(f"# Final results: epochs={epochs_actual}, final_loss={history[-1] if history else 'N/A':.6e}, ")
        log_file_handle.write(f"best_loss={best_loss:.6e}, time={training_time:.2f}s\n")
        log_file_handle.write(f"# Status: {'CONVERGED' if converged else 'EARLY_STOPPED' if early_stopped else 'COMPLETED'}\n")
        log_file_handle.write("# " + "="*60 + "\n")
        log_file_handle.close()

    # Final parameter extraction (High-ROI improvement O)
    with torch.no_grad():
        Ad, Bd = _compute_discrete_matrices()
        Ad_np = Ad.cpu().numpy()
        Bd_np = Bd.cpu().numpy()
        
        # Extract continuous-time parameters
        A_ct_np = A_ct.detach().cpu().numpy()
        B_ct_np = B_ct.detach().cpu().numpy()
        
        # ZOH consistency check (High-ROI improvement O)
        I_np = np.eye(n)
        Ad_expected = scipy.linalg.expm(A_ct_np * dt)
        Bd_expected = np.linalg.solve(A_ct_np, (Ad_expected - I_np)) @ B_ct_np if np.linalg.det(A_ct_np) > 1e-12 else Bd_np
        
        zoh_consistency_Ad = np.linalg.norm(Ad_np - Ad_expected, 'fro')
        zoh_consistency_Bd = np.linalg.norm(Bd_np - Bd_expected, 'fro')
        zoh_consistency = zoh_consistency_Ad + zoh_consistency_Bd

    # Enhanced diagnostics with CT parameters and consistency checks
    if return_diagnostics or return_history:
        # Safe final loss extraction (High-ROI improvement P) 
        final_loss = float(history[-1]) if history else float("nan")
        
        diagnostics = {
            'loss_history': np.array(history),
            'grad_norm_history': np.array(grad_norms),
            'final_loss': final_loss,
            'best_loss': best_loss,
            'epochs_actual': epochs_actual,
            'epochs_requested': epochs,
            'converged': converged,
            'early_stopped': early_stopped,
            'training_time_s': training_time,
            'convergence_rate': -np.log(final_loss / history[0]) / epochs_actual if len(history) > 1 and history[0] > 0 and np.isfinite(final_loss) else 0.0,
            # ZOH consistency metrics (High-ROI improvement O)
            'zoh_consistency_fro': zoh_consistency,
            'zoh_consistency_Ad': zoh_consistency_Ad,  
            'zoh_consistency_Bd': zoh_consistency_Bd,
            'hyperparameters': {
                'lr': lr,
                'patience': patience,
                'min_delta': min_delta,
                'convergence_tol': convergence_tol,
                'adaptive_convergence_tol': adaptive_convergence_tol,
                'max_grad_norm': max_grad_norm,
                'early_stopping': early_stopping,
                'weight_decay': weight_decay,
                'continuous_residual_weight': continuous_residual_weight,
                'warmstart_lstsq': warmstart_lstsq,
                'use_scheduler': use_scheduler,
                'device': str(device_obj),
                'dtype': str(dtype_obj),
            }
        }
        
        # Add continuous-time parameters if requested (High-ROI improvement O)
        if return_ct_params:
            diagnostics['A_ct'] = A_ct_np
            diagnostics['B_ct'] = B_ct_np
            
        return Ad_np, Bd_np, diagnostics
    
    return Ad_np, Bd_np


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


def _demean_channels(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if arr.ndim == 1:
        arr = arr[:, None]
    mean = arr.mean(axis=0, keepdims=True)
    return arr - mean, mean.ravel()


def _block_hankel(time_series: np.ndarray, s: int) -> np.ndarray:
    """Return the block Hankel matrix with ``s`` block rows.

    Parameters
    ----------
    time_series : array_like, shape (T, d)
        Time-major sequence.
    s : int
        Number of block rows.
    """
    if s <= 0:
        raise ValueError("block size s must be positive")
    T, d = time_series.shape
    L = T - s + 1
    if L <= 0:
        raise ValueError("time series too short for requested block size")
    blocks = [time_series[j:j+L, :].T for j in range(s)]
    return np.vstack(blocks)


def _solve_ridge_normal(
    A: np.ndarray,
    B: np.ndarray,
    ridge: float = 0.0,
) -> np.ndarray:
    """Solve min_X ||AX - B||_F^2 + ridge ||X||_F^2."""
    if ridge > 0.0:
        AtA = A.T @ A + ridge * np.eye(A.shape[1], dtype=A.dtype)
        AtB = A.T @ B
        return np.linalg.solve(AtA, AtB)
    sol, *_ = np.linalg.lstsq(A, B, rcond=None)
    return sol


def moesp_fit(
    u_ts: np.ndarray,
    y_ts: np.ndarray,
    s: int,
    n: Optional[int] = None,
    svd_tol: Optional[float] = None,
    projector_rcond: float = 1e-12,
    ridge: float = 0.0,
    ls_rcond: Optional[float] = None,
    return_states: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Faithful MOESP implementation following Van Overschee & De Moor.

    Parameters
    ----------
    u_ts : array_like, shape (T, m)
        Input sequence (time-major).
    y_ts : array_like, shape (T, ell)
        Output sequence (time-major).
    s : int
        Past horizon / number of block rows (must be >= model order).
    n : int, optional
        System order. If ``None``, it is inferred from the singular values
        using ``svd_tol``.
    svd_tol : float, optional
        Threshold for selecting the model order when ``n`` is ``None``. When
        omitted, a relative tolerance of ``1e-12`` is used.
    projector_rcond : float, default=1e-12
        Regularization for the pseudo-inverse inside the oblique projection.
    ridge : float, default=0.0
        Ridge parameter used when solving the shift-invariance equation for ``A``.
    ls_rcond : float, optional
        Cutoff for the least-squares solves when regressing ``B`` and ``D``.
    return_states : bool, default=False
        If ``True``, the recovered conditional state sequence is included in the
        ``info`` dictionary under the ``"state_sequence"`` key.
    """

    u_ts = np.asarray(u_ts, dtype=float)
    y_ts = np.asarray(y_ts, dtype=float)
    if u_ts.ndim == 1:
        u_ts = u_ts[:, None]
    if y_ts.ndim == 1:
        y_ts = y_ts[:, None]

    if u_ts.shape[0] != y_ts.shape[0]:
        raise ValueError("u_ts and y_ts must share the same time dimension")

    T = u_ts.shape[0]
    m = u_ts.shape[1]
    ell = y_ts.shape[1]
    if s < 1:
        raise ValueError("s must be a positive integer")
    if T < 2 * s:
        raise ValueError("time series too short relative to chosen horizon")

    u_dm, u_mean = _demean_channels(u_ts)
    y_dm, y_mean = _demean_channels(y_ts)

    L = T - 2 * s + 1
    if L <= 0:
        raise ValueError("insufficient samples for the requested horizon")

    Up = _block_hankel(u_dm[: T - s, :], s)
    Uf = _block_hankel(u_dm[s:, :], s)
    Yf = _block_hankel(y_dm[s:, :], s)

    # Left annihilator of U_f via null space of U_f
    null_basis = scipy.linalg.null_space(Uf, rcond=projector_rcond)
    if null_basis.size == 0:
        raise ValueError("input Hankel has full column rank; cannot form annihilator")
    Q2 = null_basis
    Lf = Q2.T
    rank_Uf = Uf.shape[1] - Q2.shape[1]

    Yf_t = Yf @ Q2
    Up_t = Up @ Q2
    # Orthogonal projection of Y_f onto row(U_p)
    G = Up_t @ Up_t.T
    G_pinv = np.linalg.pinv(G, rcond=projector_rcond)
    Ob = Yf_t @ Up_t.T @ G_pinv @ Up_t

    U_svd, S_svd, Vh_svd = np.linalg.svd(Ob, full_matrices=False)
    if n is None:
        if S_svd.size == 0:
            raise ValueError("no singular values available to infer the order")
        if svd_tol is None:
            svd_tol = max(1e-12, S_svd[0] * 1e-12)
        n = int(np.sum(S_svd > svd_tol))
    if n <= 0:
        raise ValueError("model order must be positive")
    if n > S_svd.size:
        raise ValueError("requested order exceeds the rank of the projection")

    Sigma_sqrt = np.sqrt(S_svd[:n])
    Gamma = U_svd[:, :n] * Sigma_sqrt[np.newaxis, :]
    Xfp = (Sigma_sqrt[:, np.newaxis] * Vh_svd[:n, :])

    # Extract C and A using shift-invariance
    C = Gamma[:ell, :]
    Gamma_up = Gamma[: (s - 1) * ell, :]
    Gamma_dn = Gamma[ell:, :]
    if Gamma_up.size == 0 or Gamma_dn.size == 0:
        raise ValueError("horizon s too small to extract system matrices")
    A = _solve_ridge_normal(Gamma_up, Gamma_dn, ridge=ridge)

    # Recover conditional states (aligned with columns of Xfp)
    X_cond = Xfp

    # Regress D first using the aligned subset of samples
    idx = np.arange(s, s + X_cond.shape[1])
    if idx[-1] >= T:
        idx = idx[idx < T]
        X_cond = X_cond[:, : idx.size]
    U_reg = u_dm[idx, :]
    Y_reg = y_dm[idx, :]

    # y_k ≈ C x_k + D u_k
    residual_y = (Y_reg - (X_cond.T @ C.T))
    D_t, *_ = np.linalg.lstsq(U_reg, residual_y, rcond=ls_rcond)
    D = D_t.T

    # x_{k+1} ≈ A x_k + B u_k
    if X_cond.shape[1] < 2:
        raise ValueError("not enough aligned states to regress B")
    Xk = X_cond[:, :-1]
    Xk1 = X_cond[:, 1:]
    U_state = u_dm[idx[:-1], :]
    rhs = (Xk1 - A @ Xk).T
    B_t, *_ = np.linalg.lstsq(U_state, rhs, rcond=ls_rcond)
    B = B_t.T

    info: Dict[str, Any] = {
        "s": s,
        "n": n,
        "singular_values": S_svd,
        "rank_Uf": rank_Uf,
        "u_mean": u_mean,
        "y_mean": y_mean,
        "Lf": Lf,
    }
    if return_states:
        info["state_sequence"] = X_cond

    return A, B, C, D, info

# Backward-compat wrapper: same call sites as old code/tests
def moesp_fit_old(
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