# pyident/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, Dict, Any
import numpy as np


@dataclass
class SolverOpts:
    """Generic nonlinear solver options."""
    maxit: int = 150
    tol_grad: float = 1e-10
    random_state: Optional[int] = 0

    rcond: float = 1e-10

    # gradient estimator defaults 
    gd_steps: int = 200
    gd_lr: Optional[float] = None
    gd_opt: str = "adam"            # "adam" or "sgd"
    gd_ridge: Optional[float] = None
    gd_project: Optional[str] = None  # None | "ct" | "dt"
    gd_proj_params: Optional[Dict[str, float]] = None


@dataclass
class ExpConfig:
    """Base configuration for an identification experiment."""
    n: int = 20
    m: int = 10
    T: int = 200
    dt: float = 0.05
    seed: int = 42  

    ensemble: str = 'A_stbl_B_ctrb'

    # --- Sparse–continuous ensemble options ---
    p_density: float = 0.8                      
    sparse_which: Literal["A", "B", "both"] = "both"
    p_density_A: Optional[float] = None          
    p_density_B: Optional[float] = None         

    x0_mode: Literal["gaussian", "rademacher", "ones", "zero"] = "gaussian"

    # --- Input signal options ---
    signal: Literal["prbs", "multisine"] = "prbs"
    sigPE: int = 12 # desired PE order for the control input                   
    U_restr: Optional[np.ndarray] = None # pointwise constraint basis W \in R^{m×q}
    PE_r: Optional[int] = None                   # moment-PE (nonlocal) constraint

    # --- Identification algorithms ---
    moesp_s: Optional[int] = None

    light: bool = True

    # Derived/validated fields (filled in __post_init__)
    _density_A: float = field(init=False, repr=False)
    _density_B: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Validate densities & set effective A/B densities
        def _pick(name: str, val: Optional[float], fallback: float) -> float:
            x = fallback if val is None else float(val)
            if not (0.0 <= x <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {x}.")
            return x

        self._density_A = _pick("p_density_A", self.p_density_A, self.p_density)
        self._density_B = _pick("p_density_B", self.p_density_B, self.p_density)

        # Validate U_restr shape if provided
        if self.U_restr is not None:
            if self.U_restr.ndim != 2 or self.U_restr.shape[0] != self.m:
                raise ValueError(
                    f"U_restr must have shape (m, q) with first dim == m={self.m}, got {self.U_restr.shape}."
                )

        # Validate PE order target if used
        if self.PE_r is not None and self.PE_r <= 0:
            raise ValueError("PE_r must be a positive integer if provided.")
        
        if self.moesp_s is not None and self.moesp_s <= 0:
            raise ValueError("moesp_s must be positive when provided.")


@dataclass
class RunMeta:
    seed: int
    version: str = "0.1.0"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig(ExpConfig):
    """Extended configuration for identification experiments."""
    q_filter: float = 0.7  # quantile threshold for filtering
    dwell: int = 1        # PRBS dwell time
    u_scale: float = 5.0  # input scaling
    noise_std: float = 0.0  # measurement noise
    n_trials: int = 200   # number of trials
    T: int = 200         # trajectory length
    
    def __post_init__(self):
        super().__post_init__()
        if not hasattr(self, 'seed'):
            self.seed = 42
