# pyident/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, Dict, Any
import numpy as np

@dataclass
class SolverOpts:
    maxit: int = 150
    tol_grad: float = 1e-8
    random_state: Optional[int] = 0

@dataclass
class ExpConfig:
    n: int = 10
    m: int = 3
    T: int = 400 # time horizon (number of steps)
    dt: float = 0.02

    ensemble: Literal["ginibre", "sparse", "stable", "binary"] = "ginibre"

    # --- Sparse-continuous ensemble options ---
    p_density: float = 0.8                      
    sparse_which: Literal["A", "B", "both"] = "both" # which matrix to sparsify
    p_density_A: Optional[float] = None         
    p_density_B: Optional[float] = None        
         
    x0_mode: Literal["gaussian", "rademacher", "ones", "zero"] = "gaussian"

    # --- Input signal options ---
    signal: Literal["prbs", "multisine"] = "prbs"
    pe_order_target: int = 12            
    U_restr: Optional[np.ndarray] = None  # Pointwise input constraints - generator matrix
    PE_r: Optional[int] = None # PE (nonlocal) constraints - order of excitation

    # --- Identification algorithms --- 
    estimators: Sequence[str] = ("dmdc", "moesp") 

    light: bool = True # leightweight io toggle                  

@dataclass
class RunMeta:
    seed: int
    version: str = "0.1.0"
    extra: Dict[str, Any] = field(default_factory=dict)
