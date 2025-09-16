from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class SolverOpts:
    max_iter: int = 200
    tol_grad: float = 1e-8
    tol_f: float = 1e-10
    num_seeds: int = 16
    grid_omega: int = 512
    omega_max_factor: float = 2.0
    random_state: Optional[int] = 0

@dataclass
class ExperimentConfig:
    n: int = 5
    m: int = 5
    ensemble: str = "ginibre"      
    density: float = 0.5           
    scale: float = 1.0
    horizon_t: float = 5.0
    dt: float = 0.01
    u_type: str = "prbs"              
    pe_order: int = 10 # Better make it a function of n, m
    contour_grid: int = 80
    save_plots: bool = False
    save_pgf: bool = False
    tags: List[str] = field(default_factory=list)

@dataclass
class RunMeta:
    seed: int = 0
    n: int = 0
    m: int = 0
    ensemble: str = ""
    density: float = 1.0