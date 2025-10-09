from __future__ import annotations
from typing import *

# Remove outdated imports
# from .discarded.PEorder import (
#     sweep_pe_order_controllable,
#     sweep_pe_order_uncontrollable,
# )

# Current experiment collection
from .experiments.sim_eqv_cl import run_experiment as run_eqvcl
from .experiments.sim_eqvcl_sweep import run_experiment as run_eqvcl_sweep
from .experiments.sim_undera import run_experiment as run_undera
from .experiments.sim_sparse import run_experiment as run_sparse
from .experiments.F1_x0_boxplot import run_experiment as run_x0_filter

__all__ = [
    # Core experiments
    'run_eqvcl',          # Single equivalence class experiment
    'run_eqvcl_sweep',    # Parameter sweep for equivalence classes
    'run_undera',         # Underactuated system experiments
    'run_sparse',         # Sparse system identification
    'run_x0_filter',      # Initial state filtering analysis
]

# Experiment groupings for easier access
IDENTIFICATION_EXPERIMENTS = {
    'eqvcl': run_eqvcl,
    'sparse': run_sparse,
    'undera': run_undera,
}

ANALYSIS_EXPERIMENTS = {
    'eqvcl_sweep': run_eqvcl_sweep,
    'x0_filter': run_x0_filter,
}
