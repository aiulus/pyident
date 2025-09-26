"""
Back-compat shim for experiment sweeps.
"""


# Install dotted aliases: pyident.experiments.PEorder -> pyident._experiments
import sys as _sys
import pyident._experiments as _exp

for _name in ("PEorder", "initstate", "sparsity", "underactuation",
              "exp_ctrl_prbs", "exp_x0_unctrl"):
    _sys.modules[__name__ + "." + _name] = _exp

del _sys, _exp  # keep module clean