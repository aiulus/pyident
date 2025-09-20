import csv
from pathlib import Path
import numpy as np
import pytest

def _count_rows(csv_path: Path) -> int:
    with csv_path.open() as f:
        return sum(1 for _ in f) - 1  # minus header

def test_sparsity_sweep_smoke(tmp_path):
    from ..experiments.sparsity import sweep_sparsity
    out = tmp_path / "sparsity.csv"
    sweep_sparsity(
        n=4, m=2, p_values=(0.8, 0.4), T=60, dt=0.05,
        sparse_which="both", sigPE=12, seeds=range(2),
        out_csv=str(out), use_jax=False, x0_mode="gaussian", U_restr_dim=None,
        algs=("dmdc",)
    )
    assert out.exists() and _count_rows(out) >= 2

def test_underactuation_sweep_smoke(tmp_path):
    from ..experiments.underactuation import sweep_underactuation
    out = tmp_path / "underact.csv"
    sweep_underactuation(
        n=4, m_values=(1, 2), T=60, dt=0.05, sigPE=12, seeds=range(2),
        out_csv=str(out), use_jax=False, x0_mode="gaussian",
        algs=("dmdc",)
    )
    assert out.exists() and _count_rows(out) >= 2

def test_peorder_controllable_smoke(tmp_path):
    from ..experiments.PEorder import sweep_pe_order_controllable
    out = tmp_path / "pe_cont.csv"
    sweep_pe_order_controllable(
        n=4, m=2, T=60, dt=0.05, sigPE=12,
        r_values=(1, 2), seeds=range(2), out_csv=str(out), algs=("dmdc",)
    )
    assert out.exists() and _count_rows(out) >= 2

def test_peorder_uncontrollable_smoke(tmp_path):
    from ..experiments.PEorder import sweep_pe_order_uncontrollable
    out = tmp_path / "pe_uncont.csv"
    sweep_pe_order_uncontrollable(
        n=4, m=2, T=60, dt=0.05, sigPE=12,
        r_values=(1, 2), seeds=range(2), out_csv=str(out), algs=("dmdc",)
    )
    assert out.exists() and _count_rows(out) >= 2

def test_initstate_sweep_smoke(tmp_path):
    from ..experiments.initstate import sweep_initial_states
    from ..config import ExpConfig
    cfg = ExpConfig(n=4, m=2, T=60, dt=0.05, ensemble="ginibre", signal="prbs", sigPE=12, light=True)
    out = tmp_path / "x0.csv"
    sweep_initial_states(cfg=cfg, seed=0, n_x0=32, x0_mode="gaussian",
                         analysis_mode=None, out_csv=str(out), run_estimators=False)
    assert out.exists() and _count_rows(out) >= 1
