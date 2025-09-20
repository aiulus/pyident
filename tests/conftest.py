import os
import numpy as np
import pytest

from ..config import ExpConfig, SolverOpts

def _cfg_base(**kw):
    # Small, fast defaults; override per-test
    c = ExpConfig(
        n=4, m=2, T=60, dt=0.05,
        ensemble="ginibre",
        signal="prbs",
        sigPE=12,
    )
    for k,v in kw.items():
        setattr(c, k, v)
    return c

@pytest.fixture(scope="session")
def seeds():
    # small seed set keeps runtime short; bump as needed
    return list(range(8))

@pytest.fixture(scope="session")
def sopts():
    return SolverOpts()

@pytest.fixture(scope="session")
def has_jax():
    try:
        import jax  # noqa
        return True
    except Exception:
        return False

@pytest.fixture
def cfg():
    return _cfg_base()
