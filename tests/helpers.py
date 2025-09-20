import numpy as np
from statistics import median
from typing import Dict, Any, Sequence
from ..run_single import run_single
from ..config import ExpConfig, SolverOpts

def run_many(cfg: ExpConfig, seeds: Sequence[int], sopts: SolverOpts,
             algs=("dmdc",), use_jax=False):
    outs = []
    for sd in seeds:
        out = run_single(cfg, seed=sd, sopts=sopts, algs=algs, use_jax=use_jax)
        outs.append(out)
    return outs

def med(outs, key):
    vals = [o[key] for o in outs if o.get(key) is not None]
    return median(vals) if vals else None

def assert_monotone_nondec(xs, slack=0.0):
    # allows tiny dips within slack; robust for stochastic medians
    last = xs[0]
    for i,x in enumerate(xs[1:], start=1):
        assert x + slack >= last, f"sequence decreases at {i}: {xs}"
        last = max(last, x)
