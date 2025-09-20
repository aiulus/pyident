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

def assert_monotone_nondec(xs, slack=0.0, rel_slack=0.0, allow_dips=0):
    """
    Assert xs is nondecreasing up to (slack + rel_slack*prev) tolerance.
    allow_dips: number of tolerated true decreases beyond tolerance.
    """
    dips = 0
    last = xs[0]
    for i, x in enumerate(xs[1:], start=1):
        tol = slack + rel_slack * abs(last)
        if x + tol < last:
            dips += 1
            if dips > allow_dips:
                raise AssertionError(f"sequence decreases at {i}: {xs}")
        last = max(last, x)

