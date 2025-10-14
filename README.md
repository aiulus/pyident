##### Ovewview
## PyIdent

PyIdent is a small research toolkit for experimenting with identifiability in
linear dynamical systems.  It provides random system generators, persistently
exciting signal builders, metrics, and a collection of estimator implementations
that can be combined to produce reproducible experiments and sweeps.

```
pyident/
  __init__.py
  cli.py                # argparse entry point
  config.py             # dataclasses for SolverOpts, ExpConfig, …
  ensembles.py          # random matrix factories (ginibre, sparse, stable, …)
  estimators/           # identification algorithms (DMDc, MOESP, …)
  experiments/          # experiment entry points
  io_utils.py           # JSON/CSV/NPZ writers
  jax_accel.py          # optional JAX accelerators & precision toggles
  loggers/              # helpers for detailed tracebacks during long sweeps
  metrics.py            # PBH, Gramian, Krylov and other identifiability metrics
  projectors.py         # subspace projectors
  run_single.py         # programmatic entry point for a single experiment
  signals.py            # PE input generators + PE order tests
  simulation.py         # shared trajectory simulation routines
```
---

### Installation

The project targets Python 3.10+.  Install the core package with pip:

```bash
pip install -e .
```

JAX acceleration is optional; the CLI will enable it when `jax` is installed and
`--use-jax` is passed.

---
