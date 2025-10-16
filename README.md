##### Overview
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

### Experiments
#### Single system score-REE correlation (§5.1.1)
```
python -m pyident.experiments.sim_scoree \
 --outdir multicol_simcor_3xfinal \
--n 10 --m 10 --zoom --deficiency 1 --T 100 
```
#### dim(V(x0))-stratified score-REE correlation over multiple systems (§5.1.1)
```
python -m pyident.experiments.sim_mse --zoom
--ensvol 200 --x0count 200 \
--T 100 --n 5 --m 5 --dt 0.01
```
#### Equivalence class membership tests (§5.1.2)
```
python -m pyident.experiments.sim_escon --single --det
```
#### Single-varying-axis parameter regime sweeps
##### S1: Sparsity (§5.2.1)
```
python -m pyident.experiments.sim_regcomb --axes "sparsity" \
    --sparsity-grid 0.0:0.1:1.0 --samples 100 \
    --x0-samples 100 --outdir results/sim3_sparse
```

##### S2: State dimension (§5.2.2)
```
python -m pyident.experiments.sim_regcomb --axes "ndim" \
    --ndim-grid 2:2:20 --samples 100 \
    --x0-samples 100 --outdir results/sim3_state
```
##### S3: Underactuation (§5.2.3)
```
python -m pyident.experiments.sim_regcomb --axes "underactuation" \
     --samples 100 --x0-samples 100 --outdir results/sim3_underactuation
```
#### Double-varying-axis parameter regime sweeps
##### D1: Sparsity vs. state dimension (§5.2.4)
```
python -m pyident.experiments.sim_regcomb --axes "sparsity, ndim" \
    --sparsity-grid 0.0:0.1:1.0 --ndim-grid 2:2:20 --samples 100 \
    --x0-samples 100 --outdir results/sim3_sparse_state
```
##### D2: State dimension vs. underactuation (§5.2.5)
```
python -m pyident.experiments.sim_regcomb --axes "ndim, underactuation" \
    --ndim-grid 2:2:20 --samples 100 \
    --x0-samples 100 --outdir results/sim3_state_underactuation
```
##### D3: Underaction vs. sparsity (§5.2.6)
```
python -m pyident.experiments.sim_regcomb --axes "underactuation, sparsity" \
        --sparsity-grid 0.0:0.1:1.0 --samples 100 \
        --x0-samples 100 --outdir results/sim3_underactuation_sparsity
```
#### Input sufficiency (§5.3)
```
 python -m pyident.experiments.simpe --target-visible 3 --target-visible 8 --n 8
```
