##### Ovewview
```
pyident/
  __init__.py
  config.py             # dataclasses for SolverOpts, ExpConfig, RunMeta
  ensembles.py          # random matrix factories (ginibre, sparse, stable, underactuated…)
  signals.py            # PE input generators + PE order tests
  metrics.py            # metrics (PBH, Gramian, Krylov, angles)
  estimators/           # identification algorithms (plugins)
    __init__.py
    dmdc.py
    moesp.py            #  simple MOESP/N4SID
    gradient_based.py
  experiments/
    initstate.py
    PEorder.py
    sparsity.py
    underactuation.py
  loggers/              # Functions that help produce detailed traceback files
  io_utils.py           # JSON/CSV/NPZ writers, “light” toggle, version capture
  plots.py              # sigma_min contours, rank bars, histograms, PGF export
  run_single.py         # run_single(cfg, seed, sopts, estimators=[...])
  cli.py                # argparse entry point
  jax_accel.py
  sys_utils.py
```
##### Examples
###### 1) Single run:
```
python -m ..cli single --n 6 --m 2 --T 120 --dt 0.05 \
  --ensemble ginibre --signal prbs --sigPE 12 \
  --algs dmdc --seed 0 \
  --out-json runs/single_dense.json
```
###### 2) Sparsity sweep:
```
python -m pyident.cli sweep-sparsity \
  --n 6 --m 2 --T 80 --dt 0.05 \
  --signal prbs --sigPE 12 \
  --p-values 1.0,0.8,0.6,0.4,0.2 \
  --seeds 0:20 \
  --algs dmdc,moesp \
  --out-csv runs/sparsity/results.csv \
  --plots --plot-dir runs/sparsity/plots
```

###### 3) Underactuation sweep
```
python -m pyident.cli sweep-underactuation \
  --n 6 --T 80 --dt 0.05 \
  --ensemble ginibre --signal prbs --sigPE 12 \
  --m-values 1,2,3,4 \
  --seeds 0:20 \
  --algs dmdc,moesp \
  --out-csv runs/underact/results.csv \
  --plots --plot-dir runs/underact/plots
```
