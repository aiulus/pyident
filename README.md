pyident/
  __init__.py
  config.py             # dataclasses for SolverOpts, ExpConfig, RunMeta
  ensembles.py          # random matrix factories (ginibre, sparse, stable, underactuated…)
  signals.py            # PE input generators + PE order tests
  metrics.py            # metrics (PBH, Gramian, Krylov, angles)
  pbh.py                # PBH-based metrics
  estimators/           # identification algorithms (plugins)
    __init__.py
    dmdc.py
    moesp.py            #  simple MOESP/N4SID
  io_utils.py           # JSON/CSV/NPZ writers, “light” toggle, version capture
  plots.py              # sigma_min contours, rank bars, histograms, PGF export
  run_single.py         # run_single(cfg, seed, sopts, estimators=[...])
  cli.py                # argparse entry point
  jax_accelerator.py          
