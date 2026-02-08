# PyIdent ICLR Archive

This is a minimal, standalone subset of the original repository required to run:

```
python -m pyident.iclr \
  --base-outdir pyident_results/iclr \
  --ensemble-dir fresh_ensemble \
  --x0-dir fresh_ABx0 \
  --pbh-dir fresh_nonidentifiable_ABx0 \
  --sparsity-grid 0.0:0.1:1.0 \
  --ndim-grid 2:1:10 \
  --samples 10000 \
  --density-min 0.3 --density-max 0.7 --density-source AB \
  --x0-samples 10 --mask-ps 0.25 0.5 0.75 --outlier-trim 0.05 \
  --mask-ps-pbh 0.25 0.5 0.75 \
  --pbh-threshold 1e-6 \
  --seed 12345 --T 100 --dt 1.0 --u-scale 3.0 --dwell 1 \
  --algos DMDc
```

Notes:
- Only the `DMDc` estimator is supported in this archive subset.
- Heavy dependencies (`torch`, `pysindy`, `jax`) are removed.
