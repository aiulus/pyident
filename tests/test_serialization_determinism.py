import json
from pathlib import Path
from ..config import ExpConfig, SolverOpts
from ..run_single import run_single
from ..io_utils import save_json

def test_save_json_determinism_and_light_toggle(tmp_path):
    cfg = ExpConfig(n=4, m=2, T=60, dt=0.05, ensemble="ginibre", signal="prbs", sigPE=12)
    out1 = run_single(cfg, seed=42, sopts=SolverOpts(), algs=("dmdc",), use_jax=False, light=True)
    out2 = run_single(cfg, seed=42, sopts=SolverOpts(), algs=("dmdc",), use_jax=False, light=False)

    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    save_json(out1, p1)
    save_json(out2, p2)

    j1 = json.loads(p1.read_text())
    j2 = json.loads(p2.read_text())

    # Essential scientific keys exist and match across light toggles
    keys = ["K_rank", "gram_mode", "delta_pbh", "n", "m", "T", "dt"]
    for k in keys:
        assert k in j1 and k in j2
        assert j1[k] == j2[k]
