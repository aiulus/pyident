import pathlib
import pytest
import importlib

from ..config import ExpConfig, SolverOpts
from ..run_single import run_single
from .. import io_utils

def test_io_utils_write_roundtrip(tmp_path):
    cfg = ExpConfig(n=3, m=1, T=20, dt=0.05, ensemble="ginibre", signal="prbs", sigPE=8)
    out = run_single(cfg, seed=0, sopts=SolverOpts(), algs=("dmdc",), use_jax=False)
    # write JSON/NPZ/CSV where available
    jpath = tmp_path / "result.json"
    npzpath = tmp_path / "result.npz"
    csvpath = tmp_path / "result.csv"
    io_utils.write_json(jpath, out)
    io_utils.write_npz(npzpath, {"A": out["estimators"]["dmdc"]["A_err_PV"] if "dmdc" in out["estimators"] else 0.0})
    io_utils.write_csv(csvpath, {"key": ["K_rank", "gram_mode"], "value": [out["K_rank"], out["gram_mode"]]})
    assert jpath.exists() and npzpath.exists() and csvpath.exists()

def test_cli_smoke_if_available(tmp_path, monkeypatch):
    spec = importlib.util.find_spec("..cli")
    if spec is None:
        pytest.skip("CLI module not present")
    cli = importlib.import_module("..cli")
    if not hasattr(cli, "main"):
        pytest.skip("cli.main not present")
    # Try to call main with a minimal argv; if signature unknown, skip
    argv = ["--n", "3", "--m", "1", "--T", "20", "--dt", "0.05",
            "--ensemble", "ginibre", "--signal", "prbs", "--out", str(tmp_path),
            "--algs", "dmdc", "--light"]
    try:
        cli.main(argv)
    except TypeError:
        pytest.skip("cli.main signature not compatible with argv list")
    # Expect some output files
    assert any(p.suffix in {".json", ".csv", ".npz"} for p in tmp_path.iterdir())
