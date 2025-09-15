import json, csv, platform, subprocess
from pathlib import Path
import numpy as np
from datetime import datetime, timezone

def versions():
    import numpy, scipy
    v = dict(python=platform.python_version(),
             numpy=numpy.__version__, scipy=scipy.__version__)
    try:
        import jax; v["jax"]=jax.__version__
    except Exception: pass
    try:
        v["git_commit"]=subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip()
    except Exception:
        v["git_commit"]=None
    v["datetime"]=datetime.now(timezone.utc).isoformat()
    return v

def init_summary_csv(csv_path):
    cols = ["tag","seed","n","m","ensemble","density",
            "delta_fix","lower","imag","upper",
            "krylov_rank","krylov_sigma_min","eta0","theta0","mu_min"]
    with open(csv_path,"w",newline="") as f:
        csv.DictWriter(f, fieldnames=cols).writeheader()

def save_results(res, outdir: Path, tag: str, light: bool):
    meta = res.get("meta", {})
    meta |= versions()
    meta["arrays_npz"] = None

    # arrays
    if not light:
        npz = outdir / f"{tag}.npz"
        np.savez_compressed(npz, A=res["A"], B=res["B"], x0=res["x0"],
                            K=res["krylov"]["K"], W=res["gramian"]["W"])
        meta["arrays_npz"] = npz.name

    # json
    payload = {k:v for k,v in res.items() if k not in ("A","B","x0","krylov","gramian")}
    payload["meta"] = meta
    with open(outdir / f"{tag}.json","w") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    # csv
    row = dict(tag=tag, seed=meta.get("seed"),
               n=meta.get("n"), m=meta.get("m"),
               ensemble=meta.get("ensemble"), density=meta.get("density"),
               delta_fix=res["delta_fix"],
               lower=res["bounds"]["lower"], imag=res["bounds"]["imag"], upper=res["bounds"]["upper"],
               krylov_rank=res["krylov"]["rank"], krylov_sigma_min=res["krylov"]["sigma_min"],
               eta0=res["angles"]["eta0"], theta0=res["angles"]["theta0"], mu_min=res["modewise"]["mu_min"])
    with open(outdir / "summary.csv","a",newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys()); w.writerow(row)

def _json_default(o):
    import numpy as np
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, complex): return [o.real, o.imag]
    return str(o)
