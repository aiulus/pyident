import json
from ..config import ExpConfig, SolverOpts
from ..run_single import run_single

def sweep(sigPE_list=(4,8,16,31,63), seed0=0, out="out/ctrl_prbs.json"):
    rows=[]
    for i,spe in enumerate(sigPE_list):
        cfg = ExpConfig(n=8, m=3, T=800, dt=0.05, ensemble="ginibre",
                        signal="prbs", sigPE=int(spe), light=True)
        outi = run_single(cfg, seed=seed0+i, sopts=SolverOpts(), algs=("dmdc","moesp"), use_jax=False)
        rows.append({"sigPE": int(spe), **{k:outi[k] for k in ("K_rank","delta_pbh","estimators")}})
    import os; os.makedirs("out", exist_ok=True)
    with open(out,"w") as f: json.dump({"rows":rows}, f, indent=2)

if __name__ == "__main__":
    sweep()
