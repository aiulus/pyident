import os, json, csv, sys
from pathlib import Path
import argparse, logging
from identifex.config import SolverOpts, ExpConfig, RunMeta
from identifex.run_one import run_one
from identifex.io_utils import save_results, init_summary_csv

def parse_args():
    p = argparse.ArgumentParser(description="Identifex sweeps")
    # instance shape
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--ensemble", type=str, default="ginibre",
                  choices=["ginibre","binary","sparse","stable","brunovsky"])
    p.add_argument("--density", type=float, default=0.1)       # sparse
    p.add_argument("--scale", type=float, default=1.0)
    # experiment
    p.add_argument("--num-instances", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="runs")
    p.add_argument("--light", action="store_true")
    p.add_argument("--tags", type=str, default="", help="comma tags")
    # solver opts
    p.add_argument("--num-seeds", type=int, default=16)
    p.add_argument("--maxit", type=int, default=200)
    p.add_argument("--tol-grad", type=float, default=1e-8)
    # PE + simulation (if you use estimators)
    p.add_argument("--T", type=float, default=5.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--u", type=str, default="prbs", choices=["prbs","multisine","white"])
    p.add_argument("--pe-order", type=int, default=10)
    # plotting / latex
    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--save-pgf", action="store_true")
    p.add_argument("--contour-grid", type=int, default=80)
    # estimators
    p.add_argument("--estimators", type=str, default="dmdc", help="comma list")
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    offset = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    base_seed = args.seed + offset

    cfg = ExpConfig(n=args.n, m=args.m, ensemble=args.ensemble, density=args.density,
                    scale=args.scale, horizon_t=args.T, dt=args.dt, u_type=args.u,
                    pe_order=args.pe_order, contour_grid=args.contour_grid,
                    save_plots=args.save_plots, save_pgf=args.save_pgf,
                    tags=args.tags.split(",") if args.tags else [])
    sopts = SolverOpts(maxit=args.maxit, tol_grad=args.tol_grad,
                       num_seeds=args.num_seeds, random_state=None)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    summary_csv = outdir / "summary.csv"
    if not summary_csv.exists():
        init_summary_csv(summary_csv)

    est_list = [e.strip() for e in args.estimators.split(",") if e.strip()]

    for k in range(args.num_instances):
        seed_k = base_seed + k
        res = run_one(cfg, seed_k, sopts, estimators=est_list)
        tag = f"{cfg.ensemble}_n{cfg.n}_m{cfg.m}_seed{seed_k}"
        save_results(res, outdir, tag, light=args.light)
        logging.info("%s: delta_fix=%.4e, bounds=(%.3e, %.3e, %.3e), " \
                    "rankK=%d, sigma_minK=%.3e, eta_0=%.3f, mu_min=%.3e",
                     tag, res['delta_fix'],
                     res['bounds']['lower'], res['bounds']['imag'], res['bounds']['upper'],
                     res['krylov']['rank'], res['krylov']['sigma_min'],
                     res['angles']['eta0'], res['modewise']['mu_min'])

if __name__ == "__main__":
    main()
