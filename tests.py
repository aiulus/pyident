import pandas as pd, numpy as np
import os

# ---- paths (adjust if you write elsewhere) ----
P_U_SMALL   = "runs/underact_small.csv"      # old regime (x0 != 0, dense B)
P_U_X0ZERO  = "runs/underact_x0zero.csv"     # new regime (x0 = 0)
P_U_SPARSEB = "runs/underact_sparseB.csv"    # new regime (sparse B)
P_S_SMALL   = "runs/sparsity_small.csv"
P_S_WIDE    = "runs/sparsity_wide.csv"

def spearman_manual(x, y):
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0,1])

def header(msg):
    print("\n" + msg)
    print("-" * len(msg))

# ---------- Underactuation (old file) ----------
if os.path.exists(P_U_SMALL):
    df = pd.read_csv(P_U_SMALL)
    ok = sum((g.sort_values("m")["K_rank"].diff().fillna(0) >= -1e-9).all()
             for _, g in df.groupby("seed"))
    tot = df["seed"].nunique()
    print(f"Monotone K_rank across m (by seed): {ok}/{tot} = {ok/tot:.1%}")

    g = df.groupby("m")["est.dmdc.A_err_PV"].mean().reset_index()
    rho = spearman_manual(g["m"], g["est.dmdc.A_err_PV"])
    print(f"Spearman(m, A_err) on underact_small: {rho:.3f}  (will be unreliable if V_dim is constant)")

    header("Underact_small: V_dim by m (should be constant in your old run)")
    print(df.groupby("m")["V_dim"].mean())

# ---------- Underactuation (x0=0) ----------
if os.path.exists(P_U_X0ZERO):
    header("Underact_x0zero: normalized A_err vs m")
    df = pd.read_csv(P_U_X0ZERO)
    g = df.groupby("m").agg(A_err=("est.dmdc.A_err_PV","mean"),
                            V=("V_dim","mean"))
    rho_norm = spearman_manual(g.index, g["A_err"]/np.maximum(g["V"],1))
    print("V_dim by m:\n", df.groupby("m")["V_dim"].mean())
    print(f"Spearman(m, A_err/V_dim) = {rho_norm:.3f}  (expect < 0)")

# ---------- Underactuation (sparse B) ----------
if os.path.exists(P_U_SPARSEB):
    header("Underact_sparseB: V_dim vs m")
    df = pd.read_csv(P_U_SPARSEB)
    print("V_dim by m (mean):\n", df.groupby("m")["V_dim"].mean())

    # success in this regime (often all succeed once visible)
    tau = 1e-8
    succ_m = (df["est.dmdc.A_err_PV"] < tau).groupby(df["m"]).mean()
    print("DMDc success rate by m:\n", succ_m)

# ---------- Sparsity (wide) ----------
if os.path.exists(P_S_WIDE):
    header("Sparsity_wide: visibility & identification success vs density")
    dfs = pd.read_csv(P_S_WIDE)
    n = int(dfs["n"].iloc[0])

    # visibility success (K_rank == n)
    vis = (dfs["K_rank"] >= n).groupby(dfs["p_density"]).mean().rename("vis_succ")

    # full identification success (visible & small errors)
    tauA, tauB = 1e-8, 1e-8  # adjust to taste
    hasA = "est.dmdc.A_err_PV" in dfs.columns
    hasB = "est.dmdc.B_err_PV" in dfs.columns
    okA = (dfs["est.dmdc.A_err_PV"] < tauA) if hasA else True
    okB = (dfs["est.dmdc.B_err_PV"] < tauB) if hasB else True
    full = ((dfs["K_rank"] >= n) & okA & okB).groupby(dfs["p_density"]).mean().rename("full_succ")

    print(pd.concat([vis, full], axis=1))

    # trend on E[K_rank]
    g = dfs.groupby("p_density")["K_rank"].agg(["count","mean","std"])
    slope = (g["mean"].iloc[-1] - g["mean"].iloc[0]) / (g.index[-1] - g.index[0])
    print("\nK_rank by density:\n", g)
    print(f"\ndE[K_rank]/d(density) = {slope:.3f}  (expect > 0)")

# ---------- Sparsity (small) ----------
if os.path.exists(P_S_SMALL):
    header("Sparsity_small: trend check (likely saturated)")
    dfs = pd.read_csv(P_S_SMALL)
    g = dfs.groupby("p_density")["K_rank"].mean().reset_index()
    slope = (g["K_rank"].iloc[-1] - g["K_rank"].iloc[0]) / (g["p_density"].iloc[-1] - g["p_density"].iloc[0])
    print(f"dE[K_rank]/d(density): {slope:.3f}")
