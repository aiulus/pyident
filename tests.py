import pandas as pd, numpy as np
df = pd.read_csv("runs/underact_small.csv")

# % of seeds where K_rank is non-decreasing in m
ok = 0; total = 0
for seed, g in df.groupby("seed"):
    g = g.sort_values("m")
    total += 1
    ok += int((g["K_rank"].diff().fillna(0) >= -1e-9).all())
print(f"Monotone K_rank across m (by seed): {ok}/{total} = {ok/total:.1%}")

# Spearman correlation: m ↑ ⇒ A_err_PV ↓  (should be negative)
g = df.groupby("m")["est.dmdc.A_err_PV"].mean().reset_index()
rho = g["m"].corr(g["est.dmdc.A_err_PV"], method="spearman")
print(f"Spearman(m, A_err): {rho:.3f}  (expect < 0)")


dfs = pd.read_csv("runs/sparsity_small.csv")
g = dfs.groupby("p_density")["K_rank"].mean().reset_index()
slope = (g["K_rank"].iloc[-1] - g["K_rank"].iloc[0]) / (g["p_density"].iloc[-1] - g["p_density"].iloc[0])
print(f"dE[K_rank]/d(density): {slope:.3f}  (expect > 0)")
