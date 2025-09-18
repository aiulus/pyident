
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def plot_sigma_contour(alpha, beta, Sigma, out_png, out_pdf):
    A,B = np.meshgrid(alpha, beta, indexing="ij")
    plt.figure()
    cs = plt.contourf(A, B, Sigma, levels=30)
    plt.colorbar(cs, label=r"$\sigma_{\min}$")
    plt.xlabel(r"$\Re(\lambda)$"); plt.ylabel(r"$\Im(\lambda)$")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.savefig(out_pdf)
    plt.close()

def emit_pgfplots_tex(csv_path, tex_path, xlabel="Re(λ)", ylabel="Im(λ)"):
    with open(tex_path,"w") as f:
        f.write("""\begin{tikzpicture}
\begin{axis}[view={0}{90}, colorbar, xlabel=%s, ylabel=%s]
\addplot3[surf,shader=interp] table[x=alpha,y=beta,z=sigma,col sep=comma] {%s};
\end{axis}
\end{tikzpicture}
""" % (xlabel, ylabel, csv_path))
