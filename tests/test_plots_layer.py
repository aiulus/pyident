import numpy as np
from pathlib import Path
from ..plots import (
    plot_sigma_contour, plot_histogram, plot_rank_bars, plot_scree, plot_subspace_angles
)

def _nonempty(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0

def test_plot_sigma_and_histos(tmp_path):
    # sigma contour
    alpha = np.linspace(-1.0, 0.0, 16)
    beta  = np.linspace(-0.5, 0.5, 16)
    A = np.array([[0.0]])
    K = np.array([[1.0]])
    # Simple σ_min surface: distance to λi == 0 with K augment
    # Use toy grid values (monotone bowl)
    AA, BB = np.meshgrid(alpha, beta, indexing="ij")
    Sigma = np.sqrt(AA**2 + BB**2) + 1e-3
    png = tmp_path / "sig.png"; pdf = tmp_path / "sig.pdf"
    plot_sigma_contour(alpha, beta, Sigma, str(png), str(pdf), eigs=[0.0])
    assert _nonempty(png) and _nonempty(pdf)

    # histogram
    png2 = tmp_path / "hist.png"; pdf2 = tmp_path / "hist.pdf"
    data = np.random.standard_normal(100)
    plot_histogram(data, bins=10, out_png=str(png2), out_pdf=str(pdf2),
                   title="hist", xlabel="x")
    assert _nonempty(png2) and _nonempty(pdf2)

def test_plot_ranks_scree_angles(tmp_path):
    # rank bars
    vals = [5, 3, 1]
    labels = ["K", "Obs", "Ctrb"]
    p1 = tmp_path / "ranks.png"; p1b = tmp_path / "ranks.pdf"
    plot_rank_bars(vals, labels, out_png=str(p1), out_pdf=str(p1b))
    # scree
    svals = np.linspace(5, 0.1, 10)
    p2 = tmp_path / "scree.png"; p2b = tmp_path / "scree.pdf"
    plot_scree(svals, out_png=str(p2), out_pdf=str(p2b))
    # subspace angles
    rng = np.random.default_rng(0)
    n, k = 8, 3
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    P = Q[:, :k]; R = Q[:, :k] @ np.diag([1, 1, 1])  # same subspace
    angs = np.zeros(k)
    p3 = tmp_path / "ang.png"; p3b = tmp_path / "ang.pdf"
    plot_subspace_angles(angs, out_png=str(p3), out_pdf=str(p3b))
    from pathlib import Path
    def _ok(p: Path): return p.exists() and p.stat().st_size > 0
    assert all(_ok(x) for x in (p1, p1b, p2, p2b, p3, p3b))
