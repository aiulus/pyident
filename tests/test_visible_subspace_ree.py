import numpy as np


def _orth(P: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(P)
    return Q


def _std_ree(Ahat, Bhat, A, B, eps=1e-15):
    EA = np.linalg.norm(Ahat - A, "fro") / max(np.linalg.norm(A, "fro"), eps)
    EB = np.linalg.norm(Bhat - B, "fro") / max(np.linalg.norm(B, "fro"), eps)
    return 0.5 * (EA + EB)


def _V_ree(Ahat, Bhat, A, B, PV, eps=1e-15):
    AV = PV.T @ A @ PV
    BV = PV.T @ B
    AVh = PV.T @ Ahat @ PV
    BVh = PV.T @ Bhat
    EA = np.linalg.norm(AVh - AV, "fro") / max(np.linalg.norm(AV, "fro"), eps)
    EB = np.linalg.norm(BVh - BV, "fro") / max(np.linalg.norm(BV, "fro"), eps)
    return 0.5 * (EA + EB)


def test_visible_subspace_ree_equivalence_class_perturbation():
    rng = np.random.default_rng(0)
    n, m, k = 6, 2, 3
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, m))
    # Build a single orthonormal basis and split to guarantee [PV PW] is orthonormal
    P = _orth(rng.standard_normal((n, n)))
    PV = P[:, :k]
    PW = P[:, k:]

    At = P.T @ A @ P
    At2 = At.copy()
    At2[k:, k:] += 3.0 * np.eye(n - k)
    A2 = P @ At2 @ P.T
    B2 = B.copy()

    std_err = _std_ree(A2, B2, A, B)
    V_err = _V_ree(A2, B2, A, B, PV)

    assert std_err > 1e-6
    assert V_err < 1e-10
