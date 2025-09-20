import numpy as np
from statistics import median
from ..metrics import cont2discrete_zoh, projected_errors, unified_generator
from ..estimators.dmdc import dmdc_fit

def _synth(n=5, m=2, T=180, dt=0.05, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) * 0.15 - 0.6 * np.eye(n)
    B = rng.standard_normal((n, m))
    x0 = rng.standard_normal(n)
    Ad, Bd = cont2discrete_zoh(A, B, dt)
    u = rng.standard_normal((T, m))
    X = np.empty((n, T+1)); X[:, 0] = x0
    for k in range(T):
        X[:, k+1] = Ad @ X[:, k] + Bd @ u[k, :]
    return A, B, Ad, Bd, x0, u, X

def _noisify(arr, snr_db, rng):
    """Additive white noise with target SNR (power ratio on array)."""
    sig_pow = np.mean(arr**2) + 1e-12
    snr = 10**(snr_db / 10)
    noise_pow = sig_pow / snr
    noise = rng.standard_normal(arr.shape) * np.sqrt(noise_pow)
    return arr + noise

def test_dmdc_projected_errors_monotone_with_noise():
    A, B, Ad, Bd, x0, u, X = _synth(seed=1)
    K = unified_generator(A, B, x0, mode="unrestricted")
    U, S, _ = np.linalg.svd(K, full_matrices=False)
    PV = U[:, : (S > 1e-10).sum()]

    SNRs = [60, 40, 20]
    meds = []
    rng = np.random.default_rng(2)
    for snr in SNRs:
        errs = []
        for sd in range(6):
            Xn = _noisify(X, snr, np.random.default_rng(sd))
            Un = _noisify(u.T, snr, np.random.default_rng(sd+100))
            Xtrain, Xp = Xn[:, :-1], Xn[:, 1:]
            Ahat, Bhat = dmdc_fit(Xtrain, Xp, Un)
            eA, eB = projected_errors(Ahat, Bhat, Ad, Bd, Vbasis=PV)
            errs.append(float(eA + eB))
        meds.append(median(errs))
    assert all(meds[i] <= meds[i+1] + 1e-9 for i in range(len(meds)-1))
