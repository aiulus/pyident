import numpy as np
import numpy.linalg as npl
import pytest

from pyident import ensembles, metrics, estimators

# ---- helpers ---------------------------------------------------------

def _prbs(m, T, rng, dwell=5, scale=1.0):
    U = np.empty((m, T))
    for i in range(m):
        t = 0
        s = 1.0
        while t < T:
            L = min(dwell, T - t)
            U[i, t:t+L] = s
            s = -s
            t += L
    U += 0.02 * rng.standard_normal((m, T))  # tiny dither for better PE
    return scale * U

def _simulate_dt(Ad, Bd, x0, U):
    n, T = Ad.shape[0], U.shape[1]
    X = np.zeros((n, T+1))
    X[:, 0] = x0
    for t in range(T):
        X[:, t+1] = Ad @ X[:, t] + Bd @ U[:, t]
    return X

def _mse_pair(Ahat, Bhat, Ad, Bd):
    eA = npl.norm(Ahat - Ad, "fro")
    eB = npl.norm(Bhat - Bd, "fro")
    return 0.5*(eA + eB)

def _pick_filtered_x0(Ad, Bd, rng, k_candidates=400, q=0.85):
    # sample on the unit sphere, score by structured PBH margin, keep a top-quantile one
    cand = rng.standard_normal((Ad.shape[0], k_candidates))
    cand /= npl.norm(cand, axis=0, keepdims=True) + 1e-30
    scores = [metrics.pbh_margin_structured(Ad, Bd, cand[:, j]) for j in range(k_candidates)]
    tau = np.quantile(scores, q)
    idx = [j for j, s in enumerate(scores) if s >= tau]
    return cand[:, int(idx[0]) if idx else int(np.argmax(scores))]

def _dmdc_ridge_scaled(X0, X1, U, lam_scale=1e-12):
    Z = np.vstack([X0, U])
    # scale-aware tiny ridge: lam ~ lam_scale * ||Z||_2^2
    try:
        smax = npl.svd(Z, compute_uv=False)[0] if Z.size else 1.0
    except Exception:
        smax = npl.norm(Z, 2) if Z.size else 1.0
    lam = lam_scale * (smax**2 + 1e-30)
    return estimators.dmdc_ridge(X0, X1, U, lam=lam)

# ---- main test -------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("n,m,T,dt", [(12, 3, 800, 0.05)])
def test_filtering_benefit_increases_with_d(n, m, T, dt):
    """
    Hypothesis: the paired improvement Δ(d) = MSE_random - MSE_filtered
    is (weakly) increasing in d = dim(U) = n - rank(A,B), for a single-run experiment.
    We use stable blocks, scale-aware ridge, and the same system/input per pair.
    """
    rng = np.random.default_rng(123)
    d_list = [1, 2, 3, 4]             # unreachable dimensions to test
    n_trials = 40                     # keep modest for CI; raise locally for more power
    q_filter = 0.85                   # top-quantile filter on structured PBH margin

    med_improvements = []

    for d in d_list:
        improvements = []
        for _ in range(n_trials):
            # same system for both cohorts
            A, B, _ = ensembles.draw_with_ctrb_rank(
                n=n, m=m, r=n-d, rng=rng,
                base_c="stable", base_u="stable"
            )
            Ad, Bd = metrics.cont2discrete_zoh(A, B, dt)

            # same input for both cohorts (PRBS + tiny dither)
            U = _prbs(m, T, rng, dwell=7, scale=1.0)

            # RANDOM x0
            x0_r = rng.standard_normal(n); x0_r /= npl.norm(x0_r) + 1e-30
            X  = _simulate_dt(Ad, Bd, x0_r, U)
            X0, X1 = X[:, :-1], X[:, 1:]
            Ahat_r, Bhat_r = _dmdc_ridge_scaled(X0, X1, U)
            mse_r = _mse_pair(Ahat_r, Bhat_r, Ad, Bd)

            # FILTERED x0
            x0_f = _pick_filtered_x0(Ad, Bd, rng, k_candidates=400, q=q_filter)
            Xf = _simulate_dt(Ad, Bd, x0_f, U)
            X0f, X1f = Xf[:, :-1], Xf[:, 1:]
            Ahat_f, Bhat_f = _dmdc_ridge_scaled(X0f, X1f, U)
            mse_f = _mse_pair(Ahat_f, Bhat_f, Ad, Bd)

            improvements.append(mse_r - mse_f)

        med_improvements.append(float(np.median(improvements)))

    # Soft monotonicity check: median improvement is non-decreasing with d
    # Allow a tiny slack for randomness
    for i in range(len(d_list)-1):
        assert med_improvements[i+1] + 1e-3 >= med_improvements[i], \
            f"Filtering gain did not grow from d={d_list[i]} to d={d_list[i+1]}: " \
            f"{med_improvements[i]:.3g} -> {med_improvements[i+1]:.3g}"

    # Optional: trend check via Spearman/Kendall (not failing, only informative)
    # (avoid adding scipy to hard deps; print small summary)
    print("\nMedian improvements by d:", dict(zip(d_list, med_improvements)))
