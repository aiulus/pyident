import numpy as np

from pyident.estimators import dmdc_ridge, dmdc_tls
from pyident.experiments import sim_unctrb_pbh_estimators as pbh_est
from pyident.experiments import sim_unctrb_pbh_error_boxplots as pbh_box


def _dummy_data():
    rng = np.random.default_rng(0)
    n, m, T = 3, 2, 8
    X0 = rng.standard_normal((n, T))
    X1 = rng.standard_normal((n, T))
    U = rng.standard_normal((m, T))
    return X0, X1, U


def _assert_ridge_selected(parser, module, args_list):
    args = parser.parse_args(args_list + ["--ridge", "--ridge-lam", "1e-3"])
    fn = module._dmdc_callable(args)
    X0, X1, U = _dummy_data()
    A1, B1 = fn(X0, X1, U, 1.0)
    A2, B2 = dmdc_ridge(X0, X1, U, lam=1e-3)
    assert np.allclose(A1, A2)
    assert np.allclose(B1, B2)


def _assert_tls_selected(parser, module, args_list):
    args = parser.parse_args(args_list)
    fn = module._dmdc_callable(args)
    X0, X1, U = _dummy_data()
    A1, B1 = fn(X0, X1, U, 1.0)
    A2, B2 = dmdc_tls(X0, X1, U)
    assert np.allclose(A1, A2)
    assert np.allclose(B1, B2)


def test_dmdc_ridge_flag_estimators():
    base_args = [
        "--dataset-csv",
        "dummy.csv",
        "--dataset-npz",
        "dummy.npz",
        "--outdir",
        "dummy_out",
        "--pbh-threshold",
        "1e-6",
    ]
    _assert_ridge_selected(pbh_est.build_parser(), pbh_est, base_args)
    _assert_tls_selected(pbh_est.build_parser(), pbh_est, base_args)


def test_dmdc_ridge_flag_boxplots():
    base_args = [
        "--selected-npz",
        "dummy_sel.npz",
        "--outdir",
        "dummy_out",
    ]
    _assert_ridge_selected(pbh_box.build_parser(), pbh_box, base_args)
    _assert_tls_selected(pbh_box.build_parser(), pbh_box, base_args)
