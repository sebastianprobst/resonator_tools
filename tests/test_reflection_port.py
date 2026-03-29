from pathlib import Path

import pytest

from resonator_tools import circuit

TEST_DATA = Path(__file__).parent / "test_data"

# Per-parameter tolerances: well-conditioned quantities get tight bounds,
# ill-conditioned ones (phases, coupling Qs, derived Qi) get looser bounds.
TOL_FREQ = 1e-6       # frequency: very well conditioned
TOL_Q = 5e-4          # loaded / coupling Q values
TOL_PHASE = 5e-3      # phase angles (theta0)
TOL_QI = 1e-3         # derived internal Q (error-propagation amplifies)
TOL_ERR = 5e-2        # error estimates (covariance depends heavily on BLAS)
TOL_CHI2 = 5e-2       # chi-square

_PARAM_TOL: dict[str, float] = {
    "fr":         TOL_FREQ,
    "Ql":         TOL_Q,
    "Qc":         TOL_Q,
    "Qi":         TOL_QI,
    "theta0":     TOL_PHASE,
    # error estimates
    "Ql_err":     TOL_ERR,
    "Qc_err":     TOL_ERR,
    "fr_err":     TOL_ERR,
    "chi_square": TOL_CHI2,
    "Qi_err":     TOL_ERR,
}


@pytest.fixture()
def fitted_reflection_port():
    port = circuit.reflection_port()
    port.add_fromtxt(str(TEST_DATA / "S11.txt"), "dBmagphasedeg", 1)
    port.autofit()
    return port


# Physics parameters – should be stable across platforms
EXPECTED_FIT = {
    "Qi": 930094.6001766999,
    "Qc": 348030.868112033,
    "Ql": 253262.7970861197,
    "fr": 7112934302.445624,
    "theta0": -0.004599190548892803,
}

# Error estimates – depend on Jacobian / covariance, more sensitive to BLAS impl
EXPECTED_FIT_ERRS = {
    "Ql_err": 204.8556437035391,
    "Qc_err": 252.03202246432647,
    "fr_err": 7.632195390242183,
    "chi_square": 3.068568136623117e-05,
    "Qi_err": 1787.6795826026673,
}


@pytest.mark.parametrize("key,expected", list(EXPECTED_FIT.items()))
def test_fitresults(fitted_reflection_port, key, expected):
    actual = fitted_reflection_port.fitresults[key]
    tol = _PARAM_TOL[key]
    assert actual == pytest.approx(expected, rel=tol), (
        f"{key}: {actual} != {expected} (rel_tol={tol})"
    )


@pytest.mark.parametrize("key,expected", list(EXPECTED_FIT_ERRS.items()))
def test_fitresults_errs(fitted_reflection_port, key, expected):
    actual = fitted_reflection_port.fitresults[key]
    tol = _PARAM_TOL[key]
    assert actual == pytest.approx(expected, rel=tol), (
        f"{key}: {actual} != {expected} (rel_tol={tol})"
    )


def test_single_photon_limit(fitted_reflection_port):
    spl = fitted_reflection_port.get_single_photon_limit()
    assert spl == pytest.approx(-155.44051531902153, rel=TOL_QI)
