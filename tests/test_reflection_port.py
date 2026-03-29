from pathlib import Path

import pytest

from resonator_tools import circuit

TEST_DATA = Path(__file__).parent / "test_data"

# Per-parameter tolerances: well-conditioned quantities get tight bounds,
# ill-conditioned ones (phases, coupling Qs, derived Qi) get looser bounds.
TOL_FREQ = 1e-6       # frequency: very well conditioned
TOL_Q = 1e-2          # loaded / coupling Q values
TOL_PHASE = 1e-2      # phase angles (theta0)
TOL_QI = 1e-2         # derived internal Q (error-propagation amplifies)
TOL_ERR = 0.25        # error estimates (covariance depends heavily on BLAS)
TOL_CHI2 = 0.25       # chi-square

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
    "Qi": 930094.8990016765,
    "Qc": 348030.9799289552,
    "Ql": 253262.87845553007,
    "fr": 7112934302.251634,
    "theta0": -0.004588297296866701,
}

# Error estimates – depend on Jacobian / covariance, more sensitive to BLAS impl
EXPECTED_FIT_ERRS = {
    "Ql_err": 204.6414300644987,
    "Qc_err": 251.8921341730375,
    "fr_err": 7.621016242343605,
    "chi_square": 3.06342249507221e-05,
    "Qi_err": 1786.2977088371454,
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
    assert spl == pytest.approx(-155.44051671457905, rel=TOL_QI)
