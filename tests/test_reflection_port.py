from pathlib import Path

import numpy as np
import pytest

from resonator_tools import circuit
from resonator_tools.utilities import dBm2Watt

TEST_DATA = Path(__file__).parent / "test_data"

# Per-parameter tolerances: well-conditioned quantities get tight bounds,
# ill-conditioned ones (phases, coupling Qs, derived Qi) get looser bounds.
TOL_FREQ = 1e-6  # frequency: very well conditioned
TOL_Q = 1e-2  # loaded / coupling Q values
TOL_PHASE = 1e-2  # phase angles (theta0)
TOL_QI = 1e-2  # derived internal Q (error-propagation amplifies)
TOL_ERR = 0.25  # error estimates (covariance depends heavily on BLAS)
TOL_CHI2 = 0.25  # chi-square

_PARAM_TOL: dict[str, float] = {
    "fr": TOL_FREQ,
    "Ql": TOL_Q,
    "Qc": TOL_Q,
    "Qi": TOL_QI,
    "theta0": TOL_PHASE,
    # error estimates
    "Ql_err": TOL_ERR,
    "Qc_err": TOL_ERR,
    "fr_err": TOL_ERR,
    "chi_square": TOL_CHI2,
    "Qi_err": TOL_ERR,
}


@pytest.fixture()
def fitted_reflection_port():
    port = circuit.reflection_port()
    port.add_fromtxt(str(TEST_DATA / "S11.txt"), "dBmagphasedeg", 1)
    port.autofit()
    return port


# Physics parameters – should be stable across platforms
EXPECTED_FIT = {
    "Qi": 930126.6031513178,
    "Qc": 348041.41365946847,
    "Ql": 253270.75438078836,
    "fr": 7112934295.376775,
    "theta0": -0.0042367490753214615,
}

# Error estimates – depend on Jacobian / covariance, more sensitive to BLAS impl
EXPECTED_FIT_ERRS = {
    "Ql_err": 197.16591245739986,
    "Qc_err": 247.08125547680388,
    "fr_err": 7.249268499072724,
    "chi_square": 2.8947316812742316e-05,
    "Qi_err": 1743.0305211601014,
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
    assert spl == pytest.approx(-155.44065663450948, rel=TOL_QI)


def test_single_photon_limit_watt_matches_dbm(fitted_reflection_port):
    spl_dbm = fitted_reflection_port.get_single_photon_limit(unit="dBm")
    spl_watt = fitted_reflection_port.get_single_photon_limit(unit="watt")
    assert spl_watt == pytest.approx(dBm2Watt(spl_dbm), rel=1e-9)


def test_photons_in_resonator(fitted_reflection_port):
    photons = fitted_reflection_port.get_photons_in_resonator(-140, unit="dBm")
    assert photons == pytest.approx(34.99980812271039, rel=TOL_QI)


def test_photons_in_resonator_matches_single_photon_limit(fitted_reflection_port):
    spl = fitted_reflection_port.get_single_photon_limit(unit="dBm")
    photons = fitted_reflection_port.get_photons_in_resonator(spl, unit="dBm")
    assert photons == pytest.approx(1.0, rel=1e-9)


def test_unfitted_port_warns():
    port = circuit.reflection_port()
    with pytest.warns(UserWarning):
        assert port.get_single_photon_limit() is None
    with pytest.warns(UserWarning):
        assert port.get_photons_in_resonator(-140) is None


def test_circlefit_calc_errors_false(fitted_reflection_port):
    port = fitted_reflection_port
    result = port.circlefit(port.f_data, port.z_data, calc_errors=False)

    # no covariance-based error keys should be produced by this branch
    assert "chi_square" in result
    assert "Qi_err" not in result

    # central fit values should agree with the calc_errors=True autofit result
    assert result["fr"] == pytest.approx(port.fitresults["fr"], rel=TOL_FREQ)
    assert result["Ql"] == pytest.approx(port.fitresults["Ql"], rel=TOL_Q)
    assert result["Qc"] == pytest.approx(port.fitresults["Qc"], rel=TOL_Q)

    # chi_square must be a finite, nonnegative sum of squared complex magnitudes
    p = [result["fr"], result["Qc"], result["Ql"]]
    expected_chi_square = np.sum(
        np.abs(port._residuals_directrefl(p, port.f_data, port.z_data)) ** 2
    ) / (len(port.f_data) - len(p))
    assert np.isfinite(result["chi_square"])
    assert result["chi_square"] >= 0
    assert result["chi_square"] == pytest.approx(expected_chi_square, rel=1e-9)
