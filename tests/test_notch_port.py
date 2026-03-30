from pathlib import Path

import pytest

from resonator_tools import circuit

TEST_DATA = Path(__file__).parent / "test_data"

# Per-parameter tolerances: well-conditioned quantities get tight bounds,
# ill-conditioned ones (phases, coupling Qs, derived Qi) get looser bounds.
TOL_FREQ = 1e-6  # frequency: very well conditioned
TOL_Q = 1e-2  # loaded / coupling Q values
TOL_PHASE = 1e-2  # phase angles (theta0, phi0)
TOL_QI = 1e-2  # derived internal Q (error-propagation amplifies)
TOL_ERR = 0.25  # error estimates (covariance depends heavily on BLAS)
TOL_CHI2 = 0.25  # chi-square

_PARAM_TOL: dict[str, float] = {
    "fr": TOL_FREQ,
    "Ql": TOL_Q,
    "absQc": TOL_Q,
    "Qc_dia_corr": TOL_Q,
    "Qi_dia_corr": TOL_QI,
    "Qi_no_corr": TOL_QI,
    "theta0": TOL_PHASE,
    "phi0": TOL_PHASE,
    # error estimates
    "phi0_err": TOL_ERR,
    "Ql_err": TOL_ERR,
    "absQc_err": TOL_ERR,
    "fr_err": TOL_ERR,
    "chi_square": TOL_CHI2,
    "Qi_no_corr_err": TOL_ERR,
    "Qi_dia_corr_err": TOL_ERR,
}


@pytest.fixture()
def fitted_notch_port():
    port = circuit.notch_port()
    port.add_froms2p(
        str(TEST_DATA / "S21testdata.s2p"),
        3,
        4,
        "realimag",
        fdata_unit=1e9,
        delimiter=None,
    )
    port.autofit()
    return port


# Physics parameters – should be stable across platforms
EXPECTED_FIT = {
    "Qi_dia_corr": 127756.00296208196,
    "Qi_no_corr": 127902.2147023118,
    "absQc": 273317.60431328195,
    "Qc_dia_corr": 273987.67635913135,
    "Ql": 87129.11290017859,
    "fr": 5922518993.515957,
    "theta0": -3.071640871448464,
    "phi0": 0.06995178214132913,
}

# Error estimates – depend on Jacobian / covariance, more sensitive to BLAS impl
EXPECTED_FIT_ERRS = {
    "phi0_err": 0.012190368694798224,
    "Ql_err": 823.7869748980518,
    "absQc_err": 1122.0098453263265,
    "fr_err": 458.8122462688606,
    "chi_square": 7.343748879768608e-05,
    "Qi_no_corr_err": 1714.6461190120976,
    "Qi_dia_corr_err": 1711.4071916713567,
}


@pytest.mark.parametrize("key,expected", list(EXPECTED_FIT.items()))
def test_fitresults(fitted_notch_port, key, expected):
    actual = fitted_notch_port.fitresults[key]
    tol = _PARAM_TOL[key]
    assert actual == pytest.approx(expected, rel=tol), (
        f"{key}: {actual} != {expected} (rel_tol={tol})"
    )


@pytest.mark.parametrize("key,expected", list(EXPECTED_FIT_ERRS.items()))
def test_fitresults_errs(fitted_notch_port, key, expected):
    actual = fitted_notch_port.fitresults[key]
    tol = _PARAM_TOL[key]
    assert actual == pytest.approx(expected, rel=tol), (
        f"{key}: {actual} != {expected} (rel_tol={tol})"
    )


def test_single_photon_limit(fitted_notch_port):
    spl = fitted_notch_port.get_single_photon_limit(diacorr=True)
    assert spl == pytest.approx(-148.80206724596923, rel=TOL_QI)


def test_photons_in_resonator(fitted_notch_port):
    photons = fitted_notch_port.get_photons_in_resonator(-140, unit="dBm", diacorr=True)
    assert photons == pytest.approx(7.5893874464641415, rel=TOL_QI)
