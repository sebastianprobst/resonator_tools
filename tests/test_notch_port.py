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
    "Qi_dia_corr": 131942.67358723632,
    "Qi_no_corr": 132113.9433815557,
    "absQc": 283026.8546805501,
    "Qc_dia_corr": 283816.09908586234,
    "Ql": 90070.14976430265,
    "fr": 5922518836.981035,
    "theta0": -3.0669993055994165,
    "phi0": 0.07459383464378908,
}

# Error estimates – depend on Jacobian / covariance, more sensitive to BLAS impl
EXPECTED_FIT_ERRS = {
    "phi0_err": 0.004355197164286936,
    "Ql_err": 407.8898387419578,
    "absQc_err": 713.365872966948,
    "fr_err": 176.2677364082469,
    "chi_square": 2.1858663404166833e-05,
    "Qi_no_corr_err": 825.1196040884977,
    "Qi_dia_corr_err": 819.1718123170862,
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
    assert spl == pytest.approx(-148.9373593528299, rel=TOL_QI)


def test_photons_in_resonator(fitted_notch_port):
    photons = fitted_notch_port.get_photons_in_resonator(-140, unit="dBm", diacorr=True)
    assert photons == pytest.approx(7.829534376688001, rel=TOL_QI)
