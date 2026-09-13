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
    assert spl == pytest.approx(-145.79176728932942, rel=TOL_QI)


def test_photons_in_resonator(fitted_notch_port):
    photons = fitted_notch_port.get_photons_in_resonator(-140, unit="dBm", diacorr=True)
    assert photons == pytest.approx(3.7946937232320708, rel=TOL_QI)


def test_photons_in_resonator_matches_single_photon_limit(fitted_notch_port):
    # the single-photon power must convert back to exactly one photon
    spl = fitted_notch_port.get_single_photon_limit(unit="dBm", diacorr=True)
    photons = fitted_notch_port.get_photons_in_resonator(spl, unit="dBm", diacorr=True)
    assert photons == pytest.approx(1.0, rel=1e-9)


def test_notch_photon_number_is_half_of_equivalent_reflection():
    # for identical fr, Qi, Qc and incident power, a symmetric notch driven
    # from one side stores half the photons of a genuine one-port reflection
    # resonator, since only half of its total external coupling is driven.
    fr, Qi, Qc = 6e9, 8e5, 2e5

    notch = circuit.notch_port()
    notch.fitresults = {
        "fr": fr,
        "Qc_dia_corr": Qc,
        "Qi_dia_corr": Qi,
        "absQc": Qc,
        "Qi_no_corr": Qi,
    }
    refl = circuit.reflection_port()
    refl.fitresults = {"fr": fr, "Qc": Qc, "Qi": Qi}

    n_notch = notch.get_photons_in_resonator(-140, unit="dBm", diacorr=True)
    n_refl = refl.get_photons_in_resonator(-140, unit="dBm")
    assert n_refl == pytest.approx(2.0 * n_notch, rel=1e-12)
