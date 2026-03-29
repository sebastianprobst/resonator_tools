from pathlib import Path

import pytest

from resonator_tools import circuit

TEST_DATA = Path(__file__).parent / "test_data"
REL_TOL = 1e-2  # 1 %
REL_TOL_LOOSE = 2e-1  # 20 % – for error estimates sensitive to BLAS backend


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
    "Qi_dia_corr": 134083.69749539843,
    "Qi_no_corr": 134268.24413061654,
    "absQc": 288018.48589428945,
    "Qc_dia_corr": 288871.3536268758,
    "Ql": 91576.9633014615,
    "fr": 5922518761.752142,
    "theta0": -3.064731029979871,
    "phi0": 0.07686179289378793,
}

# Error estimates – depend on Jacobian / covariance, more sensitive to BLAS impl
EXPECTED_FIT_ERRS = {
    "phi0_err": 0.002428183848052563,
    "Ql_err": 249.85481425941873,
    "absQc_err": 478.80070568518545,
    "fr_err": 97.66230198814007,
    "chi_square": 8.235384880699502e-06,
    "Qi_no_corr_err": 489.1038773500614,
    "Qi_dia_corr_err": 486.71905003127034,
}


@pytest.mark.parametrize("key,expected", list(EXPECTED_FIT.items()))
def test_fitresults(fitted_notch_port, key, expected):
    actual = fitted_notch_port.fitresults[key]
    assert actual == pytest.approx(expected, rel=REL_TOL), (
        f"{key}: {actual} != {expected}"
    )


@pytest.mark.parametrize("key,expected", list(EXPECTED_FIT_ERRS.items()))
def test_fitresults_errs(fitted_notch_port, key, expected):
    actual = fitted_notch_port.fitresults[key]
    assert actual == pytest.approx(expected, rel=REL_TOL_LOOSE), (
        f"{key}: {actual} != {expected}"
    )


def test_single_photon_limit(fitted_notch_port):
    spl = fitted_notch_port.get_single_photon_limit(diacorr=True)
    assert spl == pytest.approx(-149.0047920306113, rel=REL_TOL)


def test_photons_in_resonator(fitted_notch_port):
    photons = fitted_notch_port.get_photons_in_resonator(
        -140, unit="dBm", diacorr=True
    )
    assert photons == pytest.approx(7.952051848524633, rel=REL_TOL)