from pathlib import Path

import pytest

from resonator_tools import circuit

TEST_DATA = Path(__file__).parent / "test_data"
REL_TOL = 1e-2  # 1 %
REL_TOL_LOOSE = 2e-1  # 20 % – for error estimates sensitive to BLAS backend


@pytest.fixture()
def fitted_reflection_port():
    port = circuit.reflection_port()
    port.add_fromtxt(str(TEST_DATA / "S11.txt"), "dBmagphasedeg", 1)
    port.autofit()
    return port


# Physics parameters – should be stable across platforms
EXPECTED_FIT = {
    "Qi": 930113.7747436144,
    "Qc": 348037.4517718714,
    "Ql": 253267.70518555838,
    "fr": 7112934295.402461,
    "theta0": -0.004252139385339115,
}

# Error estimates – depend on Jacobian / covariance, more sensitive to BLAS impl
EXPECTED_FIT_ERRS = {
    "Ql_err": 197.28887351529272,
    "Qc_err": 247.0396920509826,
    "fr_err": 7.252389806956782,
    "chi_square": 2.8971121788655598e-05,
    "Qi_err": 1744.1631019343233,
}


@pytest.mark.parametrize("key,expected", list(EXPECTED_FIT.items()))
def test_fitresults(fitted_reflection_port, key, expected):
    actual = fitted_reflection_port.fitresults[key]
    assert actual == pytest.approx(expected, rel=REL_TOL), (
        f"{key}: {actual} != {expected}"
    )


@pytest.mark.parametrize("key,expected", list(EXPECTED_FIT_ERRS.items()))
def test_fitresults_errs(fitted_reflection_port, key, expected):
    actual = fitted_reflection_port.fitresults[key]
    assert actual == pytest.approx(expected, rel=REL_TOL_LOOSE), (
        f"{key}: {actual} != {expected}"
    )


def test_single_photon_limit(fitted_reflection_port):
    spl = fitted_reflection_port.get_single_photon_limit()
    assert spl == pytest.approx(-155.44060149974172, rel=REL_TOL)