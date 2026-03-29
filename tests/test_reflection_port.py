from pathlib import Path

import pytest

from resonator_tools import circuit

TEST_DATA = Path(__file__).parent / "test_data"
REL_TOL = 1e-2  # 1 %
REL_TOL_LOOSE = 1e-1  # 10 % – for error estimates sensitive to BLAS backend


@pytest.fixture()
def fitted_reflection_port():
    port = circuit.reflection_port()
    port.add_fromtxt(str(TEST_DATA / "S11.txt"), "dBmagphasedeg", 1)
    port.autofit()
    return port


# Physics parameters – should be stable across platforms
EXPECTED_FIT = {
    "Qi": 930112.8190009119,
    "Qc": 348037.0941443813,
    "Ql": 253267.4449391589,
    "fr": 7112934296.265527,
    "theta0": -0.004289407283891134,
}

# Error estimates – depend on Jacobian / covariance, more sensitive to BLAS impl
EXPECTED_FIT_ERRS = {
    "Ql_err": 198.2631598268342,
    "Qc_err": 247.70531526252788,
    "fr_err": 7.299316150758586,
    "chi_square": 2.917631301044999e-05,
    "Qi_err": 1748.981727587284,
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
    assert spl == pytest.approx(-155.4405970360725, rel=REL_TOL)
