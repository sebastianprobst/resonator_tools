# resonator_tools

A small python library to fit complex resonator scattering data.
It supports transmission, reflection and notch-type measurements.

> **Scientific correction (v2.2.0):** notch (hanger) photon-number estimates from
> `notch_port.get_photons_in_resonator()`/`get_single_photon_limit()` were too high by a factor
> of 2. Reflection-mode photon numbers and all fitted resonance parameters (`fr`, `Qi`, `Qc`,
> `Ql`) are unaffected. See [docs/photon_number_derivation.md](docs/photon_number_derivation.md)
> and the [changelog](CHANGELOG.md) for the derivation and details.

## Installation

To install the module from PyPI, run:

	pip install resonator-tools

To install from a local clone of this repository, run the following command inside the project directory:

	pip install .

This will install the package from your local source instead of downloading it from PyPI.

Description of the algorithm:
http://scitation.aip.org/content/aip/journal/rsi/86/2/10.1063/1.4907935 \
(Preprint: https://arxiv.org/abs/1410.3365)

Where is it used?
The "resonator_tools" have contributed to the following publications:
https://scholar.google.de/scholar?oi=bibs&hl=de&cites=690000812747125148&as_sdt=5

Institute where research was originally conducted:
https://www.phi.kit.edu/ustinov_downloads.php

Questions? Contact the author:
https://www.linkedin.com/in/dr-sebastian-probst-9685969a/
