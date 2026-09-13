# Changelog

All notable changes to this project are documented here. This changelog starts at v2.2.0;
earlier releases are available as [GitHub tags](https://github.com/sebastianprobst/resonator_tools/tags).

## [2.2.0] - 2026-09-13

### Fixed

- **Notch (hanger) photon-number conversion.** `notch_port.get_photons_in_resonator()` and
  `notch_port.get_single_photon_limit()` used the same coefficient (`4`) as the one-port
  reflection case, overestimating the intracavity photon number by a factor of 2 for a symmetric
  notch resonator driven from one feedline end. The correct coefficient is `2`; `reflection_port`
  was already correct and is unchanged. If you rescaled notch photon numbers yourself to work
  around this, remove that adjustment after upgrading. See
  [docs/photon_number_derivation.md](docs/photon_number_derivation.md) for the derivation and a
  literature cross-check. Reported by [@IVN-tone](https://github.com/IVN-tone) in
  [#21](https://github.com/sebastianprobst/resonator_tools/pull/21).
- `reflection_port.circlefit(calc_errors=False)` raised `ValueError: not enough values to unpack`
  because it called the 4-parameter notch residual with reflection's 3-parameter fit vector. It
  now uses the correct reflection residual and sums squared magnitudes.

### Added

- Derivation and literature cross-check for the photon-number conversion:
  [docs/photon_number_derivation.md](docs/photon_number_derivation.md).
- Regression tests for both fixes, including a reflection/notch factor-of-two consistency check
  and `calc_errors=False` coverage for both port classes.

[2.2.0]: https://github.com/sebastianprobst/resonator_tools/releases/tag/v2.2.0
