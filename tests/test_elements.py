"""Built-in atomic mass tables and the ``isotopic`` / ``average`` mass models."""

from __future__ import annotations

from typing import Any

import pytest

from pyrate.elements import (
    ATOMIC_MASSES_AMU,
    DEFAULT_MASS_MODEL,
    ISOTOPE_MASS_NUMBERS,
    ISOTOPIC_MASSES_AMU,
    MASS_MODELS,
    atomic_mass_amu,
    normalize_mass_model,
)

# AME2020 atomic masses (u) of the most abundant isotope.
ISOTOPIC_KEY_VALUES = {
    "H": 1.0078250319,
    "C": 12.0,
    "N": 14.00307400425,
    "O": 15.99491461926,
    "Cl": 34.968852694,
    "D": 2.01410177784,
    "T": 3.01604928132,
}
AVERAGE_KEY_VALUES = {
    "H": 1.00784,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "Cl": 35.45,
    "D": 2.01410177812,
    "T": 3.01604928199,
}
CROSS_CHECK_TOLERANCE_AMU = 1.0e-6


@pytest.mark.parametrize("symbol", sorted(ISOTOPIC_KEY_VALUES))
def test_isotopic_key_values(symbol: str) -> None:
    assert atomic_mass_amu(symbol, "isotopic") == ISOTOPIC_KEY_VALUES[symbol]
    assert atomic_mass_amu(symbol) == ISOTOPIC_KEY_VALUES[symbol]


@pytest.mark.parametrize("symbol", sorted(AVERAGE_KEY_VALUES))
def test_average_key_values_are_unchanged(symbol: str) -> None:
    assert atomic_mass_amu(symbol, "average") == AVERAGE_KEY_VALUES[symbol]


def test_key_isotopes_and_nist_agreement() -> None:
    assert atomic_mass_amu("C") == 12.0  # exact by definition of the unit
    assert ISOTOPE_MASS_NUMBERS["H"] == 1
    assert ISOTOPE_MASS_NUMBERS["C"] == 12
    assert ISOTOPE_MASS_NUMBERS["N"] == 14
    assert ISOTOPE_MASS_NUMBERS["O"] == 16
    assert ISOTOPE_MASS_NUMBERS["Cl"] == 35
    # NIST SRD 144 values quoted in docs/roadmap.md N4 agree within 1e-7 u.
    assert atomic_mass_amu("H") == pytest.approx(1.00782503223, abs=1.0e-9)
    assert atomic_mass_amu("Cl") == pytest.approx(34.968852682, abs=1.0e-7)


def test_default_model_is_isotopic() -> None:
    assert DEFAULT_MASS_MODEL == "isotopic"
    assert MASS_MODELS == ("isotopic", "average")
    for symbol in ATOMIC_MASSES_AMU:
        assert atomic_mass_amu(symbol) == ISOTOPIC_MASSES_AMU[symbol]


def test_deuterium_and_tritium_in_both_models() -> None:
    for model in MASS_MODELS:
        hydrogen = atomic_mass_amu("H", model)
        deuterium = atomic_mass_amu("D", model)
        tritium = atomic_mass_amu("T", model)
        assert 2.014 < deuterium < 2.0142
        assert 3.016 < tritium < 3.0161
        assert hydrogen < deuterium < tritium
    assert ISOTOPE_MASS_NUMBERS["D"] == 2
    assert ISOTOPE_MASS_NUMBERS["T"] == 3
    assert atomic_mass_amu("D", "isotopic") == pytest.approx(
        atomic_mass_amu("D", "average"), abs=1.0e-9
    )


def test_labelled_atoms_resolve_in_both_models() -> None:
    for model in MASS_MODELS:
        assert atomic_mass_amu("H'", model) == atomic_mass_amu("H", model)
        assert atomic_mass_amu("HA", model) == atomic_mass_amu("H", model)
        assert atomic_mass_amu("Cl2", model) == atomic_mass_amu("Cl", model)
        assert atomic_mass_amu(" cl ", model) == atomic_mass_amu("Cl", model)


@pytest.mark.parametrize("model", MASS_MODELS)
@pytest.mark.parametrize("label", ["Xx", "Q", "Rn", "Am", "Uuo"])
def test_unknown_symbols_raise(label: str, model: str) -> None:
    with pytest.raises(ValueError, match="no built-in mass"):
        atomic_mass_amu(label, model)  # type: ignore[arg-type]


@pytest.mark.parametrize("label", ["", "123", "*"])
def test_labels_without_letters_raise(label: str) -> None:
    with pytest.raises(ValueError, match="cannot infer element"):
        atomic_mass_amu(label)


def test_mass_model_validation() -> None:
    assert normalize_mass_model("isotopic") == "isotopic"
    assert normalize_mass_model(" Average ") == "average"
    for bad in ("monoisotopic", "", None, 1):
        with pytest.raises(ValueError, match="mass_model must be one of"):
            normalize_mass_model(bad)
    with pytest.raises(ValueError, match="mass_model must be one of"):
        atomic_mass_amu("H", "exact")  # type: ignore[arg-type]


def test_every_average_element_has_an_isotopic_value() -> None:
    assert set(ISOTOPIC_MASSES_AMU) == set(ATOMIC_MASSES_AMU)
    assert set(ISOTOPE_MASS_NUMBERS) == set(ATOMIC_MASSES_AMU)
    for symbol, mass in ISOTOPIC_MASSES_AMU.items():
        mass_number = ISOTOPE_MASS_NUMBERS[symbol]
        # Mass defects of stable nuclides stay well below 0.15 u.
        assert abs(mass - mass_number) < 0.15, symbol
        assert mass > 0.0


def _most_abundant_periodictable(symbol: str) -> tuple[int, float]:
    periodictable: Any = pytest.importorskip("periodictable")
    if symbol in {"D", "T"}:
        isotope = periodictable.H[ISOTOPE_MASS_NUMBERS[symbol]]
        return int(isotope.isotope), float(isotope.mass)
    element = getattr(periodictable, symbol)
    abundance, mass_number = max(
        (element[a].abundance, a) for a in element.isotopes if element[a].abundance > 0
    )
    assert abundance > 0
    return int(mass_number), float(element[mass_number].mass)


def _most_abundant_molmass(symbol: str) -> tuple[int, float]:
    molmass: Any = pytest.importorskip("molmass")
    if symbol == "T":
        pytest.skip("molmass lists no 3H entry")
    if symbol == "D":
        return 2, float(molmass.ELEMENTS["H"].isotopes[2].mass)
    isotopes = molmass.ELEMENTS[symbol].isotopes
    _, mass_number = max((iso.abundance, a) for a, iso in isotopes.items() if iso.abundance)
    return int(mass_number), float(isotopes[mass_number].mass)


@pytest.mark.parametrize("source", ["periodictable", "molmass"])
@pytest.mark.parametrize("symbol", sorted(ISOTOPIC_MASSES_AMU))
def test_isotopic_table_matches_independent_source(source: str, symbol: str) -> None:
    """Optional cross-check; runs only where the (non-dependency) package exists."""

    lookup = _most_abundant_periodictable if source == "periodictable" else _most_abundant_molmass
    mass_number, mass = lookup(symbol)
    assert mass_number == ISOTOPE_MASS_NUMBERS[symbol]
    assert abs(mass - ISOTOPIC_MASSES_AMU[symbol]) <= CROSS_CHECK_TOLERANCE_AMU
