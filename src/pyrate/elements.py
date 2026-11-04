"""Atomic masses used when an input atom does not provide an isotope mass."""

from __future__ import annotations

import re

# Conventional atomic weights (or representative isotope masses for elements
# without a standard atomic weight), in unified atomic mass units.  The table
# intentionally covers the elements most often encountered in kinetics input;
# users can always override a mass explicitly.
ATOMIC_MASSES_AMU: dict[str, float] = {
    "H": 1.00784,
    "D": 2.01410177812,
    "T": 3.0160492779,
    "He": 4.002602,
    "Li": 6.94,
    "Be": 9.0121831,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403163,
    "Ne": 20.1797,
    "Na": 22.98976928,
    "Mg": 24.305,
    "Al": 26.9815385,
    "Si": 28.085,
    "P": 30.973761998,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,
    "K": 39.0983,
    "Ca": 40.078,
    "Sc": 44.955908,
    "Ti": 47.867,
    "V": 50.9415,
    "Cr": 51.9961,
    "Mn": 54.938044,
    "Fe": 55.845,
    "Co": 58.933194,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.630,
    "As": 74.921595,
    "Se": 78.971,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.4678,
    "Sr": 87.62,
    "Y": 88.90584,
    "Zr": 91.224,
    "Nb": 92.90637,
    "Mo": 95.95,
    "Ru": 101.07,
    "Rh": 102.90550,
    "Pd": 106.42,
    "Ag": 107.8682,
    "Cd": 112.414,
    "In": 114.818,
    "Sn": 118.710,
    "Sb": 121.760,
    "Te": 127.60,
    "I": 126.90447,
    "Xe": 131.293,
    "Cs": 132.90545196,
    "Ba": 137.327,
    "La": 138.90547,
    "Ce": 140.116,
    "W": 183.84,
    "Pt": 195.084,
    "Au": 196.966569,
    "Hg": 200.592,
    "Pb": 207.2,
}


def canonical_symbol(label: str) -> str:
    """Convert a labelled atom such as ``H'`` or ``Cl2`` to an element symbol."""

    stripped = label.strip()
    if stripped in {"D", "T"}:
        return stripped
    match = re.match(r"([A-Za-z]{1,2})", stripped)
    if not match:
        raise ValueError(f"cannot infer element from atom label {label!r}")
    letters = match.group(1)
    symbol = letters[0].upper() + letters[1:].lower()
    if symbol not in ATOMIC_MASSES_AMU and len(symbol) == 2:
        # Labels such as HA are commonly atom names rather than elements.
        one_letter = symbol[0]
        if one_letter in ATOMIC_MASSES_AMU:
            symbol = one_letter
    return symbol


def atomic_mass_amu(label: str) -> float:
    """Return the default mass for an element or labelled atom."""

    symbol = canonical_symbol(label)
    try:
        return ATOMIC_MASSES_AMU[symbol]
    except KeyError as exc:
        raise ValueError(
            f"no built-in mass for {label!r}; provide mass_amu explicitly"
        ) from exc

# Representative neutral-atom isotopic masses (unified atomic mass units).
# Values are explicit nuclidic masses, not terrestrial-abundance atomic weights.
# Source: NIST Atomic Weights and Isotopic Compositions (neutral atoms):\n# https://pml.nist.gov/cgi-bin/Compositions/stand_alone.pl?ascii=ascii
ISOTOPE_MASSES_AMU: dict[str, float] = {
    "1H": 1.00782503223, "2H": 2.01410177812, "3H": 3.0160492779,
    "3He": 3.0160293201, "4He": 4.00260325413,
    "6Li": 6.0151228874, "7Li": 7.0160034366,
    "10B": 10.01293695, "11B": 11.00930536,
    "12C": 12.0, "13C": 13.00335483507, "14C": 14.0032419884,
    "14N": 14.00307400443, "15N": 15.00010889888,
    "16O": 15.99491461957, "17O": 16.99913175650, "18O": 17.99915961286,
    "19F": 18.99840316273,
    "28Si": 27.97692653465, "29Si": 28.97649466490, "30Si": 29.973770136,
    "31P": 30.97376199842,
    "32S": 31.9720711744, "33S": 32.9714589098, "34S": 33.967867004,
    "35Cl": 34.968852682, "37Cl": 36.965902602,
    "79Br": 78.9183376, "81Br": 80.9162897,
    "127I": 126.9044719,
}


def isotope_mass_amu(symbol: str, mass_number: int) -> float:
    """Return the neutral nuclidic mass; reject unknown isotopes explicitly."""

    canonical = canonical_symbol(symbol)
    if canonical == "D":
        canonical = "H"
        mass_number = 2
    elif canonical == "T":
        canonical = "H"
        mass_number = 3
    key = f"{int(mass_number)}{canonical}"
    try:
        return ISOTOPE_MASSES_AMU[key]
    except KeyError as exc:
        raise ValueError(f"unsupported isotope {key}; supply mass_amu explicitly") from exc
