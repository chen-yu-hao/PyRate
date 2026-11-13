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
    "T": 3.01604928199,
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


# Polyrate 2023 default monoisotopic masses (from interface.F `data ams`),
# in unified atomic mass units (12C = 12.0 amu).
POLYRATE_ATOMIC_MASSES_AMU: dict[str, float] = {
    "H": 1.007825,
    "D": 2.01400,
    "T": 3.01605,
    "He": 4.00260,
    "Li": 7.01600,
    "Be": 9.01218,
    "B": 11.00931,
    "C": 12.0,
    "N": 14.00307,
    "O": 15.99491,
    "F": 18.99840,
    "Ne": 19.99244,
    "Na": 22.9898,
    "Mg": 23.98504,
    "Al": 26.98153,
    "Si": 27.97693,
    "P": 30.97376,
    "S": 31.97207,
    "Cl": 34.96885,
    "Ar": 39.96238,
    "K": 38.96371,
    "Ca": 39.96259,
    "Sc": 44.95592,
    "Ti": 47.94795,
    "V": 50.9440,
    "Cr": 51.9405,
    "Mn": 54.9381,
    "Fe": 55.9349,
    "Co": 58.9332,
    "Ni": 57.9353,
    "Cu": 62.9298,
    "Zn": 63.9291,
    "Ga": 68.9257,
    "Ge": 73.9219,
    "As": 74.9216,
    "Se": 79.9165,
    "Br": 78.9183,
    "Kr": 83.9115,
    "Rb": 84.9117,
    "Sr": 87.9056,
    "Y": 88.9054,
    "Zr": 89.9043,
    "Nb": 92.9060,
    "Mo": 97.9055,
    "Ru": 101.9037,
    "Rh": 102.9048,
    "Pd": 105.9032,
    "Ag": 106.90509,
    "Cd": 113.9036,
    "In": 114.9041,
    "Sn": 119.9022,
    "Sb": 120.9038,
    "Te": 129.9067,
    "I": 126.9044,
    "Xe": 131.9042,
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

# Explicit isotope mass table for isotopic substitution (e.g. 1H, 2H, 13C, 18O, 37Cl).
ISOTOPE_MASSES_AMU: dict[str, float] = {
    "1H": 1.00782503223,
    "2H": 2.01410177812,
    "3H": 3.01604928199,
    "12C": 12.0,
    "13C": 13.00335483507,
    "14C": 14.0032419884,
    "14N": 14.00307400443,
    "15N": 15.00010889888,
    "16O": 15.99491461957,
    "17O": 16.99913175650,
    "18O": 17.99915961286,
    "19F": 18.99840316273,
    "28Si": 27.97692653465,
    "31P": 30.97376199842,
    "32S": 31.9720711744,
    "34S": 33.967867004,
    "35Cl": 34.968852682,
    "37Cl": 36.965902602,
    "79Br": 78.9183376,
    "81Br": 80.9162897,
}


def canonical_symbol(label: str) -> str:
    """Convert a labelled atom such as ``H'`` or ``Cl2`` to an element symbol."""

    stripped = label.strip()
    if stripped in {"D", "T"}:
        return stripped
    iso_match = re.match(r"^(\d+)([A-Za-z]{1,2})$", stripped)
    if iso_match:
        letters = iso_match.group(2)
        return letters[0].upper() + letters[1:].lower()
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


def atomic_mass_amu(label: str, *, mass_convention: str = "iupac") -> float:
    """Return the default mass for an element, isotope, or labelled atom.

    Parameters
    ----------
    label:
        Element symbol (``"C"``), atom label (``"Cl1"``), or isotope symbol (``"13C"``, ``"37Cl"``).
    mass_convention:
        ``"iupac"`` for standard atomic weights or ``"polyrate"`` for Polyrate 2023
        monoisotopic reference masses.
    """

    stripped = label.strip()
    iso_match = re.match(r"^(\d+)([A-Za-z]{1,2})$", stripped)
    if iso_match:
        iso_key = f"{iso_match.group(1)}{iso_match.group(2)[0].upper()}{iso_match.group(2)[1:].lower()}"
        if iso_key in ISOTOPE_MASSES_AMU:
            if mass_convention.lower() == "polyrate" and iso_key == "2H":
                return POLYRATE_ATOMIC_MASSES_AMU["D"]
            return ISOTOPE_MASSES_AMU[iso_key]

    symbol = canonical_symbol(label)
    convention = mass_convention.lower()
    if convention == "polyrate":
        table = POLYRATE_ATOMIC_MASSES_AMU
    elif convention == "iupac":
        table = ATOMIC_MASSES_AMU
    else:
        raise ValueError(f"mass_convention must be 'iupac' or 'polyrate', got {mass_convention!r}")
    try:
        return table[symbol]
    except KeyError as exc:
        raise ValueError(
            f"no built-in mass for {label!r}; provide mass_amu explicitly"
        ) from exc

