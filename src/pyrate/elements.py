"""Atomic masses used when an input atom does not provide an explicit mass.

Two mass models are available (see :data:`MassModel`):

``"isotopic"`` (default)
    The exact atomic mass of the most abundant naturally occurring isotope
    (``1H``, ``12C``, ``35Cl``, ...), as used by electronic-structure programs
    and Polyrate.  ``D`` and ``T`` resolve to ``2H`` and ``3H``.

    Source and edition: AME2020 atomic mass evaluation, M. Wang, W. J. Huang,
    F. G. Kondev, G. Audi and S. Naimi, Chinese Phys. C 45, 030003 (2021),
    DOI 10.1088/1674-1137/abddaf.  Values were taken from the AME2020 table as
    distributed with ``molmass`` 2026.11.1 (``3H`` from ``periodictable``
    2.1.0, which ``molmass`` does not list).  The isotope chosen for each
    element is the one with the largest CIAAW natural abundance.

    Independent cross-check (2026-11-04; none of these packages is a PyRate
    dependency):

    * ``periodictable`` 2.1.0 (AME2020 ``massround.mas20``, CIAAW 2021
      abundances): same isotope for every element, max |Δm| = 4.1e-7 u
      (``127I``; that table rounds to the mass uncertainty).
    * NUBASE2020 ``nubase_4.mas20.txt`` (F. G. Kondev et al., Chinese Phys.
      C 45, 030001 (2021)), mass excesses converted with
      1 u = 931494.10242 keV and ``IS=`` abundances: same isotope for every
      element, max |Δm| = 2.3e-7 u (``127I``).
    * NIST SRD 144 (2011 IUPAC-based table, as bundled in ``qcelemental``
      0.51.4): same isotope for every element; its older mass evaluation
      differs by at most 6.6e-6 u (``139La``) and by 3.3e-10 u for ``1H``.

    ``tests/test_elements.py`` repeats the ``periodictable`` and ``molmass``
    comparisons (≤1e-6 u) whenever those packages are installed.

``"average"``
    The conventional atomic weights used by PyRate before the isotopic model
    became the default.  The table is kept unchanged for reproducibility.
"""

from __future__ import annotations

import re
from typing import Literal

MassModel = Literal["isotopic", "average"]
"""How a missing atom mass is filled in: most abundant isotope or atomic weight."""

MASS_MODELS: tuple[MassModel, ...] = ("isotopic", "average")
DEFAULT_MASS_MODEL: MassModel = "isotopic"

# Conventional atomic weights (or representative isotope masses for elements
# without a standard atomic weight), in unified atomic mass units.  This is
# the "average" mass model.  The table intentionally covers the elements most
# often encountered in kinetics input; users can always override a mass
# explicitly.
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

# (mass number, AME2020 atomic mass in u) of the most abundant isotope of each
# element in ATOMIC_MASSES_AMU; D and T are 2H and 3H.  See the module
# docstring for the source and the cross-check.
_MOST_ABUNDANT_ISOTOPES: dict[str, tuple[int, float]] = {
    "H": (1, 1.0078250319),
    "D": (2, 2.01410177784),
    "T": (3, 3.01604928132),
    "He": (4, 4.00260325413),
    "Li": (7, 7.01600343426),
    "Be": (9, 9.012183062),
    "B": (11, 11.009305166),
    "C": (12, 12.0),
    "N": (14, 14.00307400425),
    "O": (16, 15.99491461926),
    "F": (19, 18.99840316207),
    "Ne": (20, 19.99244017525),
    "Na": (23, 22.98976928195),
    "Mg": (24, 23.985041689),
    "Al": (27, 26.981538408),
    "Si": (28, 27.97692653442),
    "P": (31, 30.97376199768),
    "S": (32, 31.97207117354),
    "Cl": (35, 34.968852694),
    "Ar": (40, 39.96238312204),
    "K": (39, 38.96370648482),
    "Ca": (40, 39.96259085),
    "Sc": (45, 44.955907051),
    "Ti": (48, 47.947940677),
    "V": (51, 50.943957664),
    "Cr": (52, 51.940504714),
    "Mn": (55, 54.93804304),
    "Fe": (56, 55.934935537),
    "Co": (59, 58.933193524),
    "Ni": (58, 57.93534165),
    "Cu": (63, 62.929597119),
    "Zn": (64, 63.929141776),
    "Ga": (69, 68.925573528),
    "Ge": (74, 73.92117776),
    "As": (75, 74.921594562),
    "Se": (80, 79.916521761),
    "Br": (79, 78.918337574),
    "Kr": (84, 83.91149772708),
    "Rb": (85, 84.91178973604),
    "Sr": (88, 87.905612253),
    "Y": (89, 88.905838156),
    "Zr": (90, 89.904698755),
    "Nb": (93, 92.90637317),
    "Mo": (98, 97.905403609),
    "Ru": (102, 101.904340312),
    "Rh": (103, 102.905494081),
    "Pd": (106, 105.903480287),
    "Ag": (107, 106.905091509),
    "Cd": (114, 113.903364998),
    "In": (115, 114.903878772),
    "Sn": (120, 119.902202557),
    "Sb": (121, 120.903811353),
    "Te": (130, 129.906222745),
    "I": (127, 126.904472592),
    "Xe": (132, 131.90415508346),
    "Cs": (133, 132.905451958),
    "Ba": (138, 137.905247059),
    "La": (139, 138.906362927),
    "Ce": (140, 139.905448433),
    "W": (184, 183.95093318),
    "Pt": (195, 194.964794325),
    "Au": (197, 196.966570103),
    "Hg": (202, 201.970643604),
    "Pb": (208, 207.976652005),
}

ISOTOPIC_MASSES_AMU: dict[str, float] = {
    symbol: mass for symbol, (_, mass) in _MOST_ABUNDANT_ISOTOPES.items()
}
"""Exact mass in u of the most abundant isotope: the "isotopic" mass model."""

ISOTOPE_MASS_NUMBERS: dict[str, int] = {
    symbol: mass_number for symbol, (mass_number, _) in _MOST_ABUNDANT_ISOTOPES.items()
}
"""Mass number of the isotope behind each entry of :data:`ISOTOPIC_MASSES_AMU`."""

_MASS_TABLES: dict[MassModel, dict[str, float]] = {
    "isotopic": ISOTOPIC_MASSES_AMU,
    "average": ATOMIC_MASSES_AMU,
}


def normalize_mass_model(value: object) -> MassModel:
    """Validate a mass-model name (case-insensitive) and return its canonical form."""

    if isinstance(value, str):
        normalized = value.strip().lower()
        for model in MASS_MODELS:
            if normalized == model:
                return model
    allowed = ", ".join(repr(model) for model in MASS_MODELS)
    raise ValueError(f"mass_model must be one of {allowed}; got {value!r}")


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


def atomic_mass_amu(label: str, model: MassModel = DEFAULT_MASS_MODEL) -> float:
    """Return the default mass in amu for an element or labelled atom.

    ``model="isotopic"`` gives the exact mass of the most abundant isotope
    (AME2020); ``model="average"`` gives the conventional atomic weight.  ``D``
    and ``T`` always denote deuterium and tritium.
    """

    table = _MASS_TABLES[normalize_mass_model(model)]
    symbol = canonical_symbol(label)
    try:
        return table[symbol]
    except KeyError as exc:
        raise ValueError(
            f"no built-in mass for {label!r}; provide mass_amu explicitly"
        ) from exc
