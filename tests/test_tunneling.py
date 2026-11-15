from __future__ import annotations

import math

from pyrate.constants import AMU_KG, ANGSTROM_M, AVOGADRO_MOL
from pyrate.models import PathPoint, Reaction, ReactionPath, Species, TunnelingProfile
from pyrate.native import thermal_wkb_kappa, wigner_kappa
from pyrate.tunneling import (
    cag_kappa,
    path_tunneling_profile,
    profile_muomt_kappa,
    profile_transmission_probability,
    profile_wkb_kappa,
    reaction_eckart_kappa,
    transition_state_wigner_kappa,
)


def test_wigner_wrapper() -> None:
    ts = Species(
        name="ts",
        mass_amu_override=10.0,
        rotor_type="atomic",
        frequencies_cm1=(500.0, -1000.0),
    )
    assert math.isclose(
        transition_state_wigner_kappa(ts, 300.0),
        wigner_kappa(1000.0, 300.0),
        rel_tol=1.0e-14,
    )


def test_flat_profile_has_unit_correction() -> None:
    coordinates = (-ANGSTROM_M, 0.0, ANGSTROM_M)
    potentials = (0.0, 0.0, 0.0)
    assert thermal_wkb_kappa(coordinates, potentials, 300.0, AMU_KG, 100) == 1.0


def test_barrier_profile_has_tunneling_enhancement() -> None:
    profile = TunnelingProfile(
        coordinates_angstrom=(-1.0, -0.5, 0.0, 0.5, 1.0),
        energies_kj_mol=(0.0, 12.0, 25.0, 12.0, 0.0),
        reduced_mass_amu=1.0,
        energy_grid_points=200,
    )
    low = profile_transmission_probability(profile, 1.0)
    high = profile_transmission_probability(profile, 20.0)
    top = profile_transmission_probability(profile, 25.0)
    above = profile_transmission_probability(profile, 30.0)
    assert 0.0 <= low < high < top < above <= 1.0
    # Kemble uniform semiclassical formula gives P(V_max) = 1 / (1 + exp(0)) = 0.5
    assert math.isclose(top, 0.5, rel_tol=1.0e-12)

    energies_j = tuple(value * 1000.0 / AVOGADRO_MOL for value in profile.energies_kj_mol)
    kappa = thermal_wkb_kappa(
        tuple(value * ANGSTROM_M for value in profile.coordinates_angstrom),
        energies_j,
        300.0,
        AMU_KG,
        200,
    )
    assert kappa > 1.0
    assert math.isfinite(kappa)


def test_sct_curvature_reduces_effective_mass_and_enhances_muomt() -> None:
    def _surf(name: str, energy: float, freq: float, imag: float | None = None) -> Species:
        freqs = (freq,) if imag is None else (freq, -abs(imag))
        return Species(
            name=name,
            mass_amu_override=15.0,
            rotor_type="linear",
            moments_of_inertia_amu_a2=(12.0,),
            electronic_energy_kj_mol=energy,
            frequencies_cm1=freqs,
        )

    coords = (-1.0, -0.5, 0.0, 0.5, 1.0)
    energies = (0.0, 18.0, 35.0, 20.0, 2.0)
    curvs = ((0.25,), (0.55,), (0.75,), (0.50,), (0.20,))
    points = tuple(
        PathPoint(
            coordinate=s,
            dividing_surface=_surf(f"pt-{i}", e, 1200.0, 1100.0 if s == 0.0 else None),
            curvatures_bohr_inv=b,
        )
        for i, (s, e, b) in enumerate(zip(coords, energies, curvs, strict=True))
    )
    path = ReactionPath(points=points, coordinate_unit="bohr", reduced_mass_amu=1.0)
    zct_prof = path_tunneling_profile(path, method="zct")
    sct_prof = path_tunneling_profile(path, method="sct")
    assert zct_prof.effective_masses_amu is None
    assert sct_prof.effective_masses_amu is not None
    assert any(m < 1.0 for m in sct_prof.effective_masses_amu)

    kappa_zct = profile_wkb_kappa(zct_prof, 300.0)
    kappa_sct = profile_wkb_kappa(sct_prof, 300.0)
    kappa_muomt = profile_muomt_kappa(zct_prof, sct_prof, 300.0)
    assert kappa_sct > kappa_zct > 1.0
    assert kappa_muomt >= max(kappa_zct, kappa_sct) - 1.0e-12


def test_cag_and_reaction_eckart_kappa() -> None:
    r = Species(name="R", mass_amu_override=10.0, rotor_type="atomic", electronic_energy_kj_mol=0.0)
    p = Species(name="P", mass_amu_override=10.0, rotor_type="atomic", electronic_energy_kj_mol=-5.0)
    ts = Species(
        name="TS",
        mass_amu_override=10.0,
        rotor_type="atomic",
        electronic_energy_kj_mol=30.0,
        frequencies_cm1=(1000.0, -1200.0),
    )
    pts = (
        PathPoint(-0.5, Species(name="p1", mass_amu_override=10.0, rotor_type="atomic", electronic_energy_kj_mol=15.0, frequencies_cm1=(1000.0,))),
        PathPoint(0.0, ts),
        PathPoint(0.3, Species(name="p2", mass_amu_override=10.0, rotor_type="atomic", electronic_energy_kj_mol=32.0, frequencies_cm1=(1000.0,))),
        PathPoint(0.8, Species(name="p3", mass_amu_override=10.0, rotor_type="atomic", electronic_energy_kj_mol=10.0, frequencies_cm1=(1000.0,))),
    )
    rxn = Reaction(reactants=(r,), transition_state=ts, products=(p,), path=ReactionPath(pts))
    cag_0 = cag_kappa(rxn, 300.0, variational_coordinate=0.0)
    assert cag_0 < 1.0
    eck = reaction_eckart_kappa(rxn, 300.0)
    assert eck > 1.0

