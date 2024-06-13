# ---------------------------------------------------------------------------------------------------------------------
# fluctmatch
# Copyright (c) 2013-2024 Timothy H. Click, Ph.D.
#
# This file is part of fluctmatch.
#
# Fluctmatch is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# Fluctmatch is distributed in the hope that it will be useful, # but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <[1](https://www.gnu.org/licenses/)>.
#
# Reference:
# Timothy H. Click, Nixon Raj, and Jhih-Wei Chu. Simulation. Meth Enzymology. 578 (2016), 327-342,
# Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics doi:10.1016/bs.mie.2016.05.024.
# ---------------------------------------------------------------------------------------------------------------------
# pyright: reportInvalidTypeVarUse = false, reportOptionalIterable = false, reportArgumentType = false
# pyright: reportAttributeAccessIssue = false
"""Class definition for beads using N, carboxyl oxygens, and sidechains."""

from types import MappingProxyType
from typing import ClassVar, Self

import MDAnalysis as mda
from MDAnalysis.core.topologyattrs import Bonds

from fluctmatch.model.base import CoarseGrainModel


class NcscModel(CoarseGrainModel):
    """Universe consisting of the amine, carboxyl, and sidechain regions.

    For each residue, two beads will be assigned for the backbone and one bead for the sidechain (excluding glycine).
    The amino bead will be positioned at the amino nitrogen with amino hydrogens and half the C-alpha and H-alpha atoms
    contributing to the mass and charge. The other backbone bead corresponds to the carboxyl oxygen; the position will
    rely upon the center-of-mass or center-of-geometry between oxygens (particularly at the C-terminux), and the
    carboxyl carbon and half the C-alpha and H-alpha atoms will account for the mass and charge of the bead. The C-beta
    position (excluding glycine) will be determined by the center-of-mass or center-of-geometry of the heavy atoms of
    each sidechain, and the mass and charge will be the total for each sidechain. Three bonds will exist per residue as
    well as a single bond between the carboxyl bead and the amino bead in the :math:`i,i+1`. Additionally, common ions
    found as cofactors within proteins (i.e., MG CAL MN FE CU ZN  AG) will also be included but have no bonds associated
    with them.

      SC
     /  \
    N - C - N
    """

    model: ClassVar[str] = "NCSC"
    description: ClassVar[str] = "c.o.m./c.o.g. of N, O, and sidechain of protein"

    def __init__(self: Self, mobile: mda.Universe, /, **kwargs: dict[str, bool]) -> None:
        super().__init__(mobile, **kwargs)

        self._mapping: MappingProxyType[str, str | MappingProxyType[str, str]] = MappingProxyType({
            "N": "protein and name N",
            "CB": "hsidechain and not name H*",
            "O": "protein and name O OT1 OT2 OXT",
        })
        self._selection: MappingProxyType[str, str] = MappingProxyType({
            "N": "amine",
            "CB": "hsidechain",
            "O": "carboxyl",
        })

    def _add_bonds(self: Self, rmin: float, rmax: float) -> None:  # noqa: ARG002
        bonds: list[tuple[int, int]] = []

        # Create bonds intraresidue atoms
        residues: mda.ResidueGroup = self._universe.select_atoms("protein").residues
        atom1a: mda.AtomGroup = residues.atoms.select_atoms("name N")
        atom2a: mda.AtomGroup = residues.atoms.select_atoms("name O")
        bonds.extend(tuple(zip(atom1a.ix, atom2a.ix, strict=True)))

        residues = self._universe.select_atoms("protein and not resname GLY").residues
        atom1b: mda.AtomGroup = residues.atoms.select_atoms("name N")
        atom2b: mda.AtomGroup = residues.atoms.select_atoms("name O")
        atom3: mda.AtomGroup = residues.atoms.select_atoms("cbeta")
        bonds.extend(tuple(zip(atom1b.ix, atom3.ix, strict=True)))
        bonds.extend(tuple(zip(atom2b.ix, atom3.ix, strict=True)))

        # Create interresidue bonds
        for segment in self._universe.segments:
            atom1c: mda.AtomGroup = segment.atoms.select_atoms("name O")
            atom2c: mda.AtomGroup = segment.atoms.select_atoms("name N")
            bonds.extend(tuple(zip(atom1c.ix[:-1], atom2c.ix[1:], strict=True)))  # noqa: PD007

        self._universe.add_TopologyAttr(Bonds(bonds))

    def _add_masses(self: Self) -> None:
        super()._add_masses()
        for cg, aa in zip(self._universe.residues, self._mobile.residues, strict=False):
            amine: mda.AtomGroup = cg.atoms.select_atoms(self._mapping["N"])
            carboxyl: mda.AtomGroup = cg.atoms.select_atoms(self._mapping["O"])
            ca_mass: float = 0.5 * aa.atoms.select_atoms("hcalpha").total_mass()
            amine.masses += ca_mass
            carboxyl.masses += ca_mass

    def _add_charges(self: Self) -> None:
        super()._add_charges()
        for cg, aa in zip(self._universe.residues, self._mobile.residues, strict=False):
            amine: mda.AtomGroup = cg.atoms.select_atoms(self._mapping["N"])
            carboxyl: mda.AtomGroup = cg.atoms.select_atoms(self._mapping["O"])
            ca_charge: float = 0.5 * aa.atoms.select_atoms("hcalpha").total_charge()
            amine.charges += ca_charge
            carboxyl.charges += ca_charge
