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
# pyright: reportInvalidTypeVarUse = false, reportOptionalIterable = false
"""Class definition for beads using C-alpha and C-beta positions."""

from types import MappingProxyType
from typing import ClassVar, Self

import MDAnalysis as mda
from MDAnalysis.core.topologyattrs import Bonds

from fluctmatch.model.base import CoarseGrainModel


class CasideModel(CoarseGrainModel):
    """Universe consisting of the C-alpha and sidechains of a protein.

    The C-alpha and C-beta atoms will be selected within the protein. The total charge and mass of each residue backbone
    will be assigned to the C-alpha atom, and the total charge and mass of each residue sidechain will be assigned to
    the C-beta atom. The C-alpha position will remain identical to the original protein C-alpha position, but the C-beta
    position (excluding glycine) will be determined by the center-of-mass or center-of-geometry of the heavy atoms of
    each sidechain. Bonds will be generated between :math:`i,i+1` between C-alpha atoms and within each residue between
    the C-alpha and C-beta beads. Additionally, common ions found as cofactors within proteins (i.e., MG CAL MN FE CU ZN
    AG) will also be included but have no bonds associated with them.
    """

    model: ClassVar[str] = "CASIDE"
    description: ClassVar[str] = "C-alpha and sidechain (c.o.m./c.o.g.) of protein"

    def __init__(self: Self, mobile: mda.Universe, /, **kwargs: dict[str, bool]) -> None:
        super().__init__(mobile, **kwargs)

        self._mapping: MappingProxyType[str, str] = MappingProxyType({
            "CA": "calpha",
            "CB": "hsidechain and not name H*",
        })
        self._selection: MappingProxyType[str, str] = MappingProxyType({
            "CA": "hbackbone",
            "CB": "hsidechain",
        })

    def _add_bonds(self: Self, rmin: float, rmax: float) -> None:  # noqa: ARG002
        bonds: list[tuple[int, int]] = []

        # Create bonds intraresidue C-alpha and C-beta atoms.
        residues: mda.ResidueGroup = self._universe.select_atoms("protein and not resname GLY").residues
        atom1: mda.AtomGroup = residues.atoms.select_atoms("calpha")
        atom2: mda.AtomGroup = residues.atoms.select_atoms("cbeta")
        bonds.extend(tuple(zip(atom1.ix, atom2.ix, strict=True)))

        # Create interresidue C-alpha bonds within a segment
        for segment in self._universe.segments:
            atoms: mda.AtomGroup = segment.atoms.select_atoms("calpha")
            bonds.extend(tuple(zip(atoms.ix[1:], atoms.ix[:-1], strict=True)))  # noqa: PD007

        self._universe.add_TopologyAttr(Bonds(bonds))
