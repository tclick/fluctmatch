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
"""Elastic network model using C-alpha atoms of a protein."""

from types import MappingProxyType
from typing import ClassVar, Self

import MDAnalysis as mda
from MDAnalysis.core.topologyattrs import Bonds

from fluctmatch.model.base import CoarseGrainModel


class CalphaModel(CoarseGrainModel):
    """Universe defined by the protein C-alpha.

    The C-alpha will be selected within the protein. The total charge and mass of each residue will be assigned to the
    corresponding carbon, and bonds will be generated between :math:`i,i+1` residues. Additionally, common ions found
    as cofactors within proteins (i.e., MG CAL MN FE CU ZN AG) will also be included but have no bonds associated with
    them.
    """

    model: ClassVar[str] = "CALPHA"
    description: ClassVar[str] = "C-alpha of a protein"

    def __init__(self: Self, mobile: mda.Universe, /, **kwargs: dict[str, bool]) -> None:
        super().__init__(mobile, **kwargs)

        self._mapping: MappingProxyType[str, str] = MappingProxyType({"CA": "calpha"})
        self._selection: MappingProxyType[str, str] = MappingProxyType({"CA": "protein"})

    def _add_bonds(self: Self, rmin: float, rmax: float) -> None:  # noqa: ARG002
        bonds: list[tuple[int, int]] = []

        # Create bonds between C-alphas in adjacent residues
        for segment in self._universe.segments:
            atom_selection: str = self._mapping[next(iter(self._mapping.keys()))]
            atoms: mda.AtomGroup = segment.atoms.select_atoms(atom_selection)
            bonds.extend(tuple(zip(atoms.ix[1:], atoms.ix[:-1], strict=True)))  # noqa: PD007
        self._universe.add_TopologyAttr(Bonds(bonds))
