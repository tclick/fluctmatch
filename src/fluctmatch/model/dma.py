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
# pyright: reportInvalidTypeVarUse=false, reportGeneralTypeIssues=false, reportOptionalMemberAccess=false
# pyright: reportOptionalIterable=false
"""Tests for dimethylamide (DMA) solvent model."""

from types import MappingProxyType
from typing import ClassVar, Self

import MDAnalysis as mda
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.core.topologyattrs import Bonds

from fluctmatch.model.base import CoarseGrainModel


class DmaModel(CoarseGrainModel):
    """Create a universe for N-dimethylacetamide."""

    model: ClassVar[str] = "DMA"
    description: ClassVar[str] = "c.o.m./c.o.g. of C1, N, C2, and C3 of DMA"

    def __init__(self: Self, mobile: mda.Universe, /, **kwargs: dict[str, bool]) -> None:
        super().__init__(mobile, **kwargs)

        self._mapping: MappingProxyType[str, str] = MappingProxyType({
            "C1": "resname DMA and name C1",
            "N": "resname DMA and name N",
            "C2": "resname DMA and name C2",
            "C3": "resname DMA and name C3",
        })
        self._selection: MappingProxyType[str, str] = MappingProxyType({
            "C1": "resname DMA and name C1 H1*",
            "N": "resname DMA and name C N O",
            "C2": "resname DMA and name C2 H2*",
            "C3": "resname DMA and name C3 H3*",
        })

        self._types: MappingProxyType[str, int] = MappingProxyType({
            key: value + 4 for key, value in zip(self._mapping.keys(), range(len(self._mapping)), strict=True)
        })

    def _add_bonds(self: Self, rmin: float, rmax: float) -> None:  # noqa: ARG002
        bonds: list[tuple[int, int]] = []
        for segment in self._universe.segments:
            atom1: AtomGroup = segment.atoms.select_atoms("name C1")
            atom2: AtomGroup = segment.atoms.select_atoms("name N")
            atom3: AtomGroup = segment.atoms.select_atoms("name C2")
            atom4: AtomGroup = segment.atoms.select_atoms("name C3")
            bonds.extend(tuple(zip(atom1.ix, atom2.ix, strict=True)))
            bonds.extend(tuple(zip(atom2.ix, atom3.ix, strict=True)))
            bonds.extend(tuple(zip(atom2.ix, atom4.ix, strict=True)))

        self._universe.add_TopologyAttr(Bonds(bonds))
