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
# pyright: reportInvalidTypeVarUse=false, reportOptionalIterable=false
"""Model for TIP3P water."""

from itertools import combinations
from types import MappingProxyType
from typing import ClassVar, Self

import MDAnalysis as mda
from MDAnalysis.core.topologyattrs import Bonds

from fluctmatch.model.base import CoarseGrainModel


class Tip3pModel(CoarseGrainModel):
    """Create a universe containing all three water atoms."""

    model: ClassVar[str] = "TIP3P"
    description: ClassVar[str] = "All-atom watter"

    def __init__(self: Self, mobile: mda.Universe, /, **kwargs: dict[str, bool]) -> None:
        super().__init__(mobile, **kwargs)

        self._mapping: MappingProxyType[str, str] = MappingProxyType({
            "OW": "water and name OW OH2",
            "HW1": "water and name HW1 H1",
            "HW2": "water and name HW2 H2",
        })
        self._selection: MappingProxyType[str, str] = MappingProxyType({
            "OW": "water and name OW MW OH2",
            "HW1": "water and name HW1 H1",
            "HW2": "water and name HW2 H2",
        })
        self._types: MappingProxyType[str, int] = MappingProxyType({
            key: value + 1 for key, value in zip(self._mapping.keys(), range(len(self._mapping)), strict=True)
        })

    def _add_bonds(self: Self, rmin: float, rmax: float) -> None:  # noqa: ARG002
        bonds: list[tuple[int, int]] = []
        for segment in self._universe.segments:
            for select1, select2 in combinations(self._selection.keys(), 2):
                atom1 = self._selection.get(select1)
                atom2 = self._selection.get(select2)
                bonds.extend(
                    tuple(zip(segment.atoms.select_atoms(atom1).ix, segment.atoms.select_atoms(atom2).ix, strict=True))
                )

        self._universe.add_TopologyAttr(Bonds(bonds))
