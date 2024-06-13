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
"""Class for a 3-bead nucleic acid."""

from types import MappingProxyType
from typing import ClassVar, Self

import MDAnalysis as mda
from MDAnalysis.core.topologyattrs import Bonds

from fluctmatch.model.base import CoarseGrainModel


class NucleicModel(CoarseGrainModel):
    """A universe of the phosphate, C4', C3', and base of the nucleic acid."""

    model: ClassVar[str] = "NUCLEIC4"
    description: ClassVar[str] = "Phosphate, C2', C4', and c.o.m./c.o.g. of C4/C5 of nucleic acid"

    def __init__(self: Self, mobile: mda.Universe, /, **kwargs: dict[str, bool]) -> None:
        super().__init__(mobile, **kwargs)

        self._mapping: MappingProxyType[str, str] = MappingProxyType({
            "P": "nucleicphosphate and not name H*",
            "C4": "name C4'",
            "C2'": "name C2'",
            "C5": "nucleiccenter and not name H*",
        })
        self._selection: MappingProxyType[str, str] = MappingProxyType({
            "P": "nucleicphosphate",
            "C4": "sugarC4",
            "C2'": "sugarC2",
            "C5": "hnucleicbase",
        })

    def _add_bonds(self: Self, rmin: float, rmax: float) -> None:  # noqa: ARG002
        bonds: list[tuple[int, int]] = []
        for segment in self._universe.segments:
            atom1 = segment.atoms.select_atoms("name P")
            atom2 = segment.atoms.select_atoms("name C4")
            atom3 = segment.atoms.select_atoms("name C2'")
            atom4 = segment.atoms.select_atoms("name C5")

            bonds.extend(tuple(zip(atom1.ix, atom2.ix, strict=True)))
            bonds.extend(tuple(zip(atom2.ix, atom3.ix, strict=True)))
            bonds.extend(tuple(zip(atom2.ix, atom4.ix, strict=True)))
            bonds.extend(tuple(zip(atom2.ix[:-1], atom1.ix[1:], strict=True)))  # noqa: PD007

        self._universe.add_TopologyAttr(Bonds(bonds))
