# ------------------------------------------------------------------------------
#  fluctmatch
#  Copyright (c) 2013-2024 Timothy H. Click, Ph.D.
#
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
#  Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  Neither the name of the author nor the names of its contributors may be used
#  to endorse or promote products derived from this software without specific
#  prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
#  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
#  OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
#  DAMAGE.
# ------------------------------------------------------------------------------
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
