# ------------------------------------------------------------------------------
#  fluctmatch
#  Copyright (c) 2023 Timothy H. Click, Ph.D.
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
# pyright: reportInvalidTypeVarUse=false, reportOptionalMemberAccess=false, reportGeneralTypeIssues=false
# pyright: reportOptionalIterable=false
# flake8: noqa
"""Class definition for beads using C-alpha and C-beta positions"""

from types import MappingProxyType
from typing import TypeVar, ClassVar

from MDAnalysis import AtomGroup, ResidueGroup
from MDAnalysis.core.topologyattrs import Bonds

from ..base import ModelBase
from ..selection import *

TModel = TypeVar("TModel", bound="Model")


class Model(ModelBase):
    """Universe consisting of the C-alpha and sidechains of a protein."""

    model: ClassVar[str] = "CASIDE"
    description: ClassVar[str] = "C-alpha and sidechain (c.o.m./c.o.g.) of protein"

    def __init__(
        self: TModel,
        *,
        xplor: bool = True,
        extended: bool = True,
        com: bool = True,
        guess_angles: bool = False,
        rmin: float = 0.0,
        rmax: float = 10.0,
    ) -> None:
        super().__init__(
            xplor=xplor,
            extended=extended,
            com=com,
            guess_angles=guess_angles,
            rmin=rmin,
            rmax=rmax,
        )

        self._mapping: MappingProxyType[str, str] = MappingProxyType(
            {"CA": "calpha", "CB": "hsidechain and not name H*", "ions": "bioion"}
        )
        self._selection: MappingProxyType[str, str] = MappingProxyType(
            {"CA": "hbackbone", "CB": "hsidechain", "ions": "bioion"}
        )

    def _add_bonds(self: TModel) -> None:
        bonds: list[tuple[int, int]] = []

        # Create bonds intraresidue C-alpha and C-beta atoms.
        residues: ResidueGroup = self._universe.select_atoms("protein and not resname GLY").residues
        atom1: AtomGroup = residues.atoms.select_atoms("calpha")
        atom2: AtomGroup = residues.atoms.select_atoms("cbeta")
        bonds.extend(tuple(zip(atom1.ix, atom2.ix, strict=True)))

        # Create interresidue C-alpha bonds within a segment
        for segment in self._universe.segments:
            atoms: AtomGroup = segment.atoms.select_atoms("calpha")
            bonds.extend(tuple(zip(atoms.ix[1:], atoms.ix[:-1], strict=True)))

        self._universe.add_TopologyAttr(Bonds(bonds))
