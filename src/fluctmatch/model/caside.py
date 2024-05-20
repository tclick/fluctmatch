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
            "ions": "bioion",
        })
        self._selection: MappingProxyType[str, str] = MappingProxyType({
            "CA": "hbackbone",
            "CB": "hsidechain",
            "ions": "bioion",
        })

    def _add_bonds(self: Self) -> None:
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
