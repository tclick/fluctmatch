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

        self._mapping: MappingProxyType[str, str] = MappingProxyType({"CA": "calpha", "ions": "bioion"})
        self._selection: MappingProxyType[str, str] = MappingProxyType({"CA": "protein", "ions": "bioion"})

    def _add_bonds(self: Self) -> None:
        bonds: list[tuple[int, int]] = []

        # Create bonds between C-alphas in adjacent residues
        for segment in self._universe.segments:
            atom_selection: str = self._mapping[next(iter(self._mapping.keys()))]
            atoms: mda.AtomGroup = segment.atoms.select_atoms(atom_selection)
            bonds.extend(tuple(zip(atoms.ix[1:], atoms.ix[:-1], strict=True)))  # noqa: PD007
        self._universe.add_TopologyAttr(Bonds(bonds))
