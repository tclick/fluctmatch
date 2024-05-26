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
"""Select ions typically found as cofactors within proteins."""

from collections.abc import MutableMapping
from types import MappingProxyType
from typing import ClassVar, Self

import MDAnalysis as mda
import numpy as np
from MDAnalysis.core.topologyattrs import Atomtypes, Bonds
from numpy.typing import NDArray

from fluctmatch.model.base import CoarseGrainModel


class BioionModel(CoarseGrainModel):
    """Select ions normally found within biological systems."""

    model: ClassVar[str] = "BIOIONS"
    description: ClassVar[str] = "Common ions found near proteins (Mg Ca Mn Fe Cu Zn Ag)"

    def __init__(self: Self, mobile: mda.Universe, /, **kwargs: dict[str, bool]) -> None:
        super().__init__(mobile, **kwargs)

        self._guess = False
        self._mapping: MappingProxyType[str, str] = MappingProxyType({"ions": "bioion"})
        self._selection: MappingProxyType[str, str] = self._mapping

    def _add_atomtypes(self: Self) -> None:
        resnames: NDArray = np.unique(self._universe.residues.resnames)
        restypes: MutableMapping[str, int] = dict(zip(resnames, np.arange(resnames.size) + 20, strict=False))

        atomtypes: list[int] = [restypes[atom.name] for atom in self._universe.atoms]
        self._universe.add_TopologyAttr(Atomtypes(atomtypes))

    def _add_bonds(self: Self, rmin: float, rmax: float) -> None:  # noqa: ARG002
        self._universe.add_TopologyAttr(Bonds([]))
