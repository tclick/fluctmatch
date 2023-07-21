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
# pyright: reportInvalidTypeVarUse=false
# flake8: noqa
"""Class for 6-bead nucleic acid."""

from types import MappingProxyType
from typing import ClassVar, TypeVar

from MDAnalysis.core.topologyattrs import Bonds

from ..base import ModelBase
from ..selection import *

TModel = TypeVar("TModel", bound="Model")


class Model(ModelBase):
    """A universe accounting for six sites involved with hydrogen bonding."""

    model: ClassVar[str] = "NUCLEIC6"
    description: ClassVar[str] = "Phosphate, C2', C4', and 3 sites on the nucleotide"

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

        self._mapping: MappingProxyType[str, str | tuple[str, ...]] = MappingProxyType(
            {
                "P": "nucleicphosphate and not name H*",
                "C4": "name C4'",
                "C2": "name C2'",
                "H1": (
                    "(resname ADE DA* RA* and name N6) or "
                    "(resname OXG GUA DG* RG* and name O6) or "
                    "(resname CYT DC* RC* and name N4) or "
                    "(resname THY URA DT* RU* and name O4)"
                ),
                "H2": (
                    "(resname ADE DA* RA* OXG GUA DG* RG* and name N1) or "
                    "(resname CYT DC* RC* THY URA DT* RU* and name N3)"
                ),
                "H3": (
                    "(resname ADE DA* RA* and name H2) or "
                    "(resname OXG GUA DG* RG* and name N2) or "
                    "(resname CYT DC* RC* THY URA DT* RU* and name O2)"
                ),
            }
        )
        self._selection: MappingProxyType[str, str] = MappingProxyType(
            {
                "P": "nucleicphosphate",
                "C4": "sugarC4",
                "C2": "sugarC2",
                "H1": "nucleic and name C2'",
                "H2": "nucleic and name C2'",
                "H3": "nucleic and name C2'",
            }
        )

    def _add_bonds(self: TModel) -> None:
        bonds: list[tuple[int, int]] = []
        for segment in self._universe.segments:  # type: ignore
            atom1 = segment.atoms.select_atoms("name P")
            atom2 = segment.atoms.select_atoms("name C4")
            atom3 = segment.atoms.select_atoms("name C2")
            atom4 = segment.atoms.select_atoms("name H1")
            atom5 = segment.atoms.select_atoms("name H2")
            atom6 = segment.atoms.select_atoms("name H3")

            bonds.extend(tuple(zip(atom1.ix, atom2.ix, strict=True)))
            bonds.extend(tuple(zip(atom2.ix, atom3.ix, strict=True)))
            bonds.extend(tuple(zip(atom3.ix, atom4.ix, strict=True)))
            bonds.extend(tuple(zip(atom4.ix, atom5.ix, strict=True)))
            bonds.extend(tuple(zip(atom5.ix, atom6.ix, strict=True)))
            bonds.extend(tuple(zip(atom2.ix[:-1], atom1.ix[1:], strict=True)))

        self._universe.add_TopologyAttr(Bonds(bonds))
