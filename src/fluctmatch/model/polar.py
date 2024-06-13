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
# pyright: reportInvalidTypeVarUse=false, reportOptionalIterable=false, reportArgumentType=false
# pyright: reportAttributeAccessIssue=false
"""Class definition for beads using N, carboxyl oxygens, and polar sidechains."""

from types import MappingProxyType
from typing import ClassVar, Self

import MDAnalysis as mda

from fluctmatch.model.ncsc import NcscModel


class PolarModel(NcscModel):
    """Universe consisting of the amine, carboxyl, and polar regions.

    For each residue, two beads will be assigned for the backbone and one bead for the sidechain (excluding glycine).
    The amino bead will be positioned at the amino nitrogen with amino hydrogens and half the C-alpha and H-alpha atoms
    contributing to the mass and charge. The other backbone bead corresponds to the carboxyl oxygen; the position will
    rely upon the center-of-mass or center-of-geometry between oxygens (particularly at the C-terminux), and the
    carboxyl carbon and half the C-alpha and H-alpha atoms will account for the mass and charge of the bead. The C-beta
    position (excluding glycine) will be determined by the center-of-mass or center-of-geometry of the specific atoms of
    each sidechain, and the mass and charge will be the total for each sidechain. Three bonds will exist per residue as
    well as a single bond between the carboxyl bead and the amino bead in the :math:`i,i+1`. Additionally, common ions
    found as cofactors within proteins (i.e., MG CAL MN FE CU ZN  AG) will also be included but have no bonds associated
    with them.

      SC
     /  \
    N - C - N
    """

    model: ClassVar[str] = "POLAR"
    description: ClassVar[str] = "c.o.m./c.o.g. of N, C, and polar sidechains of protein"

    def __init__(self: Self, mobile: mda.Universe, /, **kwargs: dict[str, bool]) -> None:
        super().__init__(mobile, **kwargs)

        mapping = self._mapping.copy()
        mapping["CB"] = MappingProxyType({
            "ALA": "protein and name CB",
            "ARG": "protein and name NH*",
            "ASN": "protein and name OD1 ND2",
            "ASP": "protein and name OD*",
            "CYS": "protein and name SG",
            "GLN": "protein and name OE1 NE2",
            "GLU": "protein and name OE*",
            "GLY": "",
            "HIS": "protein and name CG ND1 CD2 CE1 NE2",
            "HSD": "protein and name CG ND1 CD2 CE1 NE2",
            "HSE": "protein and name CG ND1 CD2 CE1 NE2",
            "HSP": "protein and name CG ND1 CD2 CE1 NE2",
            "ILE": "protein and name CG1 CG2 CD",
            "LEU": "protein and name CD1 CD2",
            "LYS": "protein and name NZ",
            "MET": "protein and name SD",
            "PHE": "protein and name CG CD* CE* CZ",
            "PRO": "protein and name CG",
            "SER": "protein and name OG",
            "THR": "protein and name OG1",
            "TRP": "protein and name CG CD* NE CE* CZ* CH",
            "TYR": "protein and name CG CD* CE* CZ OH",
            "VAL": "protein and name CG1 CG2",
        })
        self._mapping = MappingProxyType(mapping)
