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
"""Class definition for beads using N, carboxyl oxygens, and sidechains."""

from types import MappingProxyType
from typing import TypeVar, ClassVar

import MDAnalysis as mda
from MDAnalysis import AtomGroup, ResidueGroup

from MDAnalysis.core.topologyattrs import Bonds

from ..base import ModelBase
from ..selection import *

TModel = TypeVar("TModel", bound="Model")


class Model(ModelBase):
    """Universe consisting of the amine, carboxyl, and sidechain regions."""

    model: ClassVar[str] = "NCSC"
    description: ClassVar[str] = "c.o.m./c.o.g. of N, O, and sidechain of protein"

    def __init__(self: TModel, *, com: bool = True, guess_angles: bool = False) -> None:
        super().__init__(com=com, guess_angles=guess_angles)

        self._mapping: MappingProxyType[str, str | MappingProxyType[str, str]] = MappingProxyType(
            {
                "N": "protein and name N",
                "CB": "hsidechain and not name H*",
                "O": "protein and name O OT1 OT2 OXT",
                "ions": "bioion",
            }
        )
        self._selection: MappingProxyType[str, str] = MappingProxyType(
            {"N": "amine", "CB": "hsidechain", "O": "carboxyl", "ions": "bioion"}
        )

    def _add_bonds(self: TModel) -> None:
        bonds: list[tuple[int, int]] = []

        # Create bonds intraresidue atoms
        residues: ResidueGroup = self._universe.select_atoms("protein").residues
        atom1: AtomGroup = residues.atoms.select_atoms("name N")
        atom2: AtomGroup = residues.atoms.select_atoms("name O")
        bonds.extend(tuple(zip(atom1.ix, atom2.ix, strict=True)))

        residues = self._universe.select_atoms("protein and not resname GLY").residues
        atom1: AtomGroup = residues.atoms.select_atoms("name N")
        atom2: AtomGroup = residues.atoms.select_atoms("name O")
        atom3: AtomGroup = residues.atoms.select_atoms("cbeta")
        bonds.extend(tuple(zip(atom1.ix, atom3.ix, strict=True)))
        bonds.extend(tuple(zip(atom2.ix, atom3.ix, strict=True)))

        # Create interresidue bonds
        for segment in self._universe.segments:  # type: ignore
            atom1: AtomGroup = segment.atoms.select_atoms("name O")
            atom2: AtomGroup = segment.atoms.select_atoms("name N")
            bonds.extend(tuple(zip(atom1.ix[:-1], atom2.ix[1:], strict=True)))

        self._universe.add_TopologyAttr(Bonds(bonds))

    def _add_masses(self: TModel, universe: mda.Universe) -> None:
        super()._add_masses(universe)
        for cg, aa in zip(self._universe.residues, universe.residues, strict=False):  # type: ignore
            amine: AtomGroup = cg.atoms.select_atoms(self._mapping["N"])
            carboxyl: AtomGroup = cg.atoms.select_atoms(self._mapping["O"])
            if aa.atoms.select_atoms("hcalpha"):
                ca_mass: float = 0.5 * aa.atoms.select_atoms("hcalpha").total_mass()
                amine.masses += ca_mass  # type: ignore
                carboxyl.masses += ca_mass  # type: ignore

    def _add_charges(self: TModel, universe: mda.Universe) -> None:
        super()._add_charges(universe)
        for cg, aa in zip(self._universe.residues, universe.residues, strict=False):  # type: ignore
            amine: AtomGroup = cg.atoms.select_atoms(self._mapping["N"])
            carboxyl: AtomGroup = cg.atoms.select_atoms(self._mapping["O"])
            if aa.atoms.select_atoms("hcalpha"):
                ca_charge: float = 0.5 * aa.atoms.select_atoms("hcalpha").total_charge()
                amine.charges += ca_charge  # type: ignore
                carboxyl.charges += ca_charge  # type: ignore
