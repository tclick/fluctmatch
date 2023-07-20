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
# pyright: reportInvalidTypeVarUse=false, reportGeneralTypeIssues=false
# flake8: noqa
"""Model a generic system of all atoms."""
import itertools
from types import MappingProxyType
from typing import ClassVar, TypeVar

import MDAnalysis as mda
import numpy as np
from MDAnalysis import transformations
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.topology import guessers
from numpy.typing import NDArray

from .. import base
from ..selection import *

TModel = TypeVar("TModel", bound="Model")


class Model(base.ModelBase):
    """Universe consisting of the amine, carboxyl, and sidechain regions."""

    model: ClassVar[str] = "GENERIC"
    description: ClassVar[str] = "all heavy atoms excluding proteins and nucleic acids"

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
            {"bead": "not (protein or nucleic or bioion or water)"}
        )
        self._selection: MappingProxyType[str, str] = self._mapping

    def create_topology(self: TModel, universe: mda.Universe, /) -> None:
        """Determine the topology attributes and initialize the universe.

        Parameters
        ----------
        universe : :class:`~MDAnalysis.Universe`
            An all-atom universe
        """
        atom_group: list[AtomGroup] = [universe.select_atoms(_) for _ in self._mapping.values()]
        self._universe: mda.Universe = mda.Merge(*atom_group)  # type: ignore

        float_type = universe.atoms.masses.dtype
        int_type = universe.atoms.resids.dtype

        # Atom
        atomids = np.arange(self._universe.atoms.n_atoms, dtype=int_type)
        attributes = (
            ("radii", np.zeros_like(atomids, dtype=float_type)),
            ("ids", np.zeros_like(atomids, dtype=float_type)),
        )
        for attribute in attributes:
            self._universe.add_TopologyAttr(*attribute)

        self._add_masses(universe)
        self._add_charges(universe)

    def add_trajectory(self: TModel, universe: mda.Universe, /) -> None:
        """Add coordinates to the new system.

        Parameters
        ----------
        universe: :class:`~MDAnalysis.Universe`
            An all-atom universe
        """
        if not hasattr(self, "_universe"):
            msg = "Topologies need to be created before bonds can be added."
            raise AttributeError(msg)

        if not hasattr(universe, "trajectory"):
            msg = "The provided universe does not have coordinates defined."
            raise AttributeError(msg)

        selections = itertools.product(universe.residues, self._mapping.items())  # type: ignore
        beads: list[AtomGroup] = []  # type: ignore
        total_beads: list[AtomGroup] = []  # type: ignore
        for residue, (key, selection) in selections:
            value = selection.get(residue.resname) if isinstance(selection, dict) else selection
            if residue.atoms.select_atoms(value):
                beads.append(residue.atoms.select_atoms(value))

            other_selection = getattr(self._selection, key)
            total_beads.append(residue.atoms.select_atoms(other_selection))

        position_array: list[NDArray] = []
        dimension_array: list[NDArray] = []
        universe.trajectory.rewind()
        for ts in universe.trajectory:
            dimension_array.append(ts.dimensions)

            # Positions
            try:
                positions = [_.positions for _ in beads if _]
                position_array.append(np.concatenate(positions, axis=0))
            except (AttributeError, mda.NoDataError):
                pass

        self._universe.trajectory.dimensions_array = np.asarray(dimension_array)
        if self._universe.trajectory.ts.has_positions:
            dim = np.asarray([999.0, 999.0, 999.0, 90.0, 90.0, 90.0], dtype=float)
            transform = transformations.boxdimensions.set_dimensions(dim)
            self._universe.load_new(np.asarray(position_array), format=MemoryReader)
            self._universe.trajectory.add_transformations(transform)
        universe.trajectory.rewind()

    def _add_masses(self: TModel, universe: mda.Universe, /) -> None:
        selections = itertools.product(universe.residues, self._selection.values())
        try:
            masses = np.concatenate(
                [
                    residue.atoms.select_atoms(selection).masses
                    for residue, selection in selections
                    if residue.atoms.select_atoms(selection)
                ]
            )
        except (AttributeError, mda.NoDataError):
            masses = np.zeros(self._universe.atoms.n_atoms, dtype=universe.atoms.masses.dtype)

        self._universe.add_TopologyAttr("masses", masses)

    def _add_charges(self: TModel, universe: mda.Universe, /) -> None:
        selections = itertools.product(universe.residues, self._selection.values())
        try:
            charges = np.concatenate(
                [
                    residue.atoms.select_atoms(selection).charges
                    for residue, selection in selections
                    if residue.atoms.select_atoms(selection)
                ]
            )
        except (AttributeError, mda.NoDataError):
            charges = np.zeros(self._universe.atoms.n_atoms, dtype=universe.atoms.masses.dtype)

        self._universe.add_TopologyAttr("charges", charges)

    def _add_bonds(self: TModel) -> None:
        try:
            atoms: AtomGroup = self._universe.atoms
            positions = self._universe.atoms.positions

            bonds: list[tuple[int, int]] = guessers.guess_bonds(atoms, positions)
            self._universe.add_TopologyAttr("bonds", bonds)
        except AttributeError:
            pass
