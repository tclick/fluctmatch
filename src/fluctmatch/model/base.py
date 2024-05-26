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
# pyright: reportInvalidTypeVarUse = false, reportCallIssue = false, reportArgumentType = false
# pyright: reportAssignmentType = false, reportAttributeAccessIssue = false, reportOptionalMemberAccess = false
"""Base classes for the model and the factory."""

import abc
import itertools
from types import MappingProxyType
from typing import Self

import MDAnalysis as mda
import MDAnalysis.core.groups as groups
import MDAnalysis.topology.base as topbase
import MDAnalysis.topology.guessers as guessers
import MDAnalysis.transformations as transformations
import numpy as np
from class_registry import AutoRegister, ClassRegistry
from loguru import logger
from MDAnalysis.coordinates.memory import MemoryReader
from numpy.typing import NDArray

from fluctmatch.model.selection import *

coarse_grain = ClassRegistry("model")


class CoarseGrainModel(metaclass=AutoRegister(coarse_grain)):
    """Base class for creating coarse-grain core.

    Parameters
    ----------
    mobile : Universe
        All-atom universe

    Attributes
    ----------
    _universe : :class:`~MDAnalysis.Universe`
        The transformed universe
    """

    def __init__(self: Self, mobile: mda.Universe, /, **kwargs: dict[str, bool]) -> None:  # noqa: ARG002
        """Initialise like a normal MDAnalysis Universe but give the mapping and com keywords.

        Mapping must be a dictionary with atom names as keys.
        Each name must then correspond to a selection string,
        signifying how to split up a single residue into many beads.
        eg:
        mapping = {"CA":"protein and name CA",
                   "CB":"protein and not name N HN H HT* H1 H2 H3 CA HA* C O OXT
                   OT*"}
        would split residues into 2 beads containing the C-alpha atom and the
        sidechain.
        """
        # Coarse grained Universe
        # Make a blank Universe for myself.
        self._mobile = mobile.copy()
        self._universe: mda.Universe = mda.Universe.empty(0)

        # Named tuple for specific bead selections. This is primarily used to
        # determine positions.
        self._mapping: MappingProxyType = MappingProxyType({})

        # Named tuple for all-atom to bead selection. This is particularly
        # useful for mass, charge, velocity, and force topology attributes.
        self._selection: MappingProxyType = MappingProxyType({})

        # Beads from all-atom system
        self._beads: list[mda.AtomGroup] = []
        self._mass_beads: list[mda.AtomGroup] = []

        # Residues with corresponding atoms and selection criteria.
        self._residues: tuple[tuple[groups.Residue, str, str | mda.ResidueGroup], ...] | None = None

    @property
    def universe(self: Self) -> mda.Universe:
        """Retrieve coarse-grain universe.

        Returns
        -------
        MDAnalysis.Universe
            coarse-grain model
        """
        return self._universe

    @property
    def atoms(self: Self) -> mda.AtomGroup:
        """Retrieve coarse-grain model.

        Returns
        -------
        MDAnalysis.AtomGroup
            coarse-grain model
        """
        return self._universe.atoms

    def create_topology(self: Self) -> None:
        """Determine the topology attributes and initialize the universe."""
        # Allocate arrays
        residues: mda.ResidueGroup = self._mobile.residues

        atomnames: list[str] = []
        self._residues = tuple(
            (residue, key, value) for residue, (key, value) in itertools.product(residues, self._mapping.items())
        )

        logger.debug("Creating the coarse-grain topology.")
        for residue, key, selection in self._residues:
            value: str = selection.get(residue.resname) if isinstance(selection, MappingProxyType) else selection
            bead: mda.AtomGroup = residue.atoms.select_atoms(value)
            if bead:
                self._beads.append(bead)
                atomnames.append(key)

        attributes: dict[str, NDArray] = {}

        # Atom
        atomids = np.arange(len(self._beads), dtype=int)
        attributes["names"] = np.asarray(atomnames, dtype=object)
        attributes["radii"] = np.zeros_like(atomids, dtype=float)
        attributes["ids"] = np.zeros_like(atomids, dtype=float)
        if not np.issubdtype(self._mobile.atoms.types.dtype, np.int64):
            attributes["types"] = np.asarray(atomnames, dtype=object)

        # Residue
        resids = np.asarray([bead.resids[0] for bead in self._beads], dtype=int)
        resnames = np.asarray([bead.resnames[0] for bead in self._beads], dtype=object)
        segids = np.asarray([bead.segids[0].split("_")[-1] for bead in self._beads], dtype=object)
        residx, (new_resids, new_resnames, perres_segids) = topbase.change_squash(
            (resids, resnames, segids), (resids, resnames, segids)
        )

        # transform from atom:Rid to atom:Rix
        attributes["resids"] = new_resids
        attributes["resnums"] = new_resids
        attributes["resnames"] = new_resnames

        # Segment
        segidx, perseg_segids = topbase.squash_by(perres_segids)[:2]
        attributes["segids"] = perseg_segids

        # Create universe and add attributes
        self._universe = mda.Universe.empty(
            len(atomids),
            n_residues=len(new_resids),
            n_segments=len(perseg_segids),
            atom_resindex=residx,
            residue_segindex=segidx,
            trajectory=self._mobile.trajectory.ts.has_positions,
            velocities=self._mobile.trajectory.ts.has_velocities,
            forces=self._mobile.trajectory.ts.has_forces,
        )

        # Add additonal attributes
        for attr, value in attributes.items():
            self._universe.add_TopologyAttr(topologyattr=attr, values=value)
        self._add_masses()
        self._add_charges()

    def generate_bonds(self: Self, rmin: float = 0.0, rmax: float = 10.0, guess: bool = False) -> None:
        """Add bonds, angles, dihedrals, and improper dihedrals to the universe.

        Parameters
        ----------
        rmin : float
            Minimum bond distance
        rmax : bool
            Maximum bond distance
        guess : bool, optional
            Guess angles and dihedral and improper dihedral angles
        """
        if self._universe.atoms.n_atoms == 0:
            message = "Topologies need to be created before trajectory can be added."
            raise AttributeError(message)

        self._add_bonds(rmin=rmin, rmax=rmax)
        if guess:
            self._add_angles()
            self._add_dihedrals()
            self._add_impropers()

    def add_trajectory(
        self: Self, start: int | None = None, stop: int | None = None, step: int | None = None, com: bool = True
    ) -> None:
        """Add coordinates to the new system.

        Parameters
        ----------
        start : int, optional
            Beginning frame
        stop : int, optional
            Final frame
        step : int, optional
            Number of frames to skip
        com : bool, optional
            Define positions either by center of mass (default) or of geometry
        """
        if self._universe.atoms.n_atoms == 0:
            message = "Topologies need to be created before trajectory can be added."
            raise AttributeError(message)

        if not hasattr(self._mobile, "trajectory"):
            message = "The provided universe does not have coordinates defined."
            raise AttributeError(message)

        logger.debug("Creating the coarse-grain trajectory.")
        selections = itertools.product(self._mobile.residues, self._mapping.items())
        beads: list[mda.AtomGroup] = []
        total_beads: list[mda.AtomGroup] = []
        for residue, (key, selection) in selections:
            value = selection.get(residue.resname) if isinstance(selection, MappingProxyType) else selection
            if residue.atoms.select_atoms(value):
                beads.append(residue.atoms.select_atoms(value))

            other_selection = self._selection[key]
            total_beads.append(residue.atoms.select_atoms(other_selection))

        position_array: list[NDArray] = []
        self._mobile.trajectory.rewind()
        for _ts in self._mobile.trajectory[start:stop:step]:
            # Positions
            try:
                positions = np.asarray([_.center_of_mass() if com else _.center_of_geometry() for _ in beads if _])
                position_array.append(positions)
            except (AttributeError, mda.NoDataError):
                pass

        if self._universe.trajectory.ts.has_positions:
            dim = np.asarray([999.0, 999.0, 999.0, 90.0, 90.0, 90.0], dtype=float)
            transform = transformations.boxdimensions.set_dimensions(dim)
            self._universe.load_new(np.asarray(position_array), format=MemoryReader)
            self._universe.trajectory.add_transformations(transform)
        self._mobile.trajectory.rewind()

    def transform(
        self: Self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        rmin: float = 0.0,
        rmax: float = 10.0,
        com: bool = True,
        guess: bool = False,
    ) -> mda.Universe:
        """Convert an all-atom universe to a coarse-grain model.

        Topologies are generated, bead connections are determined, and positions
        are read. This is a wrapper for the other three methods.

        Parameters
        ----------
        start : int, optional
            Beginning frame
        stop : int, optional
            Final frame
        step : int, optional
            Number of frames to skip
        rmin : float
            Minimum bond distance
        rmax : bool
            Maximum bond distance
        com : bool, optional
            Define positions either by center of mass (default) or of geometry
        guess : bool, optional
            Guess angles and dihedral and improper dihedral angles

        Returns
        -------
        A coarse-grain model
        """
        logger.debug("Transforming an all-atom system to an elastic network model.")
        self.create_topology()
        self.generate_bonds(rmin=rmin, rmax=rmax, guess=guess)
        self.add_trajectory(start=start, stop=stop, step=step, com=com)
        return self._universe

    def _add_masses(self: Self) -> None:
        residues: mda.ResidueGroup = self._mobile.residues
        atoms: mda.AtomGroup = residues.atoms
        selections = itertools.product(residues, self._selection.values())

        try:
            logger.debug("Assigning masses to each bead.")
            masses = np.asarray(
                [
                    residue.atoms.select_atoms(selection).total_mass()
                    for residue, selection in selections
                    if residue.atoms.select_atoms(selection)
                ],
                dtype=atoms.masses.dtype,
            )
        except (AttributeError, mda.NoDataError):
            masses = np.zeros(self._universe.atoms.n_atoms, dtype=atoms.masses.dtype)

        self._universe.add_TopologyAttr("masses", masses)

    def _add_charges(self: Self) -> None:
        residues = self._mobile.residues
        atoms = residues.atoms
        selections = itertools.product(self._mobile.residues, self._selection.values())

        try:
            logger.debug("Assigning charges to the beads.")
            charges = np.asarray(
                [
                    residue.atoms.select_atoms(selection).total_charge()
                    for residue, selection in selections
                    if residue.atoms.select_atoms(selection)
                ],
                dtype=atoms.masses.dtype,
            )
        except (AttributeError, mda.NoDataError):
            charges = np.zeros(self._universe.atoms.n_atoms)

        self._universe.add_TopologyAttr("charges", charges)

    @abc.abstractmethod
    def _add_bonds(self: Self, rmin: float, rmax: float) -> None:
        msg = "Subclass must implement this method."
        raise NotImplementedError(msg)

    def _add_angles(self: Self) -> None:
        try:
            logger.debug("Guessing the angles.")
            angles = guessers.guess_angles(self._universe.bonds)
            self._universe.add_TopologyAttr("angles", angles)
        except AttributeError:
            pass

    def _add_dihedrals(self: Self) -> None:
        try:
            logger.debug("Guessing the dihedral angles.")
            dihedrals = guessers.guess_dihedrals(self._universe.angles)
            self._universe.add_TopologyAttr("dihedrals", dihedrals)
        except AttributeError:
            pass

    def _add_impropers(self: Self) -> None:
        try:
            logger.debug("Guessing the improper dihedral angles.")
            impropers = guessers.guess_improper_dihedrals(self._universe.angles)
            self._universe.add_TopologyAttr("impropers", impropers)
        except AttributeError:
            pass
