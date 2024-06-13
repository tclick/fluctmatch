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
# pyright: reportAssignmentType=false, reportAttributeAccessIssue=false, reportArgumentType=false
"""Various utilities for the models."""

import string
from collections.abc import MutableMapping
from concurrent.futures import ProcessPoolExecutor

import MDAnalysis as mda
import MDAnalysis.transformations as transformations
import numpy as np
from loguru import logger
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.coordinates.memory import MemoryReader
from numpy.typing import NDArray


def _get_trajectory(ag: mda.AtomGroup) -> NDArray:
    """Retrieve the trajectory positions of a universe.

    Parameters
    ----------
    ag : mda.AtomGroup
        Atom group

    Returns
    -------
    NDArray
        A 3D array containing the trajectory positions of a universe.
    """
    return AnalysisFromFunction(lambda ag: ag.positions.copy(), ag).run().results["timeseries"]


def merge(*universes: mda.Universe) -> mda.Universe:
    """Combine multiple coarse-grain systems into one.

    Parameters
    ----------
    universes : iterable of :class:`~MDAnalysis.Universe`

    Returns
    -------
    Universe
        A merged universe

    Raises
    ------
    ValueError
        if trajectory size or total atom numbers differ
    """
    logger.warning("This might take a while depending upon the number of trajectory frames.")

    # Merge universes
    multiverse: list[mda.AtomGroup] = [u.atoms for u in universes]
    universe = mda.Merge(*multiverse)
    atoms = universe.atoms

    universe1: mda.Universe = universes[0]
    trajectory1 = universe1.trajectory
    frames = np.fromiter([u.trajectory.n_frames == trajectory1.n_frames for u in universes], dtype=bool)
    if not all(frames):
        message = "The trajectories are not the same length."
        logger.exception(message)
        raise ValueError(message)

    trajectory1.rewind()
    if trajectory1.n_frames > 1:
        # Accumulate coordinates, velocities, and forces.
        with ProcessPoolExecutor(max_workers=len(multiverse)) as executor:
            positions = list(executor.map(_get_trajectory, multiverse))
        positions = np.concatenate(positions, axis=1)
        if atoms.n_atoms != positions.shape[1]:
            message = "The number of sites does not match the number of coordinates."
            logger.exception(message)
            raise ValueError(message)

        n_frames, n_beads, _ = positions.shape
        logger.info(f"The new universe has {n_beads:d} beads in {n_frames:d} frames.")
        universe.load_new([positions], format=MemoryReader)

        box_dimension = np.asarray([999.0, 999.0, 999.0, 90.0, 90.0, 90.0], dtype=float)
        transform = transformations.boxdimensions.set_dimensions(box_dimension)
        universe.trajectory.add_transformations(transform)

    return universe


def rename_universe(universe: mda.Universe, /) -> None:
    """Rename the atoms and residues within a universe.

    Standardizes naming of the universe by renaming atoms and residues based
    upon the number of segments. Atoms are labeled as 'A001', 'A002', 'A003',
    ..., 'A999' for the first segment, and 'B001', 'B002', 'B003', ..., 'B999'
    for the second segment. Residues are named in a similar fashion according to
    their segment.

    Parameters
    ----------
    universe : :class:`~MDAnalysis.Universe`
        A collection of atoms in a universe.
    """
    logger.info("Renaming atom names and atom core within the universe.")
    segments: mda.SegmentGroup = universe.segments
    for letter, segment in zip(string.ascii_uppercase, segments, strict=False):
        for i, atom in enumerate(segment.atoms, 1):
            atom.name = f"{letter}{i:0>5d}"
            atom.type = f"{letter}{i:0>5d}"
        for i, residue in enumerate(segment.residues, 1):
            residue.resname = f"{letter}{i:0>5d}"


def compare_dict_keys(dict1: MutableMapping, dict2: MutableMapping, /, message: str = "") -> None:
    """Compare two dictionaries by keys.

    Parameters
    ----------
    dict1 : MutableMapping
        The first mutable dictionary
    dict2 : MutableMapping
        The second mutable dictionary
    message : str, optional
        Message to pass to exception

    Raises
    ------
    ValueError
        If the key sets of the two OrderedDicts don't match.
    """
    if set(dict1.keys()) != set(dict2.keys()):
        _message = message if message else "Key sets do not match."
        raise ValueError(_message)
