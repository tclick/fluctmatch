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
# pyright: reportAssignmentType=false, reportAttributeAccessIssue=false
"""Various utilities for the models."""

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
