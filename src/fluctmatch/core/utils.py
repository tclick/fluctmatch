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
# pyright: reportGeneralTypeIssues=false
"""Various utilities for the models."""

import string
from pathlib import Path
from typing import TypeVar

import MDAnalysis as mda
import numpy as np
from loguru import logger
from MDAnalysis import transformations
from MDAnalysis.coordinates.memory import MemoryReader
from numpy.typing import NDArray

from .. import _MODELS
from .base import ModelBase

T = TypeVar("T")


def modeller(*args: str | Path, **kwargs: T) -> mda.Universe:
    """Create coarse-grain model from universe selection.

    Parameters
    ----------
    topology : Path or str
        A topology file containing atomic information about a system.
    trajectory : Path or str
        A trajectory file with coordinates of atoms
    model : list[str], optional
        Name(s) of coarse-grain core

    Returns
    -------
    A coarse-grain model
    """
    models: list[str] = [_.upper() for _ in kwargs.pop("model", ["polar"])]
    try:
        if "ENM" in models:
            logger.warning("ENM model detected. All other core are being ignored.")
            model: ModelBase = _MODELS.get("ENM", **kwargs)
            return model.transform(mda.Universe(*args, **kwargs))
    except Exception as exc:
        logger.exception("An error occurred while trying to create the universe.")
        raise RuntimeError from exc

    try:
        universe: list[mda.Universe] = [_MODELS.get(_, **kwargs).transform(mda.Universe(*args)) for _ in models]
    except KeyError as err:
        message: str = f"One of the core is not implemented. Please try {_MODELS.keys()}"
        logger.exception(message)
        raise KeyError(message) from err
    else:
        return merge(*universe)


def merge(*args: mda.Universe) -> mda.Universe:
    """Combine multiple coarse-grain systems into one.

    Parameters
    ----------
    args : iterable of :class:`~MDAnalysis.Universe`

    Returns
    -------
    :class:`~MDAnalysis.Universe`
        A merged universe.
    """
    logger.warning("This might take a while depending upon the number of trajectory frames.")

    # Merge universes
    multiverse: list[mda.AtomGroup] = [u.atoms for u in args]
    universe = mda.Merge(*multiverse)
    atoms = universe.atoms

    universe1: mda.Universe = args[0]
    trajectory1 = universe1.trajectory
    frames = np.fromiter([u.trajectory.n_frames == trajectory1.n_frames for u in args], dtype=bool)
    if not all(frames):
        message = "The trajectories are not the same length."
        logger.exception(message)
        raise ValueError(message)

    trajectory1.rewind()
    if trajectory1.n_frames > 1:
        # Accumulate coordinates, velocities, and forces.
        positions: list[NDArray] = [np.asarray(ts.positions for ts in u.trajectory if ts.has_positions) for u in args]

        pos = np.concatenate(positions, axis=1)
        if atoms.n_atoms != pos.shape[1]:
            message = "The number of sites does not match the number of coordinates."
            logger.exception(message)
            raise RuntimeError(message)
        n_frames, n_beads, _ = pos.shape
        logger.info(f"The new universe has {n_beads:d} beads in {n_frames:d} frames.")
        universe.load_new(positions, format=MemoryReader)

        dim = np.asarray([999.0, 999.0, 999.0, 90.0, 90.0, 90.0], dtype=float)
        transform = transformations.boxdimensions.set_dimensions(dim)
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
    atoms: mda.AtomGroup = universe.atoms
    types: NDArray = atoms.types
    segments: mda.SegmentGroup = universe.segments
    attributes: dict[str, NDArray] = {}

    attributes["names"] = np.array(
        [
            f"{letter}{i:0>3d}"
            for letter, segment in zip(string.ascii_uppercase, segments, strict=False)
            for i, _ in enumerate(segment.atoms, 1)
        ]
    )
    attributes["resnames"] = np.array(
        [
            f"{letter}{i:0>3d}"
            for letter, segment in zip(string.ascii_uppercase, segments, strict=False)
            for i, _ in enumerate(segment.residues, 1)
        ]
    )
    if not np.issubdtype(types.dtype, np.int64):
        attributes["types"] = attributes["names"]

    for attr, value in attributes.items():
        universe.add_TopologyAttr(topologyattr=attr, values=value)
