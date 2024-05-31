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
# pyright: reportAttributeAccessIssue=false, reportAssignmentType=false, reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
"""Write trajectory files."""

from pathlib import Path

import MDAnalysis as mda
from MDAnalysis.analysis.align import AverageStructure


def write_trajectory(
    universe: mda.Universe, /, filename: str | Path = "filename.dcd", start: int | None = None, stop: int | None = None
) -> None:
    """Asynchronously write a trajectory file from a slice of the trajectory.

    Parameters
    ----------
    universe : mda.Universe
        Universe to be written.
    filename : str or Path, default=filename.dcd
        new trajectory file
    start : int, optional
        beginning frame of the trajectory
    stop : int, optional
        final frame of the trajectory
    """
    with mda.Writer(filename, n_atoms=universe.atoms.n_atoms) as w:
        for _ in universe.trajectory[start:stop]:
            w.write(universe.atoms)


def write_average_structure(
    universe: mda.Universe, /, filename: str | Path = "filename.crd", start: int | None = None, stop: int | None = None
) -> None:
    """Asynchronously write a coordinate file from a slice of the trajectory.

    Parameters
    ----------
    universe : :class:`MDAnalysis.Universe`
        Universe to be written.
    filename : str or Path, default=filename.crd
        new trajectory file
    start : int, optional
        beginning frame of the trajectory
    stop : int, optional
        final frame of the trajectory
    """
    average = AverageStructure(universe)
    average_universe: mda.AtomGroup = average.run(start=start, stop=stop).results.universe.atoms
    average_universe.convert_to("PARMED").save(filename, overwrite=True)
