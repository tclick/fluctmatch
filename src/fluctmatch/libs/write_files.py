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

import asyncio
import datetime
import getpass
from pathlib import Path

import aiofiles
import MDAnalysis as mda
import numpy as np
import parmed as pmd
from loguru import logger
from MDAnalysis.analysis.align import AverageStructure
from numpy.typing import NDArray
from parmed.charmm.parameters import CharmmParameterSet
from parmed.utils.fortranformat import FortranRecordWriter


async def write_trajectory(
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
            await asyncio.sleep(0)  # Yield control for async behavior


async def write_average_structure(
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
    await asyncio.sleep(0)


async def write_stream(
    universe: mda.Universe, /, filename: str | Path = "filename.str", title: str | None = None
) -> None:
    """Write a CHARMM stream file containing bond information.

    Parameters
    ----------
    universe : :class:`MDAnalysis.Universe`
        universe to be written.
    filename : str or Path, default=filename.str
        new stream file
    title : str
        initial information to write in the file

    Raises
    ------
    AttributeError
        if the universe does not contain bond information
    """
    if not hasattr(universe, "bonds"):
        message = "The universe does not contain bond information."
        logger.exception(message)
        raise AttributeError(message)

    now: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user: str = getpass.getuser()
    _title: str = f"* Created by fluctmatch on {now}.\n* User: {user}\n".upper() if title is None else title

    logger.info(f"Saving bond information to {filename}.")
    async with aiofiles.open(Path(filename), mode="w") as stream:
        await stream.write(_title)
        for atom1, atom2 in universe.bonds:
            await stream.write("IC EDIT\n")
            await stream.write(
                f"DIST {atom1.segid:<8}{atom1.resid:>5d} {atom1.name:<8} {atom2.segid:<8}{atom2.resid:>5d} {atom2.name:<8} {0.0:>5.1f}\n"
            )
            await stream.write("END\n\n")
        await stream.write("RETURN\n")


async def write_intcor(
    universe: mda.Universe, data: NDArray, /, filename: str | Path = "filename.str", title: str | None = None
) -> None:
    """Write a CHARMM stream file containing bond information.

    Parameters
    ----------
    universe : :class:`MDAnalysis.Universe`
        universe to be written.
    data : NDArray
        bond data to include in the file
    filename : str or Path, default=filename.str
        new internal coordinate file
    title : str
        initial information to write in the file

    Raises
    ------
    AttributeError
        if the universe does not contain bond information
    """
    if not hasattr(universe, "bonds"):
        message = "The universe does not contain bond information."
        logger.exception(message)
        raise AttributeError(message)

    # Header information
    now: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user: str = getpass.getuser()
    _title: str = f"* Created by fluctmatch on {now}.\n* User: {user}\n".upper() if title is None else title
    header1_info = np.zeros(20, dtype=int)
    header1_info[0], header1_info[1] = 30, 2
    header2_info = np.array([len(universe.bonds), 2], dtype=int)

    # FORTRAN formatters
    header1 = FortranRecordWriter("20I4")
    header2 = FortranRecordWriter("2I5")
    line = FortranRecordWriter("I10,4(1X,A8,1X,A8,1X,A8,A2),F12.6,3F12.4,F12.6")

    logger.info(f"Saving internal coordinates to {filename}.")
    async with aiofiles.open(Path(filename), mode="w") as ic_file:
        other_info = 2 * ["??", "??", "??", ":"]
        await ic_file.write(_title)
        await ic_file.write(header1.write(header1_info) + "\n")
        await ic_file.write(header2.write(header2_info) + "\n")
        info = np.zeros(5, dtype=universe.bonds.values().dtype)

        for i, (bond, value) in enumerate(zip(universe.bonds, data, strict=True), 1):
            atom1, atom2 = bond.atoms
            atom1_info = [i, f"{atom1.segid}", atom1.resid, f"{atom1.name}", " :"]
            atom2_info = [f"{atom2.segid}", atom2.resid, f"{atom2.name}", " :"]
            info[0] = value

            await ic_file.write(line.write(atom1_info + atom2_info + other_info + info.tolist()) + "\n")


async def write_parameters(universe: mda.Universe, /, **kwargs: dict[str, NDArray | str | Path]) -> None:
    """Write bond information to a CHARMM parameter and topology file.

    Parameters
    ----------
    universe : :class:`MDAnalysis.Universe`
        Universe
    forces : NDArray
        Forces between bonds (in kcal/mol-A^2)
    distances : NDArray
        Bond distances
    par : str or Path, default=filename.prm
        CHARMM parameter file
    top : str or Path, default=filename.rtf
        CHARMM topology file
    """
    if not hasattr(universe, "bonds"):
        message = "The universe does not contain bond information."
        logger.exception(message)
        raise AttributeError(message)

    par: Path = Path(kwargs.get("par", "filename.prm"))
    top: Path = Path(kwargs.get("top", "filename.rtf"))
    stream_file = par.with_suffix(".str")
    forces: NDArray = kwargs.get("forces", np.zeros(len(universe.bonds), dtype=universe.bonds.values().dtype))
    distances: NDArray = kwargs.get("distances", np.zeros(len(universe.bonds), dtype=universe.bonds.values().dtype))

    parameters = CharmmParameterSet()
    atoms = universe.atoms
    keys = universe.bonds.topDict.keys()

    logger.info("Initializing CHARMM parameters.")
    atom_types = {atom.type: pmd.AtomType(atom.name, -1, atom.mass, charge=atom.charge) for atom in atoms}
    parameters.atom_types.update(atom_types)
    for k in parameters.atom_types:
        parameters.atom_types[k].epsilon = 0.0
        parameters.atom_types[k].epsilon_14 = 0.0
        parameters.atom_types[k].sigma = 0.0
        parameters.atom_types[k].sigma_14 = 0.0

    bond_types = {key: pmd.BondType(k=k, req=req) for key, k, req in zip(keys, forces, distances, strict=True)}
    parameters.bond_types.update(bond_types)

    logger.info(f"Writing the topology/parameter/stream files to {par}/{top}/{stream_file}.")
    parameters.write(par=par.as_posix(), top=top.as_posix(), stream=stream_file.as_posix())
    await asyncio.sleep(0)
