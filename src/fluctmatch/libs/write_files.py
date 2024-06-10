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

import datetime
import getpass
from importlib import resources
from pathlib import Path
from string import Template

import MDAnalysis as mda
from loguru import logger
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
    average_universe.convert_to("PARMED").save(filename.as_posix(), overwrite=True)


def write_charmm_input(
    *,
    topology: Path | str = "fluctmatch.psf",
    trajectory: Path | str = "cg.dcd",
    directory: Path | str | None = None,
    prefix: str = "fluctmatch",
    temperature: float = 300.0,
    sim_type: str = "fluctmatch",
) -> None:
    """Save the CHARMM input file.

    Parameters
    ----------
    topology : Path or str (default fluctmatch.psf)
        Topology file
    trajectory : Path or str (default cg.dcd)
        Trajectory file
    directory : Path or str, optional (default=None)
        Directory various files are located
    prefix : str, optional (default="fluctmatch")
        Prefix for output files
    temperature : float (default 300.0)
        Temperature in Kelvin
    sim_type : {'fluctmatch', 'thermodynamics'}, optional
        Type of simulation

    Raises
    ------
    FileNotFound
        Either because an incorrect `sim_type` was selected or the header template is missing.

    Notes
    -----
    `sim_type` currently can either be "fluctmatch" or "thermodynamics". "Fluctmatch" will prepare the CHARMM input
    file for normal mode analysis used in fluctuation matching; "Thermodynamics" will prepare the CHARMM input file
    for the calculation of thermodynamic properties (enthalpy, entropy, and free energy) using the provided
    'trajectory'.
    """
    module: str = "fluctmatch.templates.charmm"
    data: dict[str, Path | str | float] = {"topology": topology, "trajectory": trajectory, "temperature": temperature}
    lines: list[str] = []

    now: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user: str = getpass.getuser()
    _title: list[str] = [f"* Created by fluctmatch on {now}.\n", f"* User: {user}\n"]
    data["title"] = "".join(_title)

    _directory: Path = Path.cwd() if directory is None else Path(directory)
    data["prefix"] = _directory.joinpath(prefix)
    filename: Path = _directory.joinpath(f"{sim_type}.inp")

    header_file = resources.files(module).joinpath("cg_header.inp")
    try:
        logger.debug("Opening the template file for the header of the CHARMM input file.")
        header: str = header_file.read_text()
        lines.append(Template(header).substitute(**data))
    except FileNotFoundError as exc:
        exc.add_note(f"{header_file} seems to be missing. Please make sure you installed 'fluctmatch' properly.")
        logger.exception(exc)
        raise

    body_file = resources.files(module).joinpath(f"{sim_type}.inp")
    try:
        logger.debug(f"Opening the template file for the body of the CHARMM input file for '{sim_type}' simulation.")
        body: str = body_file.read_text()
        lines.append(Template(body).substitute(**data))
    except FileNotFoundError as exc:
        exc.add_note(f"{body_file} seems to be missing. Please make sure you installed 'fluctmatch' properly.")
        logger.exception(exc)
        raise

    with Path(filename).open("w") as file:
        logger.info(f"Writing the CHARMM input file to {filename}.")
        file.write("".join(lines))
