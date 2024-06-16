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
    average_universe.convert_to("PARMED").save(Path(filename).as_posix(), overwrite=True)


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
