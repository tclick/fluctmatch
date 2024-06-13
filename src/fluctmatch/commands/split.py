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
"""Split a trajectory into smaller trajectories."""

import json
from pathlib import Path

import click
import MDAnalysis as mda
from click_help_colors import HelpColorsCommand
from loguru import logger

from fluctmatch import __copyright__
from fluctmatch.libs import write_files
from fluctmatch.libs.logging import config_logger


@click.command(
    cls=HelpColorsCommand,
    help=f"{__copyright__}\nAlign a trajectory.",
    short_help="Split a trajectory into smaller trajectories.",
    help_headers_color="yellow",
    help_options_color="blue",
    context_settings={"max_content_width": 120},
)
@click.option(
    "-s",
    "--topology",
    metavar="FILE",
    default=Path.cwd().joinpath("input.parm7"),
    show_default=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path),
    help="Topology file",
)
@click.option(
    "-f",
    "--trajectory",
    metavar="FILE",
    default=Path.cwd().joinpath("input.nc"),
    show_default=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path),
    help="Trajectory file",
)
@click.option(
    "--json",
    "windows_input",
    metavar="JSON",
    show_default=True,
    default=Path.cwd().joinpath("setup.json"),
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="JSON file",
)
@click.option(
    "-o",
    "--outfile",
    metavar="FILE",
    default="cg.dcd",
    show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Trajectory output file",
)
@click.option(
    "-c",
    "--crdfile",
    metavar="FILE",
    default="cg.crd",
    show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Average structure file",
)
@click.option(
    "-l",
    "--logfile",
    metavar="WARNING",
    show_default=True,
    default=Path.cwd() / Path(__file__).with_suffix(".log"),
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to log file",
)
@click.option("--average", is_flag=True, help="Save the average structure of the trajectory")
@click.option(
    "-v",
    "--verbosity",
    default="INFO",
    show_default=True,
    help="Minimum severity level for log messages",
)
@click.help_option("-h", "--help", help="Show this help message and exit")
def split(
    topology: Path,
    trajectory: Path,
    outfile: Path,
    windows_input: Path,
    logfile: Path,
    average: bool,
    crdfile: Path,
    verbosity: str,
) -> None:
    """Split a trajectory into smaller trajectories using the JSON file created during setup.

    Parameters
    ----------
    topology : Path, default=$CWD/input.parm7
        Topology file
    trajectory : Path, default=$CWD/input.nc
        Trajectory file
    outfile : Path, default=cg.dcd
        Trajectory output file
    logfile : Path, default=$CWD/setup.log
        Location of log file
    windows_input : Path, default=$CWD/setup.json
        JSON file
    average : bool
        Save the average structure of the trajectory
    crdfile : Path
        Average structure of the trajectory
    verbosity : str, default=INFO
        Level of verbosity for logging output
    """
    config_logger(name=__name__, logfile=logfile, level=verbosity)
    click.echo(__copyright__)

    with windows_input.open() as f:
        setup_input: dict = json.load(f)

    universe = mda.Universe(topology, trajectory)

    logger.info("Splitting trajectory into smaller trajectories...")
    info = ((Path(outdir).joinpath(outfile), data["start"], data["stop"]) for outdir, data in setup_input.items())
    for traj_file, start, stop in info:
        traj_file.parent.mkdir(exist_ok=True)
        write_files.write_trajectory(universe.copy(), traj_file.as_posix(), start=start, stop=stop)

    if average:
        logger.info("Saving the average structures of each trajectory...")
        info = ((Path(outdir).joinpath(crdfile), data["start"], data["stop"]) for outdir, data in setup_input.items())
        for crd_file, start, stop in info:
            write_files.write_average_structure(universe.copy(), crd_file.as_posix(), start=start, stop=stop)
