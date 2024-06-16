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
"""Split a trajectory into smaller trajectories.

This script splits a trajectory into smaller trajectories based on the file created during the `setup` subcommand. The
JSON file will provide the location of the subdirectory and the frame start/stop times. The user can opt to have the
average structure of each trajectory written to a CHARMM coordinate file.

Usage
-----
    $ fluctmatch split -s <topology> -f <trajectory> -j <json-file> -o <trajectory> -l <logfile> --average

Notes
-----
.. warn:: Depending upon the trajectory length and universe size, this process can take a while.

The script can be run immediately after `setup` or after `convert`. If it is run after `setup`, it is recommended not to
use '--average' because fluctuation matching will depend upon the average structure of the transformed system.

Examples
--------
    $ fluctmatch split -s trex1.psf -f trex1.dcd -j setup.json -o trex1.dcd -l split.log --average
"""

import json
from pathlib import Path

import click
import MDAnalysis as mda
from click_help_colors import HelpColorsCommand
from loguru import logger

from fluctmatch import __copyright__
from fluctmatch.libs import write_files
from fluctmatch.libs.logging import config_logger

__help__ = """Trajectory spliiter


This script splits a trajectory into smaller trajectories based on the file created during the `setup` subcommand. The
JSON file will provide the location of the subdirectory and the frame start/stop times. The user can opt to have the
average structure of each trajectory written to a CHARMM coordinate file."""


@click.command(
    cls=HelpColorsCommand,
    help=f"{__copyright__}\n{__help__}",
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
    "-j",
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
    "--trajout",
    metavar="FILE",
    default="cg.dcd",
    show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Trajectory output file",
)
@click.option(
    "-l",
    "--logfile",
    metavar="FILE",
    show_default=True,
    default=Path.cwd().joinpath(__file__).with_suffix(".log"),
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to log file",
)
@click.option(
    "-v",
    "--verbosity",
    metavar="LEVEL",
    default="INFO",
    show_default=True,
    type=click.Choice("INFO DEBUG WARNING ERROR CRITICAL".split(), case_sensitive=False),
    help="Minimum severity level for log messages",
)
@click.help_option("-h", "--help", help="Show this help message and exit")
def split(
    topology: Path,
    trajectory: Path,
    trajout: Path,
    windows_input: Path,
    logfile: Path,
    verbosity: str,
) -> None:
    """Split a trajectory into smaller trajectories using the JSON file created during setup.

    Parameters
    ----------
    topology : Path, default=$CWD/input.parm7
        Topology file
    trajectory : Path, default=$CWD/input.nc
        Trajectory file
    trajout : Path, default=cg.dcd
        Trajectory output file
    logfile : Path, default=$CWD/setup.log
        Location of log file
    windows_input : Path, default=$CWD/setup.json
        JSON file
    verbosity : str, default=INFO
        Level of verbosity for logging output
    """
    config_logger(name=__name__, logfile=logfile, level=verbosity)
    click.echo(__copyright__)

    with windows_input.open() as f:
        setup_input: dict[str, dict[str, int]] = json.load(f)

    universe = mda.Universe(topology, trajectory)

    logger.info("Splitting trajectory into smaller trajectories...")
    info = ((Path(outdir).joinpath(trajout), data["start"], data["stop"]) for outdir, data in setup_input.items())
    for traj_file, start, stop in info:
        traj_file.parent.mkdir(exist_ok=True)
        write_files.write_trajectory(universe.copy(), traj_file.as_posix(), start=start, stop=stop)
