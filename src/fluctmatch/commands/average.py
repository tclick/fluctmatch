# ---------------------------------------------------------------------------------------------------------------------
#  fluctmatch
#  Copyright (c) 2024 Timothy H. Click, Ph.D.
#
#  This file is part of fluctmatch.
#
#  Fluctmatch is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
#  License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
#  later version.
#
#  Fluctmatch is distributed in the hope that it will be useful, # but WITHOUT ANY WARRANTY; without even the implied
#  warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with this program.
#  If not, see <[1](https://www.gnu.org/licenses/)>.
#
#  Reference:
#  Timothy H. Click, Nixon Raj, and Jhih-Wei Chu. Simulation. Meth Enzymology. 578 (2016), 327-342,
#  Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics doi:10.1016/bs.mie.2016.05.024.
# ---------------------------------------------------------------------------------------------------------------------
"""Average the coordinates of a trajectory.

This script reads a trajectory file and calculates the average coordinates.

Usage
-----
    $ fluctmatch average -s <topology> -f <trajectory> -o <output> -l <logfile>

Notes
-----
.. warn:: Depending upon the length of the trajectory and the size of the universe, the process can take several minutes.
minutes.

Examples
--------
    $ fluctmatch average -s cg.psf -f cg.dcd -o cg.crd -l average.log
"""

from pathlib import Path

import click
import MDAnalysis as mda
from click_help_colors import HelpColorsCommand
from loguru import logger

import fluctmatch.model.selection  # noqa: F401
from fluctmatch import __copyright__
from fluctmatch.libs.logging import config_logger
from fluctmatch.libs.write_files import write_average_structure

__help__ = """Average the coordinates of a trajectory.

This script reads a trajectory file and calculates the average coordinates."""


@click.command(
    cls=HelpColorsCommand,
    help=f"{__copyright__}\n{__help__}",
    short_help="Align a trajectory to the first frame or to a reference structure",
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
    "-o",
    "--output",
    metavar="FILE",
    default=Path.cwd(),
    show_default=True,
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    help="Coordinates of average structure",
)
@click.option(
    "-d",
    "--directory",
    metavar="DIR",
    default=Path(),
    show_default=True,
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    help="Parent directory if wanting to average multiple trajectories",
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
def average(
    topology: Path,
    trajectory: Path,
    output: Path,
    directory: Path,
    logfile: Path,
    verbosity: str,
) -> None:
    """Average a molecular dynamics trajectory.

    This function averages a trajectory. The average coordinates are saved in the specified output file.
    Logging of the process is configurable through the verbosity level and the logfile path.

    Parameters
    ----------
    topology : Path, default: $CWD/input.parm7
        Path to the topology file.
    trajectory : Path, default: $CWD/input.parm7
        Path to the trajectory file.
    output : Path, default: $CWD/cg.crd
        Path to the output file where the average coordinates will be saved.
    directory : Path, optional
        If wanting to average multiple trajectories, the directory containing subdirectories with trajectories
    logfile : Path, default: $CWD/logging.log
        Path to the log file for recording the process.
    verbosity : {'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'}
        Logging verbosity level.

    Returns
    -------
    None

    Notes
    -----
    The function initializes the logger to record messages to the specified log file and console. It selects atoms from
    the trajectory and reference structure based on the provided selection criteria. A transformation is applied to
    align the trajectory to the reference, and the aligned trajectory is written to the output directory.
    """
    config_logger(name=__name__, logfile=logfile, level=verbosity.upper())
    click.echo(__copyright__)

    try:
        if directory != Path() and directory.is_dir():
            subdirs = (_ for _ in directory.iterdir() if _.is_dir())
            for subdir in subdirs:
                top_file = subdir.joinpath(topology.name)
                traj_file = subdir.joinpath(trajectory.name)
                out_file = subdir.joinpath(output.name)
                universe = mda.Universe(top_file, traj_file)
                write_average_structure(universe, out_file)
        else:
            universe = mda.Universe(topology, trajectory)
            write_average_structure(universe, output)
    except FileNotFoundError as exception:
        logger.exception(exception)
        raise
