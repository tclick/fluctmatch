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
# pyright: reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportArgumentType=false
r"""Prepare the CHARMM input scripts.

This script allows the user to automate the creation of the simulation file that will be used for fluctuation
matching. Using `--directory` and `--prefix`, the user can set the location of the input file. `--temperature` is
specific to the calculation of thermodynamic properties. `--topology` and `--trajectory` specify the locations of the
topology and trajectory files that will be used during the simulation.

Usage
-----
    $ fluctmatch initialize -s <topology> -f <trajectory> -d <directory> -p <prefix> -l <logfile> -t <temperature>

Notes
-----
Currently, this script will create an input file specific for CHARMM. A future version of this script should have
options for other software packages like Amber or Gromacs.

Examples
--------
    $ fluctmatch initialize -s fluctmatch.psf -f cg.dcd -d . -p fluctmatch -l fluctmatch.log -t 300.0

"""

from pathlib import Path

import click
import MDAnalysis as mda
from click_help_colors import HelpColorsCommand
from loguru import logger

from fluctmatch import __copyright__
from fluctmatch.fm.charmm.fluctmatch import CharmmFluctuationMatching
from fluctmatch.libs.logging import config_logger

__help__ = """Simulation initialization

This script allows the user to automate the creation of the simulation file that will be used for fluctuation
matching. Using `--directory` and `--prefix`, the user can set the location of the input file. `--temperature` is
specific to the calculation of thermodynamic properties. `--topology` and `--trajectory` specify the locations of the
topology and trajectory files that will be used during the simulation."""


@click.command(
    cls=HelpColorsCommand,
    help=f"{__copyright__}\n{__help__}",
    short_help="Initialize files for fluctuation matching.",
    help_headers_color="yellow",
    help_options_color="blue",
    context_settings={"max_content_width": 120},
)
@click.option(
    "-s",
    "--topology",
    metavar="FILE",
    default=Path.cwd().joinpath("cg.psf"),
    show_default=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path),
    help="Topology file",
)
@click.option(
    "-f",
    "--trajectory",
    metavar="FILE",
    default=Path.cwd().joinpath("cg.dcd"),
    show_default=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path),
    help="Trajectory file",
)
@click.option(
    "-d",
    "--dir",
    "directory",
    metavar="DIR",
    default=Path.cwd(),
    show_default=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Output directory",
)
@click.option(
    "-p",
    "--prefix",
    metavar="FILE",
    default="fluctmatch",
    show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Filename prefix",
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
    "-t",
    "--temp",
    "temperature",
    metavar="TEMP",
    show_default=True,
    default=300.0,
    type=click.FloatRange(min=0.1, clamp=True),
    help="Simulation temperature (in K)",
)
@click.option(
    "-v",
    "--verbosity",
    metavar="LEVEL",
    default="INFO",
    show_default=True,
    type=click.Choice("INFO DEBUG WARNING ERROR CRITICAL".split()),
    help="Minimum severity level for log messages",
)
@click.help_option("-h", "--help", help="Show this help message and exit")
def initialize(
    topology: Path,
    trajectory: Path,
    directory: Path,
    prefix: Path,
    logfile: Path,
    temperature: float,
    verbosity: str,
) -> None:
    """Split a trajectory into smaller trajectories using the JSON file created during setup.

    Parameters
    ----------
    topology : Path, default=$CWD/cg.psf
        Topology file
    trajectory : Path, default=$CWD/cg.dcd
        Trajectory file
    directory : Path, default=$CWD
        Output directory
    prefix : Path, default=fluctmatch
        Filename prefix
    logfile : Path, default=$CWD/fluctmatch.log
        Location of log file
    temperature : float, default=300.0
        Simulation temperature
    verbosity : {INFO, DEBUG, WARNING, ERROR, CRITICAL}
        Level of verbosity for logging output
    """
    config_logger(name=__name__, logfile=logfile, level=verbosity)
    click.echo(__copyright__)

    logger.info("Loading the universe.")
    universe = mda.Universe(topology, trajectory)
    CharmmFluctuationMatching(universe, temperature=temperature, output_dir=directory, prefix=prefix).initialize()
