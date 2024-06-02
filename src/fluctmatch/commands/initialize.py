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
# pyright: reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportArgumentType=false
"""Split a trajectory into smaller trajectories."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import click
import MDAnalysis as mda
import numpy as np
from click_help_colors import HelpColorsCommand
from loguru import logger
from scipy import constants

from fluctmatch import __copyright__
from fluctmatch.io.charmm import BondData
from fluctmatch.io.charmm.intcor import CharmmInternalCoordinates
from fluctmatch.io.charmm.parameter import CharmmParameter
from fluctmatch.io.charmm.stream import CharmmStream
from fluctmatch.libs.bond_info import BondInfo
from fluctmatch.libs.logging import config_logger


@click.command(
    cls=HelpColorsCommand,
    help=f"{__copyright__}\nInitialize files for fluctuation matching.",
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
    metavar="WARNING",
    show_default=True,
    default=Path.cwd() / Path(__file__).with_suffix(".log"),
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
    type=click.FloatRange(min=1.0, clamp=True),
    help="Simulation temperature (in K)",
)
@click.option(
    "-v",
    "--verbosity",
    metavar="LEVEL",
    default="INFO",
    show_default=True,
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
    verbosity : str, default=INFO
        Level of verbosity for logging output
    """
    config_logger(name=__name__, logfile=logfile, level=verbosity)
    click.echo(__copyright__)

    boltzmann: float = temperature * (constants.k * constants.N_A / (constants.calorie * constants.kilo))

    logger.info("Loading the universe.")
    universe = mda.Universe(topology, trajectory)

    logger.info("Determining the average bond distance and the corresponding bond fluctuations.")
    bond_info = BondInfo(universe.atoms, verbose=verbosity == "DEBUG")
    bond_info.run()
    lengths: BondData = bond_info.results.mean
    fluct: BondData = bond_info.results.std
    forces: BondData = OrderedDict({key: boltzmann / np.square(std) for key, std in bond_info.results.std.items()})

    # CHARMM parameter, topology, and stream files
    parameters = CharmmParameter()
    parameters.initialize(universe, forces=forces, lengths=lengths)
    prm_file = directory / prefix.with_suffix(".prm")
    rtf_file = prm_file.with_suffix(".rtf")
    str_file = prm_file.with_suffix(".str")
    parameters.write(par=prm_file, top=rtf_file, stream=str_file)

    # Stream file with bond information
    str_file = prm_file.with_suffix(".bonds.str")
    stream = CharmmStream()
    stream.initialize(universe)
    stream.write(str_file)

    # Internal coordinate files
    avg_ic_file = prm_file.with_suffix(".average.ic")
    average_ic = CharmmInternalCoordinates()
    average_ic.initialize(universe, data=lengths)
    average_ic.write(avg_ic_file)

    fluct_ic_file = prm_file.with_suffix(".fluct.ic")
    fluct_ic = CharmmInternalCoordinates()
    fluct_ic.initialize(universe, data=fluct)
    fluct_ic.write(fluct_ic_file)
