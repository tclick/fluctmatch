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
r"""Perform self-consistent fluctuation matching.

This script performs self-consistent fluctuation matching using CHARMM.

Usage
-----
    $ fluctmatch simulate -s <topology> -f <trajectory> -d <directory> -l <logfile> --target <target-ic> \
        --param <param-file> --exec <exec-file> -p <prefix> -t <temperature> --max <max-cycles> --tol <tolerance>

Notes
-----
Depending upon the size of the model and the number of bonds, a single simulation can take several minutes. The
simulation will complete either when the maximum number of cycles has been reached, or when the tolerance level has
been achieved. The parameter file will be overwritten throughout the process with the varying force constants.

For CHARMM to work with the fluctuation matching code, it must be recompiled with some modifications to the source code.
`ATBMX`, `MAXATC`, `MAXCB` (located in dimens.fcm [c35] or dimens_ltm.src [c39]) must be increased. `ATBMX` determines
the number of bonds allowed per atom, `MAXATC` describes the maximum number of atom core, and `MAXCB` determines the
maximum number of bond parameters in the CHARMM parameter file. Additionally, `CHSIZE` may need to be increased if
using an earlier version (< c36).


Examples
--------
    $ fluctmatch simulate -s fluctmatch.psf -f cg.dcd -d fluctmatch/1 -l fluctmatch.log --target target.ic \
        --param fluctmatch.str --exec /opt/local/bin/charmm -p fluctmatch -t 300.0 --max 200 --tol 0.001
"""

import csv
from pathlib import Path

import click
import MDAnalysis as mda
from click_help_colors import HelpColorsCommand
from loguru import logger

from fluctmatch import __copyright__
from fluctmatch.fm.charmm.fluctmatch import CharmmFluctuationMatching
from fluctmatch.libs.logging import config_logger


@click.command(
    cls=HelpColorsCommand,
    help=f"{__copyright__}\nPerform fluctuation matching.",
    short_help="Perform fluctuation matching",
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
    "-d",
    "--directory",
    metavar="DIR",
    default=Path.cwd().joinpath("fluctmatch"),
    show_default=True,
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    help="Parent directory",
)
@click.option(
    "-l",
    "--logfile",
    metavar="LOGFILE",
    show_default=True,
    default=Path.cwd() / Path(__file__).with_suffix(".log"),
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to log file",
)
@click.option(
    "--target",
    metavar="TARGET",
    show_default=True,
    default=Path.cwd().joinpath("target.ic"),
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to target fluctuations",
)
@click.option(
    "--param",
    metavar="PARAM",
    show_default=True,
    default=Path.cwd().joinpath("fluctmatch.str"),
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to parameter file",
)
@click.option(
    "-e",
    "--exec",
    "executable",
    metavar="FILE",
    type=click.Path(exists=True, file_okay=True, resolve_path=True),
    help="CHARMM executable file",
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
    "-t",
    "--temperature",
    metavar="TEMP",
    type=click.FloatRange(min=0.0, max=1000.0, clamp=True),
    default=300.0,
    show_default=True,
    help="Temperature of simulation",
)
@click.option(
    "--max",
    "max_cycles",
    metavar="MAXCYCLES",
    type=click.IntRange(min=1, clamp=True),
    default=300,
    show_default=True,
    help="maximum number of fluctuation matching cycles",
)
@click.option(
    "--tol",
    "tolerance",
    metavar="TOL",
    type=click.FloatRange(min=1e-4, clamp=True),
    default=1.0e-4,
    show_default=True,
    help="Tolerance level between simulations",
)
@click.option("--restart", is_flag=True, help="Restart simulation")
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
def simulate(
    topology: Path,
    trajectory: Path,
    directory: Path,
    logfile: Path,
    target: Path,
    param: Path,
    executable: Path,
    prefix: Path,
    temperature: float,
    max_cycles: int,
    tolerance: float,
    restart: bool,
    verbosity: str,
) -> None:
    """Run fluctuation matching simulation.

    Parameters
    ----------
    topology : Path, default=$CWD/input.parm7
        Topology file
    trajectory : Path, default=$CWD/input.nc
        Trajectory file
    directory : Path, default=$CWD/fluctmatch
        Output directory
    logfile : Path, default=$CWD/setup.log
        Location of log file
    target : Path, default=$CWD/target.ic
        Location of target fluctuations
    param : Path, default=$CWD/fluctmatch.str
        Location of parameter file
    executable : Path
        Location of executable file
    prefix : Path, default=fluctmatch
        Filename prefix
    temperature : float, default=300.0
        Simulation temperature
    max_cycles : int, default=300.
        Number of fluctuation matching cycles to complete
    tolerance : float, default=0.0001
        Tolerance level for r.m.s.e. if simulation should stop prior to `max_cycles`
    restart : bool
        Restart the simulation from a previous run
    verbosity : {'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'}
        Level of verbosity for logging output
    """
    config_logger(name=__name__, logfile=logfile, level=verbosity)
    click.echo(__copyright__)

    universe = mda.Universe(topology, trajectory)

    fm = CharmmFluctuationMatching(universe, output_dir=directory, temperature=temperature, prefix=prefix)
    fm.load_target(target).load_parameters(param)

    error_file = directory.joinpath("rmse.csv")
    mode = "a" if restart else "w"
    with error_file.open(mode=mode) as error_stream:
        writer = csv.writer(error_stream)
        writer.writerow("cycle rmse".split())
        for i in range(max_cycles):
            fm.simulate(executable=executable)
            rmse = fm.calculate()

            line = f"{i + 1:>8d} {rmse:>.8f}"
            logger.info(line)
            writer.writerow(f"{i + 1:d} {rmse:.4f}".split())
            if rmse <= tolerance:
                msg = (
                    f"Tolerance level ({tolerance}) for fluctuations exceeded. Simulation stopping after cycle {i + 1}."
                )
                logger.warning(msg)
                break
