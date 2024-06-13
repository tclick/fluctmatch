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
"""Simulation for fluctuation matching."""

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
    "-o",
    "--outdir",
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
@click.option(
    "-v",
    "--verbosity",
    default="INFO",
    show_default=True,
    help="Minimum severity level for log messages",
)
@click.help_option("-h", "--help", help="Show this help message and exit")
def simulate(
    topology: Path,
    trajectory: Path,
    outdir: Path,
    logfile: Path,
    target: Path,
    param: Path,
    executable: Path,
    prefix: Path,
    temperature: float,
    max_cycles: int,
    tolerance: float,
    verbosity: str,
) -> None:
    """Run fluctuation matching simulation.

    Parameters
    ----------
    topology : Path, default=$CWD/input.parm7
        Topology file
    trajectory : Path, default=$CWD/input.nc
        Trajectory file
    outdir : Path, default=$CWD/fluctmatch
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
    verbosity : str, default=INFO
        Level of verbosity for logging output
    """
    config_logger(name=__name__, logfile=logfile, level=verbosity)
    click.echo(__copyright__)

    universe = mda.Universe(topology, trajectory)

    fm = CharmmFluctuationMatching(universe, output_dir=outdir, temperature=temperature, prefix=prefix)
    fm.load_target(target).load_parameters(param)

    error_file = outdir.joinpath("rmse.csv")
    with error_file.open("w") as error_stream:
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
