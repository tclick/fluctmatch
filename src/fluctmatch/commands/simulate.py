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
