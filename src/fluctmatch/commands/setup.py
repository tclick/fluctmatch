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
r"""Create subdirectories representing individual windows for fluctuation matching.

This script determines the number of windows needed for fluctuation matching by analysis of the provided trajectory
and the given window size, `winsize`. From this analysis, the number of frames within the trajectory is calculated
and the respective start and stop numbers are noted. Additionally, subdirectories will be created for the number of
windows. Depending upon the number of windows, smaller numbered subdirectories will begin with a '0' to ensure proper
sorting when either listed using commands like `ls` or globbed in other scripts. A JSON file will be written with the
style: {'<subdir>': {'start': start, 'stop': stop},}

Usage
-----
    fluctmatch setup -s <topology> -f <trajectory> -d <directory> -o <json-output> -l <logfile> -w <win-size>

Notes
-----
Windows will have overlapping frames to ensure that information is not lost. For instance, a window with 10,000 frames
will have starting frames at 0, 5000, 10000, 15000, etc. This means that a 1-ns trajectory with output ever 1 ps and
a window size of 10,000 will have 99 windows: :math:`n_windows = \frac{n_frames}/{winsize} - 1`

Examples
--------
    $ fluctmatch setup -s trex1.tpr -f trex1.xtc -d fluctmatch -o cg.json -l setup.log -w 10000
"""

import json
from itertools import zip_longest
from pathlib import Path

import click
import MDAnalysis as mda
from click_help_colors import HelpColorsCommand
from loguru import logger

from fluctmatch import __copyright__
from fluctmatch.commands import FILE_MODE
from fluctmatch.libs.logging import config_logger

__help__ = """Setup subdirectories

This script determines the number of windows needed for fluctuation matching by analysis of the provided trajectory
and the given window size, `winsize`. From this analysis, the number of frames within the trajectory is calculated
and the respective start and stop numbers are noted. Additionally, subdirectories will be created for the number of
windows. Depending upon the number of windows, smaller numbered subdirectories will begin with a '0' to ensure proper
sorting when either listed using commands like `ls` or globbed in other scripts. A JSON file will be written with the
style: {'<subdir>': {'start': start, 'stop': stop},}."""


@click.command(
    cls=HelpColorsCommand,
    help=f"{__copyright__}\n{__help__}",
    short_help="Create directories for fluctuation matching",
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
    "-o",
    "--output",
    "output",
    metavar="JSON",
    show_default=True,
    default=Path.cwd().joinpath("setup.json"),
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="JSON file",
)
@click.option(
    "-l",
    "--logfile",
    metavar="FILE",
    show_default=True,
    default=Path.cwd() / Path(__file__).with_suffix(".log"),
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to log file",
)
@click.option(
    "-w",
    "--winsize",
    metavar="WINSIZE",
    show_default=True,
    default=10000,
    type=click.IntRange(min=2, max_open=True, clamp=True),
    help="Size of each window",
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
def setup(
    topology: Path, trajectory: Path, directory: Path, winsize: int, output: Path, logfile: Path, verbosity: str
) -> None:
    """Create simulation directories.

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
    winsize : int, default=10000
        Window size
    output : Path, default=$CWD/setup.json
        JSON file
    verbosity : str, default=INFO
        Level of verbosity for logging output
    """
    config_logger(name=__name__, logfile=logfile, level=verbosity)
    click.echo(__copyright__)

    n_frames = mda.Universe(topology, trajectory).trajectory.n_frames
    if n_frames % 2 > 0:
        msg = "An uneven number of frames exists. This will cause the size of windows to be unequal."
        logger.warning(msg)
        UserWarning(msg)
    if winsize > n_frames:
        msg = f"Window size is larger than the number of frames. ({winsize} > {n_frames})"
        logger.exception(msg)
        raise ValueError(msg)
    if winsize == n_frames:
        msg = "Window size is equivalent to the number of frames. You will only have one subdirectory."
        logger.warning(msg)
        UserWarning(msg)

    half_size = winsize // 2
    total_windows = (n_frames // half_size) - 1
    if winsize % 2 > 0 or n_frames % half_size > 0:
        msg = "Unexpected results may occur with an uneven number of frames."
        logger.warning(msg)
        UserWarning(msg)

    start = 1 if trajectory.suffix == ".trr" or trajectory.suffix == ".xtc" or trajectory.suffix == ".tng" else 0
    stop = n_frames + 1
    beginning = range(start, stop, half_size)
    end = range(start + winsize, stop, half_size)
    logger.debug(f"start: {start}; stop: {stop}; half_size: {half_size}; total_windows: {total_windows}")

    # Create directories for fluctuation matching
    trajectory_range = zip_longest(beginning, end)
    width = len(str(total_windows))
    ranges = {
        directory.joinpath(f"{n:>0{width}d}").as_posix(): {"start": i, "stop": j}
        for n, (i, j) in enumerate(trajectory_range, 1)
        if j is not None
    }

    # Create the parent subdirectory
    with directory.joinpath(output).open(mode="w", newline="") as json_file:
        logger.info(f"Writing {output}")
        json.dump(ranges, json_file, sort_keys=True, indent=4)

    # Create subdirectories
    for subdirectory in ranges:
        Path(subdirectory).mkdir(parents=True, exist_ok=True, mode=FILE_MODE)
