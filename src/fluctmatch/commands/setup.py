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
"""Prepare subdirectories for fluctuation matching."""

from __future__ import annotations

import json
from itertools import zip_longest
from pathlib import Path

import click
import MDAnalysis as mda
from click_help_colors import HelpColorsCommand

from fluctmatch import __copyright__
from fluctmatch.commands import FILE_MODE
from fluctmatch.libs.logging import config_logger


@click.command(
    cls=HelpColorsCommand,
    help=f"{__copyright__}\nCreate simulation directories.",
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
    "-o",
    "--outdir",
    metavar="DIR",
    default=Path.cwd().joinpath("fluctmatch"),
    show_default=True,
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    help="Parent directory",
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
    "--json",
    "windows_output",
    metavar="JSON",
    show_default=True,
    default=Path.cwd().joinpath("setup.json"),
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="JSON file",
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
    "-v",
    "--verbosity",
    default="INFO",
    show_default=True,
    help="Minimum severity level for log messages",
)
@click.help_option("-h", "--help", help="Show this help message and exit")
def setup(
    topology: Path, trajectory: Path, outdir: Path, winsize: int, windows_output: Path, logfile: Path, verbosity: str
) -> None:
    """Create simulation directories.

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
    winsize : int, default=10000
        Window size
    windows_output : Path, default=$CWD/setup.json
        JSON file
    verbosity : str, default=INFO
        Level of verbosity for logging output
    """
    logger = config_logger(name=__name__, logfile=logfile, level=verbosity)
    click.echo(__copyright__)

    n_frames = mda.Universe(topology, trajectory).trajectory.n_frames
    if winsize > n_frames:
        msg = f"Window size is larger than the number of frames. ({winsize} > {n_frames})"
        logger.exception(msg)
        raise ValueError(msg)
    if winsize == n_frames:
        logger.warning("Window size is equivalent to the number of frames. You will only have one subdirectory.")

    half_size = winsize // 2
    total_windows = (n_frames // half_size) - 1

    start = 1 if trajectory.suffix == ".trr" or trajectory.suffix == ".xtc" or trajectory.suffix == ".tng" else 0
    stop = n_frames + 1
    beginning = range(start, stop, half_size)
    end = range(start + winsize, stop, half_size)
    logger.debug(f"start: {start}; stop: {stop}; half_size: {half_size}; total_windows: {total_windows}")

    # Create directories for fluctuation matching
    trajectory_range = zip_longest(beginning, end)
    width = len(str(total_windows))
    ranges = {
        str(outdir / f"{n:>0{width}d}"): {"start": i, "stop": j}
        for n, (i, j) in enumerate(trajectory_range, 1)
        if j is not None
    }

    # Create the parent subdirectory
    with windows_output.open(mode="w", newline="") as json_file:
        print(f"Writing {windows_output}")
        logger.info(f"Writing {windows_output}")
        json.dump(ranges, json_file)

    # Create subdirectories
    for subdirectory in ranges:
        Path(subdirectory).mkdir(parents=True, exist_ok=True, mode=FILE_MODE)
