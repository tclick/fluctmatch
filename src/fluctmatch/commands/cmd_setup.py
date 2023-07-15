# ------------------------------------------------------------------------------
#  fluctmatch
#  Copyright (c) 2013-2023 Timothy H. Click, Ph.D.
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

import csv
from itertools import zip_longest
from pathlib import Path

import click
import MDAnalysis as mda
from click_extra import help_option, timer_option

from .. import __copyright__, config_logger
from . import FILE_MODE


@click.command(
    "setup",
    help=f"{__copyright__}\nCreate simulation subdirectories.",
    short_help="Create subdirectories for fluctuation matching",
)
@click.option(
    "-s",
    "--topology",
    metavar="FILE",
    default=Path.cwd() / "input.parm7",
    show_default=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path),
    help="Topology file",
)
@click.option(
    "-f",
    "--trajectory",
    metavar="FILE",
    default=Path.cwd() / "input.nc",
    show_default=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path),
    help="Trajectory file",
)
@click.option(
    "-o",
    "--outdir",
    metavar="DIR",
    default=Path.cwd() / "fluctmatch",
    show_default=True,
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    help="Parent directory",
)
@click.option(
    "-l",
    "--logfile",
    metavar="LOG",
    show_default=True,
    default=Path.cwd() / Path(Path(__file__).stem[4:]).with_suffix(".log"),
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Log file",
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
    "--csv",
    "windows_output",
    metavar="CSV",
    show_default=True,
    default=Path("setup.csv"),
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="CSV file",
)
@click.option(
    "-v",
    "--verbose",
    metavar="VERBOSE",
    show_default=True,
    default="INFO",
    type=click.Choice("CRITICAL ERROR WARNING INFO DEBUG".split()),
    help="Verbosity level",
)
@help_option()
@timer_option()
def cli(
    topology: Path, trajectory: Path, outdir: Path, logfile: Path, winsize: int, windows_output: Path, verbose: str
) -> None:
    """Create simulation subdirectories.

    Parameters
    ----------
    topology : Path
        Topology file
    trajectory : Path
        Trajectory file
    outdir : Path
        Output directory
    logfile : Path
        Location of log file
    winsize : int
        Window size
    windows_output : Path
        CSV file
    verbose : str
        Level of verbosity for logging output
    """
    logger = config_logger(logfile=logfile.as_posix(), level=verbose)
    click.echo(__copyright__)

    n_frames = mda.Universe(topology, trajectory).trajectory.n_frames
    if winsize > n_frames:
        msg = f"Window size is larger than the number of frames. ({winsize} > {n_frames})"
        raise ValueError(msg)
    if winsize == n_frames:
        logger.warning("Window size is equivalent to the number of frames. You will only have two subdirectories.")

    half_size = winsize // 2
    total_windows = (n_frames // half_size) - 1

    start = 1 if trajectory.suffix == ".trr" or trajectory.suffix == ".xtc" or trajectory.suffix == ".tng" else 0
    stop = n_frames + 1
    beginning = range(start, stop, half_size)
    end = range(start + winsize, stop, half_size)
    logger.debug(f"start: {start}; stop: {stop}; half_size: {half_size}; total_windows: {total_windows}")

    # Create subdirectories for fluctuation matching
    trajectory_range = zip_longest(beginning, end)
    ranges = []
    for n, (i, j) in enumerate(trajectory_range, 1):
        if j is not None:
            ranges.append((n, i, j))

            subdir = outdir / f"{n:d}"
            logger.debug(f"Making {subdir}")
            subdir.mkdir(mode=FILE_MODE, parents=True, exist_ok=True)

    # Create the parent subdirectory
    outdir.mkdir(mode=FILE_MODE, parents=True, exist_ok=True)
    with open(windows_output, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(ranges)
