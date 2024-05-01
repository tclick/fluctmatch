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

import click
import click_extra

from fluctmatch import __copyright__
from fluctmatch import click_loguru


@click_extra.extra_command(
    help=f"{__copyright__}\nCreate simulation directories.",
    short_help="Create directories for fluctuation matching",
)
@click_loguru.init_logger()
@click_loguru.log_elapsed_time(level="info")
@click.help_option()
def setup(
    # self: Self,
    # topology: Path,
    # trajectory: Path,
    # outdir: Path,
    # logfile: Path,
    # winsize: int,
    # windows_output: Path,
    # nthreads: int,
    # verbose: str,
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
    nthreads : int, default=4
        Number of threads
    verbose : str, default=INFO
        Level of verbosity for logging output
    """
    click_extra.echo(__copyright__)
