# ------------------------------------------------------------------------------
#  fluctmatch
#  Copyright (c) 2023 Timothy H. Click, Ph.D.
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
"""Align a trajectory to its first frame or to a reference structure."""
from __future__ import annotations

from pathlib import Path

import click
import MDAnalysis as mda
import numpy as np
from click_extra import help_option, timer_option
from MDAnalysis.analysis.align import AlignTraj

from .. import __copyright__, config_logger


@click.command(
    "align",
    help=f"{__copyright__}\nAlign a trajectory.",
    short_help="Align a trajectory to the first frame or to a reference structure",
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
    "-r",
    "--ref",
    "reference",
    metavar="FILE",
    default=Path.cwd() / "ref.pdb",
    show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path),
    help="Reference file",
)
@click.option(
    "-o",
    "--outdir",
    metavar="DIR",
    default=Path.cwd(),
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
    "-t",
    "--select",
    metavar="TYPE",
    show_default=True,
    default="ca",
    type=click.Choice("all ca cab backbone".split()),
    help="Atom selection for alignment",
)
@click.option("--mass", is_flag=True, help="Mass-weighted alignment")
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
    topology: Path,
    trajectory: Path,
    reference: Path,
    outdir: Path,
    logfile: Path,
    select: str,
    mass: bool,
    verbose: str,
) -> None:
    """Create simulation subdirectories.

    Parameters
    ----------
    topology : Path
        Topology file
    trajectory : Path
        Trajectory file
    reference : Path
        Reference structure
    outdir : Path
        Output directory
    logfile : Path
        Location of log file
    select : str
        Atom selection
    mass : bool
        Mass-weighted alignment
    verbose : str
        Level of verbosity for logging output
    """
    logger = config_logger(logfile=logfile.as_posix(), level=verbose)
    click.echo(__copyright__)

    # Setup variables
    selection = {"all": "all", "ca": "name CA", "cab": "name CA CB", "backbone": "backbone or nucleicbackbone"}
    weight = "mass" if mass else None
    prefix = outdir / "rmsfit_"

    # Load universe and reference
    universe = mda.Universe(topology, trajectory)
    ref = mda.Universe(topology, reference) if reference.exists() else universe

    # Setup alignment
    align = AlignTraj(universe, ref, select=selection[select], prefix=prefix.as_posix(), weights=weight, verbose=True)
    if align.filename is None:
        return

    # Align trajectory
    filename = Path(align.filename)
    logger.info(f"Aligning the trajectory and saving to {filename}")
    align.run()

    # Save r.m.s.d data
    rmsd_file = filename.with_suffix(".txt")
    logger.info(f"Saving r.m.s.d. information to {rmsd_file}")
    np.savetxt(rmsd_file, align.results.rmsd, fmt="%.4f")
