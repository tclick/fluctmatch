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
# pyright: reportAttributeAccessIssue=false
"""Align a trajectory to its first frame or to a reference structure."""

from __future__ import annotations

from pathlib import Path

import click
import MDAnalysis as mda
from MDAnalysis import transformations

from fluctmatch import __copyright__
from fluctmatch.libs.logging import config_logger


@click.command(
    help=f"{__copyright__}\nAlign a trajectory.",
    short_help="Align a trajectory to the first frame or to a reference structure",
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
    metavar="WARNING",
    show_default=True,
    default=Path.cwd() / Path(__file__).with_suffix(".log"),
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to log file",
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
    "--verbosity",
    default="INFO",
    show_default=True,
    help="Minimum severity level for log messages",
)
@click.help_option("-h", "--help", help="Show this help message and exit")
def align(
    topology: Path,
    trajectory: Path,
    reference: Path,
    outdir: Path,
    logfile: Path,
    select: str,
    mass: bool,
    verbosity: str,
) -> None:
    """Align the trajectory.

    Parameters
    ----------
    topology : Path, default=$CWD/input.parm7
        Topology file
    trajectory : Path, default=$CWD/input.nc
        Trajectory file
    reference : Path, default=$CWD/ref.pdb
        Reference structure
    outdir : Path, default=$CWD
        Output directory
    logfile : Path, default=align.log
        Location of log file
    select : str, default=ca
        Atom selection
    mass : bool
        Mass-weighted alignment
    verbosity : str, default=INFO
        Level of verbosity for logging output
    """
    logger = config_logger(name=__name__, logfile=logfile, level=verbosity)

    click.echo(__copyright__)

    selection = {
        "all": "all",
        "ca": "protein and name CA",
        "cab": "protein and name CA CB",
        "backbone": "backbone or nucleicbackbone",
    }
    weight = "mass" if mass else None
    output = outdir / f"aligned_{trajectory.name}"
    click.echo(output)

    universe = mda.Universe(topology, trajectory)
    mobile = universe.select_atoms(selection[select])
    ref = mda.Universe(topology, reference).select_atoms(selection[select])

    transform = transformations.fit_rot_trans(mobile, ref, weights=weight)
    universe.trajectory.add_transformations(transform)

    with mda.Writer(output.as_posix(), n_atoms=universe.atoms.n_atoms) as out:
        logger.info("Aligning trajectory to the reference structure.")
        for _ in universe.trajectory:
            out.write(universe)
        logger.info(f"The structure has been aligned in {output}.")
