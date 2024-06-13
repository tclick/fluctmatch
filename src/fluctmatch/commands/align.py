# ---------------------------------------------------------------------------------------------------------------------
# fluctmatch
# Copyright (c) 2013-2024 Timothy H. Click, Ph.D.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
# later version.

# This program is distributed in the hope that it will be useful, # but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.

# You should have received a copy of the GNU General Public License along with this program.
# If not, see <[1](https://www.gnu.org/licenses/)>.
# ---------------------------------------------------------------------------------------------------------------------
# pyright: reportAttributeAccessIssue=false
"""Align a trajectory to its first frame or to a reference structure."""

from __future__ import annotations

from pathlib import Path

import click
import MDAnalysis as mda
from click_help_colors import HelpColorsCommand
from loguru import logger
from MDAnalysis import transformations

from fluctmatch import __copyright__
from fluctmatch.commands import FILE_MODE
from fluctmatch.libs.logging import config_logger


@click.command(
    cls=HelpColorsCommand,
    help=f"{__copyright__}\nAlign a trajectory.",
    short_help="Align a trajectory to the first frame or to a reference structure",
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
    type=click.Choice("INFO DEBUG WARNING ERROR CRITICAL".split()),
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
    """Align a molecular dynamics trajectory to a reference structure.

    This function performs an alignment of a trajectory against a reference structure based on the specified atom
    selection. If a reference structure is not provided or cannot be found, alignment will occur using the first
    frame of the provided trajectory. The alignment can be mass-weighted. The aligned trajectory is saved in the
    specified output directory. Logging of the process is configurable through the verbosity level and the logfile path.

    Parameters
    ----------
    topology : Path, default: $CWD/input.parm7
        Path to the topology file.
    trajectory : Path, default: $CWD/input.parm7
        Path to the trajectory file.
    reference : Path, default: $CWD/ref.pdb
        Path to the reference structure file. If the reference does not exist, the first frame of the trajectory
        will be used.
    outdir : Path, default: $CWD/output
        Path to the output directory where the aligned trajectory will be saved.
    logfile : Path, default: $CWD/logging.log
        Path to the log file for recording the process.
    select : {'ca', 'all', 'cab', 'backbone'}
        Atom selection type for alignment.
    mass : bool, default: False
        Perform a mass-weighted alignment if True.
    verbosity : {'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'}
        Logging verbosity level.

    Returns
    -------
    None

    Examples
    --------
    To align 'simulation.nc' to 'reference.pdb' using C-alpha atoms for alignment and save the aligned trajectory in
    the 'output' directory:

    >>> align(topology=Path('topology.parm7'), trajectory=Path('simulation.nc'), reference=Path('reference.pdb'),
    ...       outdir=Path('output'),  logfile=Path('alignment.log'), select='ca', mass=False, verbosity='INFO')

    Notes
    -----
    The function initializes the logger to record messages to the specified log file and console. It selects atoms from
    the trajectory and reference structure based on the provided selection criteria. A transformation is applied to
    align the trajectory to the reference, and the aligned trajectory is written to the output directory.
    """
    config_logger(name=__name__, logfile=logfile, level=verbosity)
    click.echo(__copyright__)

    selection = {
        "all": "all",
        "ca": "protein and name CA",
        "cab": "protein and name CA CB",
        "backbone": "backbone or nucleicbackbone",
    }
    weight = "mass" if mass else None
    outdir.mkdir(mode=FILE_MODE, parents=True, exist_ok=True)
    output = outdir.joinpath(f"aligned_{trajectory.name}")
    click.echo(output)

    universe = mda.Universe(topology, trajectory)
    mobile = universe.select_atoms(selection[select])
    if reference.is_file():
        ref = mda.Universe(topology, reference).select_atoms(selection[select])
    else:
        logger.warning("Reference file defaulting to the first frame of the trajectory.")
        ref = mobile

    transform = transformations.fit_rot_trans(mobile, ref, weights=weight)
    universe.trajectory.add_transformations(transform)

    with mda.Writer(output.as_posix(), n_atoms=universe.atoms.n_atoms) as out:
        logger.info("Aligning trajectory to the reference structure.")
        for _ in universe.trajectory:
            out.write(universe.atoms)
        logger.info(f"The structure has been aligned in {output}.")
