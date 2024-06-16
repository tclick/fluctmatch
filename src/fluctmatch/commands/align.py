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
# pyright: reportAttributeAccessIssue=false
r"""Align a trajectory to reference coordinates.

This script aligns a trajectory to a reference structure. If no reference structure is provided, the first frame of
the trajectory will be used. The alignment will attempt to remove translational and rotational motion while seeing to
minimize the r.m.s.d. of each frame. The frames will be aligned with the reference structure according to the atom(s)
provided by the `--select` option, and the alignment can be mass-weighted. The aligned trajectory will be written to
a file with the name 'aligned_{trajectory}'

Usage
-----
    $ fluctmatch align -s <topology> -f <trajectory> -r <reference> -d <directory> -l <logfile> -s <selection> --mass

Notes
-----
.. warn:: Depending upon the length of the trajectory and the size of the universe, the alignment process can take
several minutes.

Examples
--------
    $ fluctmatch align -s trex1.tpr -f trex1.xtc -r trex1.gro -d fluctmatch -l align.log -s cab --mass
"""

from pathlib import Path

import click
import MDAnalysis as mda
import numpy as np
from click_help_colors import HelpColorsCommand
from loguru import logger
from MDAnalysis.analysis.align import AlignTraj

import fluctmatch.model.selection  # noqa: F401
from fluctmatch import __copyright__
from fluctmatch.commands import FILE_MODE
from fluctmatch.libs.logging import config_logger

SELECTION: dict[str, str] = {
    "all": "all",
    "protein": "protein and not name H*",
    "ca": "calpha",
    "cab": "calpha or cbeta",
    "backbone": "backbone or nucleicbackbone",
    "sugar": "nucleicsugar and not name H*",
    "nucleic": "nucleic and not name H*",
}

__help__ = """Trajectory alignment

This script aligns a trajectory to a reference structure. If no reference structure is provided, the first frame of
the trajectory will be used. The alignment will attempt to remove translational and rotational motion while seeing to
minimize the r.m.s.d. of each frame. The frames will be aligned with the reference structure according to the atom(s)
provided by the `--select` option, and the alignment can be mass-weighted. The aligned trajectory will be written to
a file with the name 'aligned_{trajectory}'"""


@click.command(
    cls=HelpColorsCommand,
    help=f"{__copyright__}\n{__help__}",
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
    default=Path.cwd().joinpath("ref.pdb"),
    show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path),
    help="Reference file",
)
@click.option(
    "-d",
    "--directory",
    metavar="DIR",
    default=Path.cwd(),
    show_default=True,
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    help="Parent directory",
)
@click.option(
    "-l",
    "--logfile",
    metavar="FILE",
    show_default=True,
    default=Path.cwd().joinpath(__file__).with_suffix(".log"),
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to log file",
)
@click.option(
    "-t",
    "--select",
    metavar="TYPE",
    show_default=True,
    default="ca",
    type=click.Choice(list(SELECTION.keys()), case_sensitive=False),
    help="Atom selection for alignment",
)
@click.option("--mass", is_flag=True, help="Mass-weighted alignment")
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
def align(
    topology: Path,
    trajectory: Path,
    reference: Path,
    directory: Path,
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
    directory : Path, default: $CWD/output
        Path to the output directory where the aligned trajectory will be saved.
    logfile : Path, default: $CWD/logging.log
        Path to the log file for recording the process.
    select : {'ca', 'protein', 'all', 'cab', 'backbone', 'nucleic', 'sugar'}
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
    ...       directory=Path('output'),  logfile=Path('alignment.log'), select='ca', mass=False, verbosity='INFO')

    Notes
    -----
    The function initializes the logger to record messages to the specified log file and console. It selects atoms from
    the trajectory and reference structure based on the provided selection criteria. A transformation is applied to
    align the trajectory to the reference, and the aligned trajectory is written to the output directory.
    """
    config_logger(name=__name__, logfile=logfile, level=verbosity.upper())
    click.echo(__copyright__)

    weight = "mass" if mass else None
    directory.mkdir(mode=FILE_MODE, parents=True, exist_ok=True)
    prefix = directory.joinpath("aligned_").as_posix()

    universe = mda.Universe(topology, trajectory)
    ref = mda.Universe(topology, reference) if reference.is_file() else universe

    try:
        logger.info("Aligning trajectory to the reference structure.")
        verbose = verbosity.upper() == "DEBUG"
        aligned = AlignTraj(
            universe, ref, select=SELECTION[select], prefix=prefix, weight=weight, verbose=verbose
        ).run()
        rms_min, rms_max = np.min(aligned.results.rmsd), np.max(aligned.results.rmsd)
        logger.info(f"The structure has been aligned in {aligned.filename}.")
        logger.info(
            f"Comparison with reference structure: Min. r.m.s.d.: {rms_min:>8.3f} Max. r.m.s.d.: {rms_max:>8.3f}"
        )
    except ValueError as exception:
        exception.add_note(
            "Could not align trajectory to reference structure. Please ensure that you provided a valid selection "
            "criteria for your system."
        )
        logger.exception(exception)
        raise
