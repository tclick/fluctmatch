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
# pyright: reportArgumentType=false, reportAssignmentType=false, reportAttributeAccessIssue=false
# pyright: reportPossiblyUnboundVariable=false, reportOptionalMemberAccess=false
"""Transformation of all-atom systems.

This script transforms an all-atom system to a coarse-grain (CG) system. Several models currently can be selected and
combined. Additionally, a user can transform a CG model into an elastic network model, which contains bonds defined
by `--rmax`. The user needs to provide the location of the topology and trajectory files, and new topology and
trajectory files will be written with 'prefix' as the stem of the filename. The CG beads can either be based upon
center of geometry or center of mass depending upon the inclusion of `--com`.

Note: The transformation may take several minutes depending upon the size of the system, the number of models
selected, and the length of the trajectory.
"""

import importlib
import pkgutil
from pathlib import Path

import click
import MDAnalysis as mda
from click_help_colors import HelpColorsCommand
from loguru import logger
from MDAnalysis.analysis import align

import fluctmatch.model
from fluctmatch import __copyright__
from fluctmatch.libs import utils
from fluctmatch.libs.logging import config_logger
from fluctmatch.model.base import CoarseGrainModel, coarse_grain

for _, name, _ in pkgutil.iter_modules(fluctmatch.model.__path__, fluctmatch.model.__name__ + "."):
    importlib.import_module(name)


@click.command(
    cls=HelpColorsCommand,
    help=f"{__copyright__}\n{__doc__}",
    short_help="Convert an all-atom system to a coarse-grain model.",
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
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path),
    help="Topology file",
)
@click.option(
    "-f",
    "--trajectory",
    metavar="FILE",
    default=Path.cwd().joinpath("input.nc"),
    show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path),
    help="Trajectory file",
)
@click.option(
    "-l",
    "--logfile",
    metavar="WARNING",
    show_default=True,
    default=Path.cwd().joinpath(__file__).with_suffix(".log"),
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to log file",
)
@click.option(
    "-o",
    "--outdir",
    metavar="DIR",
    show_default=True,
    default=Path.cwd(),
    type=click.Path(exists=False, file_okay=False, resolve_path=True, path_type=Path),
    help="Directory",
)
@click.option(
    "-p",
    "--prefix",
    metavar="PREFIX",
    default="cg",
    show_default=True,
    type=click.STRING,
    help="Prefix for filenames",
)
@click.option(
    "--start",
    metavar="START",
    type=click.INT,
    default=0,
    show_default=True,
    help="Starting frame number",
)
@click.option(
    "--stop",
    metavar="STOP",
    type=click.INT,
    default=-1,
    show_default=True,
    help="Final frame number",
)
@click.option(
    "--rmin",
    metavar="DIST",
    type=click.FLOAT,
    default=0.0,
    show_default=True,
    help="Minimum distance between bonds",
)
@click.option(
    "--rmax",
    metavar="DIST",
    type=click.FLOAT,
    default=10.0,
    show_default=True,
    help="Maximum distance between bonds",
)
@click.option(
    "-m",
    "--model",
    metavar="TYPE",
    show_default=True,
    default=["calpha"],
    type=click.Choice(list(coarse_grain.keys()), case_sensitive=False),
    multiple=True,
    help="Atom selection for alignment",
)
@click.option(
    "--com / --cog",
    "com",
    default=True,
    show_default=True,
    help="Use either center of mass or center of geometry",
)
@click.option("--guess", is_flag=True, help="Guess angles, dihedrals, and improper dihedrals")
@click.option("--uniform", is_flag=True, help="Set uniform mass of beads to 1.0")
@click.option("--write", "write_traj", is_flag=True, help="Convert the trajectory file")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
@click.option(
    "--list",
    "model_list",
    is_flag=True,
    help="List available core with their descriptions",
)
@click.option(
    "-v",
    "--verbosity",
    default="INFO",
    show_default=True,
    help="Minimum severity level for log messages",
)
@click.help_option("-h", "--help", help="Show this help message and exit")
def convert(
    topology: Path,
    trajectory: Path,
    logfile: Path,
    outdir: Path,
    prefix: str,
    start: int,
    stop: int,
    rmin: float,
    rmax: float,
    model: list[str],
    com: bool,
    guess: bool,
    uniform: bool,
    write_traj: bool,
    overwrite: bool,
    model_list: bool,
    verbosity: str,
) -> None:
    """Convert an all-atom system to a coarse-grain model.

    Parameters
    ----------
    topology : Path
        topology file
    trajectory : Path
        trajectory file
    logfile : Path
        log file
    outdir : Path
        output directory
    prefix : str
        filename stem
    start : int
        starting frame number
    stop : int
        ending frame number
    rmin : float
        minimum distance between bonds
    rmax : float
        maximum distance between bonds
    model : Sequence[str]
        list of possible coarse-grain models
    com : bool
        center of mass or center of geometry for beads
    guess : bool
        guess angles, dihedrals, and improper dihedrals
    uniform : bool
        set uniform mass of beads to 1.0
    write_traj : bool
        write trajectory file
    overwrite : bool
        overwrite existing files
    model_list : bool
        list available coarse-grain models
    verbosity : str
        minimum severity level for log messages
    """
    click.echo(__copyright__)
    if model_list:
        print("Available models:")
        for key, model in coarse_grain.items():
            print(f"{key.lower():>10}: {model.description}")
        return

    config_logger(name=__name__, logfile=logfile, level=verbosity)

    filename = outdir.joinpath(prefix)
    universe = mda.Universe(topology, trajectory)

    if "enm" in model:
        cg_model = coarse_grain.get(
            "enm", universe, com=com, guess=guess, uniform=uniform, start=start, stop=stop, rmin=rmin, rmax=rmax
        )
    else:
        multiverse: list[mda.Universe] = []
        for model_type in model:
            logger.info(f"Converting all-atom system to {model_type}.")
            cg_model: CoarseGrainModel = coarse_grain.get(
                model_type, universe, com=com, guess=guess, uniform=uniform, start=start, stop=stop
            )
            multiverse.append(cg_model.transform())
        logger.info(f"Number of multiverses: {len(multiverse)}")
        universe = utils.merge(*multiverse)

    # Write CHARMM PSF file.
    try:
        new_file = filename.with_suffix(".psf")
        logger.info(f"Saving topology to {new_file}")
        universe.atoms.convert_to("PARMED").save(new_file.as_posix(), overwrite=overwrite)
    except OSError as err:
        message = f"File {new_file} already exists. To overwrite, please use '--overwrite'."
        err.add_note(message)
        logger.exception(message)
        raise

    # Determine the average structure of the trajectory and write a CHARMM coordinate file.
    verbose = verbosity == "DEBUG"
    average = align.AverageStructure(universe).run(start=start, stop=stop, verbose=verbose)
    average_model: mda.Universe = average.results.universe
    try:
        new_file = filename.with_suffix(".crd")
        logger.info(f"Saving average coordinates to {new_file}")
        average_model.atoms.convert_to("PARMED").save(new_file.as_posix(), overwrite=overwrite)
        # Create symbolic link to `filename.cor` for viewing in VMD.
        link_file = new_file.with_suffix(".cor")
        if not link_file.exists():
            logger.info(f"Linking {new_file} to {link_file} for viewing in VMD.")
            link_file.symlink_to(new_file, target_is_directory=False)
    except OSError as err:
        message = f"File {new_file} already exists. To overwrite, please use '--overwrite'."
        err.add_note(message)
        logger.exception(message)
        raise

    if write_traj:
        with mda.Writer(filename.with_suffix(".dcd").as_posix(), n_atoms=cg_model.atoms.n_atoms) as writer:
            for _ in universe.trajectory:
                writer.write(cg_model.atoms)
