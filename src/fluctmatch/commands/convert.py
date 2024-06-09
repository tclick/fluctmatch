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
# pyright: reportArgumentType=false, reportAssignmentType=false, reportAttributeAccessIssue=false
# pyright: reportPossiblyUnboundVariable=false, reportOptionalMemberAccess=false
"""Convert an all-atom system to a coarse-grain model."""

from __future__ import annotations

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
    help=f"{__copyright__}\nAA -> CG conversion.",
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
    default=Path.cwd() / Path(__file__).with_suffix(".log"),
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
