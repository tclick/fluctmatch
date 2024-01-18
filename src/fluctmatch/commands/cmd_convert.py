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
"""Convert between file types."""
# pyright: reportGeneralTypeIssues=false
import csv
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

import MDAnalysis as mda
import rich_click as click
from click_extra import help_option, timer_option
from MDAnalysis.analysis.align import AverageStructure

from fluctmatch import _MODELS, __copyright__, config_logger
from fluctmatch.commands import FILE_MODE
from fluctmatch.core.utils import modeller
from fluctmatch.libs.bond_info import BondInfo
from fluctmatch.libs.intcor import InternalCoord
from fluctmatch.libs.safe_format import safe_format


def write_average_structure(outdir: Path, prefix: str, universe: mda.Universe, *args: int) -> None:
    """Write the average structure of a trajectory.

    Parameters
    ----------
    outdir : Path
        Parent directory
    prefix : str
        Filename prefix
    universe : Universe
        an atomic universe
    args : int
        subdirectory, start, and stop
    """
    subdir, start, stop = args
    filename: Path = (outdir / str(subdir) / prefix).with_suffix(".cor")
    AverageStructure(universe, filename=filename).run(start, stop)


def write_ic_info(
    outdir: Path,
    prefix: str,
    universe: mda.Universe,
    *args: int,
    extended: bool = True,
    resid: bool = True,
    n_atoms: int | None = None,
) -> None:
    """Write the statistics of bond information to an internal coordinate file.

    Parameters
    ----------
    outdir : Path
        Parent directory
    prefix : str
        Filename prefix
    universe : Universe
        an atomic universe
    args : int
        subdirectory, start, and stop
    n_atoms : int, optional
        The number of atoms in the output trajectory.
    extended : bool, optional
        Format with wider columns than the standard column width.
    resid : bool, optional
        Include segment names within each atom definition.
    """
    subdir, start, stop = args
    average: Path = (outdir / str(subdir) / prefix).with_suffix(".average.ic")
    fluctuation: Path = (outdir / str(subdir) / prefix).with_suffix(".fluct.ic")

    average_ic = InternalCoord(n_lines=len(universe.bonds))
    fluct_ic = InternalCoord(n_lines=len(universe.bonds))
    bonds = {
        "I": universe.bonds.atom1.types.tolist(),
        "J": universe.bonds.atom2.types.tolist(),
        "segidI": universe.bonds.atom1.segids.tolist(),
        "segidJ": universe.bonds.atom2.segids.tolist(),
    }
    average_ic.data.update(bonds)
    fluct_ic.data.update(bonds)

    results = BondInfo(universe).run(start=start, stop=stop)
    average_ic.data.update({"r_IJ": results.average})
    fluct_ic.data.update({"r_IJ": results.fluctuation})

    with mda.Writer(average.as_posix(), extended=extended, resid=resid, n_atoms=n_atoms) as output:
        output.write(average_ic)
    with mda.Writer(fluctuation.as_posix(), extended=extended, resid=resid, n_atoms=n_atoms) as output:
        output.write(fluct_ic)


def write_data(outdir: Path, prefix: str, universe: mda.Universe, *args: int) -> None:
    """Write average structure and bond information to files.

    Parameters
    ----------
    outdir : Path
        Parent directory
    prefix : str
        Filename prefix
    universe : Universe
        an atomic universe
    args : int
        subdirectory, start, and stop
    """
    subdir, _, _ = args
    (outdir / str(subdir)).mkdir(mode=FILE_MODE, parents=True, exist_ok=True)
    write_average_structure(outdir, prefix, universe, *args)
    write_ic_info(outdir, prefix, universe, *args)


@click.command(
    "convert",
    help=f"{__copyright__}\nConvert a trajectory.",
    short_help="Convert a trajectory to a coarse-grain model",
)
@click.option(
    "-s",
    "topology",
    metavar="FILE",
    default=Path.cwd() / "md.tpr",
    show_default=True,
    type=click.Path(exists=False, file_okay=True, resolve_path=True, path_type=Path),
    help="Gromacs topology file (e.g., tpr gro g96 pdb brk ent)",
)
@click.option(
    "-f",
    "trajectory",
    metavar="FILE",
    default=Path.cwd() / "md.xtc",
    show_default=True,
    type=click.Path(exists=False, file_okay=True, resolve_path=True, path_type=Path),
    help="Trajectory file (e.g. xtc trr dcd)",
)
@click.option(
    "-f",
    "--data",
    metavar="FILE",
    show_default=True,
    default=Path.cwd() / "setup.csv",
    type=click.Path(exists=False, file_okay=True, resolve_path=True, path_type=Path),
    help="Frame file",
)
@click.option(
    "-l",
    "--logfile",
    metavar="LOG",
    show_default=True,
    default=Path.cwd() / "convert.log",
    type=click.Path(exists=False, file_okay=True, resolve_path=True, path_type=Path),
    help="Log file",
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
    metavar="MODEL",
    type=click.Choice(_MODELS.keys()),
    multiple=True,
    help="Model(s) to convert to",
)
@click.option(
    "-c",
    "--charmm",
    "charmm_version",
    metavar="VERSION",
    default=41,
    show_default=True,
    type=click.IntRange(min=27, max=None, clamp=True),
    help="CHARMM version",
)
@click.option(
    "--com / --cog",
    "com",
    default=True,
    show_default=True,
    help="Use either center of mass or center of geometry",
)
@click.option(
    "--extended / --standard",
    "extended",
    default=True,
    help="Output using the extended or standard columns",
)
@click.option(
    "--no-nb",
    "nonbonded",
    is_flag=True,
    help="Include nonbonded section in CHARMM parameter file",
)
@click.option(
    "--no-resid",
    "resid",
    is_flag=True,
    help="Include segment IDs in internal coordinate files",
)
@click.option(
    "--no-cmap",
    "cmap",
    is_flag=True,
    help="Include CMAP section in CHARMM PSF file",
)
@click.option(
    "--no-cheq",
    "cheq",
    is_flag=True,
    help="Include charge equilibrium section in CHARMM PSF file",
)
@click.option("--uniform", "mass", is_flag=True, help="Set uniform mass of beads to 1.0")
@click.option("--write", "write_traj", is_flag=True, help="Convert the trajectory file")
@click.option(
    "--list",
    "model_list",
    is_flag=True,
    help="List available core with their descriptions",
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
    topology: Path,
    trajectory: Path,
    data: Path,
    logfile: Path,
    outdir: Path,
    prefix: str,
    rmin: float,
    rmax: float,
    model: list[str],
    charmm_version: int,
    extended: bool,
    resid: bool,
    cmap: bool,
    cheq: bool,
    mass: bool,
    model_list: bool,
    verbose: str,
) -> None:
    """Create simulation subdirectories.

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
    verbose : str, default=INFO
        Level of verbosity for logging output
    """
    logger = config_logger(logfile=logfile.as_posix(), level=verbose)
    click.echo(__copyright__)

    if model_list:
        for k, v in _MODELS.items():
            print(safe_format("{:20}{}", k, v.description))
        return

    logger.info("Converting an all-atom universe to a coarse-grain model.")
    filename: Path = outdir / prefix
    universe: mda.Universe = modeller(topology, trajectory, model, rmin=rmin, rmax=rmax, mass=mass)
    n_atoms = universe.atoms.n_atoms

    with mda.Writer(filename.with_suffix(".dcd").as_posix()) as output:
        output.write(universe)

    if "enm" not in model_list:
        psffile = filename.with_suffix(".xplor.psf").as_posix()
        with mda.Writer(
            psffile, extended=extended, cmap=cmap, cheq=cheq, charmm_version=charmm_version, n_atoms=n_atoms
        ) as output:
            output.write(universe)

        with mda.Writer(filename.with_suffix(".str").as_posix(), n_atoms=n_atoms) as output:
            output.write(universe)

        rtffile = filename.with_suffix(".rtf").as_posix()
        with mda.Writer(rtffile, charmm_version=charmm_version, n_atoms=n_atoms) as output:
            output.write(universe)

    write = partial(write_data, outdir, prefix, universe, extended=extended, resid=resid, n_atoms=n_atoms)
    with Pool() as pool, data.open(mode="r", encoding="utf=8") as csvfile:
        reader = csv.reader(csvfile)
        data_list = (int(_) for _ in reader)
        pool.imap_unordered(write, data_list, chunksize=10)
