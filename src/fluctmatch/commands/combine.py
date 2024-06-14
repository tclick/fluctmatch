# ---------------------------------------------------------------------------------------------------------------------
#  fluctmatch
#  Copyright (c) 2024 Timothy H. Click, Ph.D.
#
#  This file is part of fluctmatch.
#
#  Fluctmatch is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
#  License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
#  later version.
#
#  Fluctmatch is distributed in the hope that it will be useful, # but WITHOUT ANY WARRANTY; without even the implied
#  warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with this program.
#  If not, see <[1](https://www.gnu.org/licenses/)>.
#
#  Reference:
#  Timothy H. Click, Nixon Raj, and Jhih-Wei Chu. Simulation. Meth Enzymology. 578 (2016), 327-342,
#  Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics doi:10.1016/bs.mie.2016.05.024.
# ---------------------------------------------------------------------------------------------------------------------
# pyright: reportAssignmentType=false, reportAttributeAccessIssue=false, reportOperatorIssue=false
"""Combination of force constants.

This script allows the user to combine the force constants from multiple parameter files. The parameter files should be
within subdirectories. The script will load the individual parameter files and extract the force constants. The
table will only include data in ascending order of resI. The table will be saved as a CSV file with the following
columns:

    - segidI resI I segidJ resJ J dir1 dir2 ... dirX

Additionally, the user can provide filenames for interresidue (resI-resJ) force constants (--resij) and/or residue
force constants (--resi). Interresidue and residue force constants are the sum of the individual beads; both options
will include both the resI-resJ interactions but also include resJ-resI interactions because the interactions affect
both residues. The user can also filter force constants that are <resI,resI+3 (--filter), which will decrease the
cumulative force constants within the interresidue and residue tables. Like the all-bead table, the interresidue and
residue tables will be saved as a CSV file with the following columns:

    - interresidue:   segidI resI segidJ resJ dir1 dir2 ... dirX

    - residue: segidI resI I dir1 dir2 ... dirX

This script requires that `pandas` and `loguru` be installed within the Python environment you are running this
script in.
"""

from pathlib import Path

import click
import pandas as pd
from click_help_colors import HelpColorsCommand
from loguru import logger

from fluctmatch import __copyright__
from fluctmatch.io.charmm.intcor import CharmmInternalCoordinates
from fluctmatch.io.charmm.parameter import CharmmParameter
from fluctmatch.libs.logging import config_logger


@click.command(
    cls=HelpColorsCommand,
    help=f"{__copyright__}\n{__doc__}",
    short_help="Create a table of force constants from multiple parameter files.",
    help_headers_color="yellow",
    help_options_color="blue",
    context_settings={"max_content_width": 120},
)
@click.option(
    "-d",
    "--directory",
    metavar="DIR",
    default=Path.cwd(),
    show_default=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Parent directory containing subdirectories with parameter files",
)
@click.option(
    "-f",
    "--parameter",
    metavar="FILE",
    show_default=True,
    default="fluctmatch.str",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Parameter file",
)
@click.option(
    "--ic",
    "intcor",
    metavar="FILE",
    show_default=True,
    default="fluctmatch.fluct.ic",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Internal coordinates file",
)
@click.option(
    "-o",
    "--output",
    metavar="FILE",
    show_default=True,
    default=Path.cwd().joinpath("force_constants_all.csv"),
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Output file with all force constants",
)
@click.option(
    "-l",
    "--logfile",
    metavar="FILE",
    show_default=True,
    default=Path.cwd() / Path(__file__).with_suffix(".log"),
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to log file",
)
@click.option(
    "--resij",
    is_flag=True,
    show_default=True,
    help="Save interresidue force constants to resij.csv",
)
@click.option(
    "--resi",
    is_flag=True,
    show_default=True,
    help="Save residue force constants to resi.csv",
)
@click.option(
    "--filter/--no-filter",
    "filter_res",
    default=False,
    show_default=True,
    help="Filter force constants to save only >resI,resI+3; applies only to `--resij` or `--resi`",
)
@click.option(
    "-v",
    "--verbosity",
    default="INFO",
    show_default=True,
    type=click.Choice("INFO DEBUG WARNING ERROR CRITICAL".split()),
    help="Minimum severity level for log messages",
)
@click.help_option("-h", "--help", help="Show this help message and exit")
def combine(
    directory: Path,
    parameter: Path,
    intcor: Path,
    output: Path,
    logfile: Path,
    resij: bool,
    resi: bool,
    filter_res: bool,
    verbosity: str,
) -> None:
    """Read multiple parameter files and combine the force constants into a table, which will be saved as a CSV file.

    Parameters
    ----------
    directory : Path, default=$CWD
        Parent directory containing subdirectories with parameter files
    parameter : Path, default=fluctmatch.str
        Parameter file
    intcor : Path, default=fluctmatch.fluct.ic
        Internal coordinates file
    output : Path, default=force_constants_all.csv
        CSV output file containing the table of force constants
    logfile : Path, default=combine.log
        Log file
    resij : bool, default=False
        Save interresidue force constants to resij.csv
    resi : bool, default=False
        Save residue force constants to resi.csv
    filter_res : bool
        Filter force constants only to include >resI,resI+3
    verbosity : str
        Logging verbosity level

    Returns
    -------
    None
    """
    config_logger(name=__name__, logfile=logfile, level=verbosity)
    click.echo(__copyright__)
    if filter_res and not resi and not resij:
        message = "'--filter' only works if '--resi' or '--resij' are selected."
        logger.warning(message)
        raise RuntimeWarning(message)

    data_frames: list[pd.Series] = []
    for subdirectory in directory.iterdir():
        if subdirectory.is_dir():
            forces = CharmmParameter().read(subdirectory.joinpath(parameter)).forces
            ic = CharmmInternalCoordinates().read(subdirectory.joinpath(intcor))
            ic.data = forces
            series = ic.to_series()
            series.name = subdirectory.name
            data_frames.append(series)

    table: pd.DataFrame = pd.concat(data_frames, axis="columns").fillna(0.0)
    columns: list[str] = table.columns.tolist()
    table = table[sorted(columns)]
    table.reset_index().to_csv(output, index=False, float_format="%.3f")

    cols = "segidI resI I segidJ resJ J".split()
    reverse_cols = "segidJ resJ J segidI resI I".split()
    if resij:
        drop_cols = "I J".split()
        groupby_cols = "segidI resI segidJ resJ".split()
        filename = output.parent.joinpath("resij.csv")
        table2 = table.copy(deep=True).reset_index()
        table2[cols] = table.reset_index()[reverse_cols]
        resij_table = pd.concat([table, table2], axis="rows")
        if filter_res:
            filtered_cols = (resij_table["resI"] > resij_table["resJ"] + 3) | (
                resij_table["resJ"] > resij_table["resI"] + 3
            )
            resij_table = resij_table[filtered_cols]
        resij_table = resij_table.groupby(groupby_cols).sum().drop(drop_cols, axis="columns")
        resij_table.reset_index().to_csv(filename, index=False, float_format="%.3f")

    if resi:
        drop_cols = "I segidJ resJ J".split()
        groupby_cols = "segidI resI".split()
        filename = output.parent.joinpath("resi.csv")
        table2 = table.copy(deep=True).reset_index()
        table2[cols] = table.reset_index()[reverse_cols]
        resi_table = pd.concat([table, table2], axis="rows")
        if filter_res:
            filtered_cols = (resi_table["resI"] > resi_table["resJ"] + 3) | (
                resi_table["resJ"] > resi_table["resI"] + 3
            )
            resi_table = resi_table[filtered_cols]

        # Half the force constant belongs to each bead.
        resi_table = 0.5 * resi_table.groupby(groupby_cols).sum().drop(drop_cols, axis="columns")
        resi_table.reset_index().to_csv(filename, index=False, float_format="%.3f")
