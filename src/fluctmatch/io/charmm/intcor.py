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
# pyright: reportInvalidTypeVarUse=false, reportArgumentType=false, reportReturnType=false
"""Class to read and write CHARMM internal coordinate files."""

import datetime
import getpass
from collections import OrderedDict
from pathlib import Path
from typing import Self

import MDAnalysis as mda
import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray
from parmed.utils.fortranformat import FortranRecordReader, FortranRecordWriter

from fluctmatch.io.base import IOBase
from fluctmatch.io.charmm import Bond, BondData
from fluctmatch.libs.utils import compare_dict_keys


class CharmmInternalCoordinates(IOBase):
    """Initialize, read, or write CHARMM internal coordinate data."""

    def __init__(self: Self) -> None:
        """Prepare internal coordinate data."""
        super().__init__()
        self._table: OrderedDict[Bond, NDArray] = OrderedDict([])
        self._writer: str = (
            "%10d %-8s %-8d %-8s: %-8s %-8d %-8s: %-8s %-8s %-8s: %-8s %-8s %-8s:%12.6f%12.4f%12.4f%12.4f%12.6f"
        )
        self._reader: FortranRecordReader = FortranRecordReader(
            "I10,2(1X,A8,1X,I8,1X,A8,1X),2(1X,A8,1X,A8,1X,A8,1X),F12.6,3F12.4,F12.6"
        )

    @property
    def table(self: Self) -> NDArray:
        """Return internal coordinate data.

        Returns
        -------
        numpy.ndarray
            Internal coordinates data
        """
        return np.array(list(self._table.values()), dtype=object)

    @property
    def data(self: Self) -> BondData:
        """Return bond data.

        Returns
        -------
        OrderedDict[tuple[str, str],float]
            Bond information
        """
        return OrderedDict({k: v[-5] for k, v in self._table.items()})

    @data.setter
    def data(self: Self, data: BondData) -> None:
        """Set bond data.

        Parameters
        ----------
        data : OrderedDict[tuple[str, str], float]
            Data to include in the `r_IJ` column

        Raises
        ------
        ValueError
            if number of bonds, force constants, or bond lengths do not match
        """
        try:
            compare_dict_keys(self._table, data, message="Provided data does not match bonds in universe.")
            for k, v in data.items():
                self._table[k][-5] = v
        except ValueError as exc:
            exc.add_note("One or more of the bonds defined within the provided data does not exist in the universe.")
            logger.exception(exc)
            raise

    def initialize(self: Self, universe: mda.Universe, /, data: BondData | None = None) -> Self:
        """Fill the internal coordinates table with atom types and bond information.

        Parameters
        ----------
        universe : :class:`mda.Universe`
            Universe with bond information
        data : OrderedDict[tuple[str, str], float], optional
            Bond information

        Returns
        -------
        CharmmInternalCoordinates
            Self

        Raises
        ------
        MDAnalysis.NoDataError
            if no bond data exists
        ValueError
            if number of bonds, force constants, or bond lengths do not match
        """
        other_info: list[str] = 2 * ["??", "??", "??"]
        info: list[float] = np.zeros(5, dtype=universe.bonds.values().dtype).tolist()
        table: OrderedDict[Bond, NDArray] = OrderedDict([])
        for i, (key, bond) in enumerate(zip(universe.bonds.topDict, universe.bonds, strict=True), 1):
            atom1, atom2 = bond.atoms
            atom1_info: list[str | int] = [i, f"{atom1.segid}", atom1.resid, f"{atom1.name}"]
            atom2_info: list[str | int] = [f"{atom2.segid}", atom2.resid, f"{atom2.name}"]
            table[key] = np.fromiter(atom1_info + atom2_info + other_info + info, dtype=object)

        try:
            if data is not None:
                compare_dict_keys(data, table, message="Provided data does not match bonds in universe.")
                for key, value in data.items():
                    table[key][-5] = value
        except ValueError as exc:
            exc.add_note("One or more of the bonds defined within the provided data does not exist in the universe.")
            logger.exception(exc)
            raise

        self._table = table.copy()
        return self

    def write(self: Self, filename: Path | str, /, title: list[str] | None = None) -> None:
        """Write internal coordinate data to a file.

        Parameters
        ----------
        filename : Path or str
            Name of output file
        title : list of str, optional
            initial information to write in the file
        """
        now: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user: str = getpass.getuser()
        _title: list[str] = (
            title if title is not None else [f"* Created by fluctmatch on {now}.\n", f"* User: {user}\n"]
        )
        header1 = FortranRecordWriter("20I4")
        header2 = FortranRecordWriter("2I5")
        header1_info = np.zeros(20, dtype=int)
        header1_info[0], header1_info[1] = 30, 2
        header2_info = np.array([len(self._table), 2], dtype=int)

        with Path(filename).open(mode="w") as intcor:
            intcor.writelines(_title)
            intcor.write(header1.write(header1_info) + "\n")
            intcor.write(header2.write(header2_info) + "\n")
            np.savetxt(intcor, self.table, fmt=self._writer)

    def read(self: Self, filename: Path | str) -> Self:
        """Read an internal coordinate file.

        Parameters
        ----------
        filename : Path or str
            Name of internal coordinate file

        Returns
        -------
        CharmmInternalCoordinates
            Self

        Raises
        ------
        FileNotFoundError
            if parameter, topology, or stream file not found
        """
        try:
            with Path(filename).open(mode="r") as intcor:
                logger.info(f"Reading internal coordinate data from {filename}")
                for line in intcor:
                    if line.startswith("*"):
                        continue
                    if all(c.isdigit() for c in line.split()):
                        continue

                    cols = [str.strip(_) if isinstance(_, str) else _ for _ in self._reader.read(line)]
                    self._table[(cols[3], cols[6])] = np.fromiter(cols, dtype=object)
        except FileNotFoundError as err:
            logger.exception(err)
            raise

        return self

    def to_dataframe(self: Self) -> pd.DataFrame:
        """Convert internal coordinate data to pandas dataframe.

        Columns will be labeled according to the internal coordinate file.

        Returns
        -------
        pandas.DataFrame
            Internal coordinate data
        """
        cols = tuple("# segidI resI I segidJ resJ J segidK resK K segidL resL L r_IJ T_IJK P_IJKL T_JKL r_KL".split())
        table = pd.DataFrame.from_dict(self._table, orient="index")
        table.columns = cols
        return table.set_index("#")

    def to_series(self: Self) -> pd.Series:
        """Convert internal coordinate data to pandas Series containing only the `r_IJ` data.

        The series will be indexed by the first six columns.

        Returns
        -------
        pandas.Series
            Bond distance data between atoms I and J
        """
        table = self.to_dataframe()
        table = table.set_index(table.columns.to_list()[:6])
        return table["r_IJ"]
