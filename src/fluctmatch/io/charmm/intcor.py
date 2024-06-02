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
# pyright: reportInvalidTypeVarUse=false, reportArgumentType=false
"""Class to read and write CHARMM internal coordinate files."""

import datetime
import getpass
from collections import OrderedDict
from pathlib import Path
from typing import Self

import MDAnalysis as mda
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from parmed.utils.fortranformat import FortranRecordReader, FortranRecordWriter

from fluctmatch.io.charmm import Bond, BondData
from fluctmatch.libs.utils import compare_dict_keys


class CharmmInternalCoordinates:
    """Initialize, read, or write CHARMM internal coordinate data."""

    def __init__(self: Self) -> None:
        """Prepare internal coordinate data."""
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

    def initialize(self: Self, universe: mda.Universe, /, data: BondData | None = None) -> None:
        """Fill the internal coordinates table with atom types and bond information.

        Parameters
        ----------
        universe : :class:`mda.Universe`
            Universe with bond information
        data : OrderedDict[tuple[str, str], float], optional
            Bond information

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

    def write(self: Self, filename: Path, /, title: list[str] | None = None) -> None:
        """Write internal coordinate data to a file.

        Parameters
        ----------
        filename : Path
            Name of output file
        title : list of str, optional
            initial information to write in the file
        """
        now: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user: str = getpass.getuser()
        _title: list[str] = title if title is not None else [f"* Created by fluctmatch on {now}.", f"* User: {user}"]
        header1 = FortranRecordWriter("20I4")
        header2 = FortranRecordWriter("2I5")
        header1_info = np.zeros(20, dtype=int)
        header1_info[0], header1_info[1] = 30, 2
        header2_info = np.array([len(self._table), 2], dtype=int)

        with filename.open(mode="w") as intcor:
            for _ in _title:
                intcor.write(_ + "\n")

            intcor.write(header1.write(header1_info) + "\n")
            intcor.write(header2.write(header2_info) + "\n")
            np.savetxt(intcor, self.table, fmt=self._writer)

    def read(self: Self, filename: Path | str) -> None:
        """Read an internal coordinate file.

        Parameters
        ----------
        filename : Path or str
            Name of internal coordinate file

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

                    cols = self._reader.read(line)
                    self._table[(cols[3], cols[6])] = np.fromiter(cols, dtype=object)
        except FileNotFoundError as err:
            logger.exception(err)
            raise
