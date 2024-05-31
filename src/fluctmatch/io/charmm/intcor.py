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
from pathlib import Path
from typing import Self

import MDAnalysis as mda
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from parmed.utils.fortranformat import FortranRecordReader, FortranRecordWriter


class CharmmInternalCoordinates:
    """Initialize, read, or write CHARMM internal coordinate data."""

    def __init__(self: Self) -> None:
        """Prepare internal coordinate data."""
        self._table: NDArray = np.empty(shape=(0, 22), dtype=object)
        self._writer: str = (
            "%10d %-8s %-8d %-8s: %-8s %-8d %-8s: %-8s %-8s %-8s: %-8s %-8s %-8s:%12.6f%12.4f%12.4f%12.4f%12.6f"
        )
        self._reader: FortranRecordReader = FortranRecordReader(
            "I10,2(1X,A8,1X,I8,1X,A8,1X),2(1X,A8,1X,A8,1X,A8,1X),F12.6,3F12.4,F12.6"
        )

    @property
    def table(self: Self) -> NDArray:
        """Return internal coordinate data."""
        return self._table

    @property
    def data(self: Self) -> NDArray:
        """Return bond data."""
        return self._table[:, -5].astype(float)

    @data.setter
    def data(self: Self, data: NDArray) -> None:
        """Set bond data."""
        if self._table[:, -5].size != data.size:
            message = "Size of data array different from internal coordinate data."
            logger.exception(message)
            raise IndexError(message)

        self._table[:, -5] = data.astype(object)

    def initialize(self: Self, universe: mda.Universe, data: NDArray) -> None:
        """Fill the internal coordinates table with atom types and bond information.

        Parameters
        ----------
        universe : :class:`mda.Universe`
            Universe with bond information
        data : :class:`numpy.ndarray`, optional
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
        table: list[list] = []
        for i, (bond, value) in enumerate(zip(universe.bonds, data, strict=True), 1):
            atom1, atom2 = bond.atoms
            atom1_info: list[str | int] = [i, f"{atom1.segid}", atom1.resid, f"{atom1.name}"]
            atom2_info: list[str | int] = [f"{atom2.segid}", atom2.resid, f"{atom2.name}"]
            info[0] = value
            table.append(atom1_info + atom2_info + other_info + info)

        self._table = np.asarray(table, dtype=object)

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
        header2_info = np.array([self._table.shape[0], 2], dtype=int)

        with filename.open(mode="w") as intcor:
            for _ in _title:
                intcor.write(_ + "\n")

            intcor.write(header1.write(header1_info) + "\n")
            intcor.write(header2.write(header2_info) + "\n")
            np.savetxt(intcor, self._table, fmt=self._writer)

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
                    if not line.startswith("*"):
                        break
                for _ in range(2):
                    line = intcor.readline()
                self._table = np.asarray([self._reader.read(line) for line in intcor], dtype=object)
        except FileNotFoundError as err:
            logger.exception(err)
            raise
