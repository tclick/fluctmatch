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
# pyright: reportInvalidTypeVarUse=false, reportGeneralTypeIssues=false
"""Tests to read and write internal coordinates."""

from pathlib import Path
from typing import TypeVar
from unittest.mock import patch

import MDAnalysis as mda
import pandas as pd
import pytest
from fluctmatch.libs.intcor import InternalCoord
from fluctmatch.parsers.parsers import ICParser
from numpy.testing import assert_allclose

from ..datafile import IC

TTestICReader = TypeVar("TTestICReader", bound="TestICReader")
TTestICWriter = TypeVar("TTestICWriter", bound="TestICWriter")


class TestICReader:
    """Test internal coordinate reader."""

    expected_rows: int = 7566
    expected_cols: int = 17

    @staticmethod
    @pytest.fixture()
    def reader() -> ICParser.Reader:
        """Return reader object.

        Returns
        -------
        Reader
            Internal coordinate reader class
        """
        return ICParser.Reader(IC)

    def test_reader(self: TTestICReader, reader: ICParser.Reader) -> None:
        """Test internal coordinates reader.

        GIVEN a file is given to the internal coordinate reader
        WHEN the file is parsed
        THEN an object containing the file data is created

        Parameters
        ----------
        reader : Reader
            Internal coordinates reader
        """
        with reader as ic:
            internal = ic.parse()

        assert isinstance(internal, InternalCoord)
        assert internal.data["r_IJ"].size == self.expected_rows
        assert len(internal.data.keys()) == self.expected_cols

    def test_table(self: TTestICReader, reader: ICParser.Reader) -> None:
        """Test the conversion of the internal coordinates to a `DataFrame`.

        Parameters
        ----------
        reader : Reader
            Internal coordinates reader
        """
        with reader as ic:
            internal = ic.parse()
            table = internal.create_table()

        assert isinstance(table, pd.DataFrame)
        assert table.shape == (self.expected_rows, self.expected_cols)


class TestICWriter:
    """Test internal coordinates writer."""

    @staticmethod
    @pytest.fixture()
    def u() -> InternalCoord:
        """Return internal coordinate object.

        Returns
        -------
        InternalCoord
            Internal coordinate object
        """
        return ICParser.Reader(IC).parse()

    def test_writer(self: TTestICWriter, u: InternalCoord, tmp_path: Path) -> None:
        """Test that the writer is accessed.

        GIVEN the internal coordinates
        WHEN a input file is provided
        THEN the writer should be accessed through MDAnalysis.Writer

        Parameters
        ----------
        u : InternalCoord
            internal coordinates class
        tmp_path : Path
            temporary path
        """
        filename = tmp_path / "temp.ic"
        with patch("fluctmatch.parsers.writers.IC.Writer.write") as icw, mda.Writer(filename.as_posix()) as ofile:
            ofile.write(u.create_table())
            icw.assert_called()

    def test_bond_distances(self, u: InternalCoord, tmp_path: Path) -> None:
        """Test that the writer works correctly.

        GIVEN the internal coordinates
        WHEN written to a file
        THEN the bond distances `r_IJ` should match

        Parameters
        ----------
        u : InternalCoord
            internal coordinates class
        tmp_path : Path
            temporary path
        """
        filename: Path = tmp_path / "temp.ic"
        with mda.Writer(filename.as_posix()) as ofile:
            ofile.write(u)

        u2 = ICParser.Reader((tmp_path / "temp.ic").as_posix()).parse()
        assert_allclose(u.data["r_IJ"], u2.data["r_IJ"], err_msg="The distances don't match.")

    def test_roundtrip(self, u: InternalCoord, tmp_path: Path):
        """Test that the written file matches the initial file.

        GIVEN the internal coordinates from a file
        WHEN a new file is written
        THEN the two files should match

        Parameters
        ----------
        u : InternalCoord
            internal coordinates class
        tmp_path : Path
            temporary path
        """
        # Write out a copy of the internal coordinates, and compare this against the original.
        # This is more rigorous as it checks all formatting.
        filename: Path = tmp_path / "temp.ic"
        with mda.Writer(filename.as_posix()) as ofile:
            ofile.write(u)

        def IC_iter(fn: str):  # noqa: N802
            with open(fn) as inf:
                for line in inf:
                    if not line.startswith("*"):
                        yield line

        for ref, other in zip(IC_iter(IC), IC_iter(filename.as_posix()), strict=True):
            assert ref == other
