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
"""Test RTF writer."""

from pathlib import Path
from typing import TypeVar

import fluctmatch.parsers.writers.RTF
import MDAnalysis as mda
import pytest
from numpy import testing
from pytest_mock import MockerFixture

from ..datafile import COR, PSF, RTF

TTestRTFWriter = TypeVar("TTestRTFWriter", bound="TestRTFWriter")


class TestRTFWriter:
    """Test RTF writer."""

    @staticmethod
    @pytest.fixture()
    def universe() -> mda.Universe:
        """Create a universe.

        Returns
        -------
        Universe
            A universe
        """
        return mda.Universe(PSF, COR)

    @staticmethod
    @pytest.fixture()
    def filename(tmp_path: Path) -> Path:
        """Return a path to the RTF file.

        Parameters
        ----------
        tmp_path : Path
            A temporary path

        Returns
        -------
        Path
            RTF ffilename
        """
        return tmp_path / "temp.rtf"

    def test_writer(self: TTestRTFWriter, universe: mda.Universe, filename: Path, mocker: MockerFixture) -> None:
        """Test RTF writer.

        GIVEN an all-atom universe
        WHEN the writer is called
        THEN the patched writer should register as called

        Parameters
        ----------
        universe : Universe
            An all-atom universe
        filename : Path
            Topology file
        """
        mocker.patch.object(fluctmatch.parsers.writers.RTF.Writer, "write")
        with mda.Writer(filename, n_atoms=universe.atoms.n_atoms) as w:
            w.write(universe.atoms)

        fluctmatch.parsers.writers.RTF.Writer.write.assert_called()

    def test_roundtrip(self: TTestRTFWriter, universe: mda.Universe, filename: Path) -> None:
        """Compare a written RTF with the original file.

        GIVEN a RTF file
        WHEN an all-atom universe is written to a new topology file
        THEN the new topology file should be equivalent line-by-line to the original file

        Parameters
        ----------
        universe : Universe
            An all-atom universe
        filename : Path
            Topology file
        """
        # Write out a copy of the Universe, and compare this against the
        # original. This is more rigorous than simply checking the coordinates
        # as it checks all formatting.
        with mda.Writer(filename, n_atoms=universe.atoms.n_atoms) as w:
            w.write(universe.atoms)

        def rtf_iter(fn: str | Path):
            with open(fn) as inf:
                for line in inf:
                    if not line.startswith("*"):
                        yield line

        for ref, other in zip(rtf_iter(RTF), rtf_iter(filename), strict=True):
            testing.assert_string_equal(other, ref)
