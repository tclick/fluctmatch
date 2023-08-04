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
# pyright: reportOptionalIterable=false, reportInvalidTypeVarUse=false, reportGeneralTypeIssues=false
# flake8: noqa
"""Tests for readers and writers for CHARMM parameters."""

from pathlib import Path
from typing import TypeVar
from unittest.mock import patch

import MDAnalysis as mda
import pytest
from fluctmatch.libs.parameters import Parameters
from fluctmatch.parsers.readers import PRM as ParamReader
from numpy.testing import assert_allclose

from tests.datafile import PRM

TTestPRMReader = TypeVar("TTestPRMReader", bound="TestPRMReader")
TTestPRMWriter = TypeVar("TTestPRMWriter", bound="TestPRMWriter")


class TestPRMReader:
    """Test reader for CHARMM parameters."""

    def test_reader(self: TTestPRMReader) -> None:
        """Test CHARMM parameter reader.

        GIVEN a parameter file
        WHEN the file is read
        THEN a parameter object is created
        """
        with ParamReader.Reader(PRM) as infile:
            parameters = infile.read()

        assert isinstance(parameters, Parameters)
        assert parameters.atoms.get("mass") is not None
        assert parameters.bonds.get("Kb") is not None
        assert parameters.angles is None
        assert parameters.dihedrals is None
        assert parameters.improper is None


class TestPRMWriter:
    """Test writer for CHARMM parameters."""

    @staticmethod
    @pytest.fixture()
    def parameters() -> Parameters:
        """Create a parameters object.

        Returns
        -------

        Parameters
        ----------
            Parameter container
        """
        return ParamReader.Reader(PRM).read()

    @staticmethod
    @pytest.fixture()
    def filename(tmp_path: Path) -> Path:
        """Return filename for CHARMM parameter file.

        Parameters
        ----------
        tmp_path : Path
            Temporary path location

        Returns
        -------
        Path
            CHARMM parameter filename
        """
        return tmp_path / "temp.prm"

    def test_writer(self: TTestPRMWriter, parameters: Parameters, filename: Path) -> None:
        """Test writer functionality.

        GIVEN a parameter set
        WHEN the `write` function is patched
        THEN the patch should be called when using MDAnalysis.Writer().write

        Parameters
        ----------
        parameters : Parameters
            CHARMM parameters
        filename : Path
            CHARMM parameter file
        """
        with patch("fluctmatch.parsers.writers.PRM.Writer.write") as writer, mda.Writer(
            filename.as_posix(), nonbonded=True
        ) as ofile:
            ofile.write(parameters)
            writer.assert_called()

    def test_parameters(self: TTestPRMWriter, parameters: Parameters, filename: Path) -> None:
        """Test full functionality of writing a CHARMM parameter file.

        GIVEN a set of all-atom parameters
        WHEN writing to a file
        THEN the data should be written to the specified filename

        Parameters
        ----------
        parameters : Parameters
            CHARMM parameters
        filename : Path
            CHARMM parameter file
        """
        with mda.Writer(filename.as_posix(), nonbonded=True) as ofile:
            ofile.write(parameters)

        new_parameters: Parameters = ParamReader.Reader(filename).read()
        assert filename.exists()
        assert_allclose(
            parameters.atoms.get("mass"),
            new_parameters.atoms.get("mass"),
            err_msg="The atomic masses don't match.",
        )
        assert_allclose(
            parameters.bonds.get("Kb"),
            new_parameters.bonds.get("Kb"),
            err_msg="The force constants don't match.",
        )

    def test_roundtrip(self: TTestPRMWriter, parameters: Parameters, filename: Path) -> None:
        """Test that the source and destination files match.

        GIVEN a source parameter set
        WHEN a new file is written
        THEN the two file should be identical

        Parameters
        ----------
        parameters : Parameters
            CHARMM parameters
        filename : Path
            CHARMM parameter file
        """
        # Write out a copy of the internal coordinates, and compare this against
        # the original. This is more rigorous as it checks all formatting.
        with mda.Writer(filename.as_posix(), nonbonded=True) as ofile:
            ofile.write(parameters)

        def prm_iter(fn: str | Path):
            with open(fn) as input_file:
                for line in input_file:
                    if not line.startswith("*"):
                        yield line

        for ref, other in zip(prm_iter(PRM), prm_iter(filename), strict=True):
            assert ref == other
