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
# pyright: reportInvalidTypeVarUse=false, reportGeneralTypeIssues=false, reportOptionalSubscript=false
# pyright: reportFunctionMemberAccess=false
# flake8: noqa
"""Test PSF reader and writer."""

from pathlib import Path
from typing import TypeVar
from pytest_mock import MockerFixture

import pytest
from numpy.testing import assert_equal

import fluctmatch
import fluctmatch.parsers.parsers.PSFParser as PSFParser
import MDAnalysis as mda
from MDAnalysisTests.topology.base import ParserBase

from ..datafile import COR, PSF
import fluctmatch.parsers.writers.PSF

TTestPSFParser = TypeVar("TTestPSFParser", bound="TestPSFParser")
TTestPSFParser = TypeVar("TTestPSFParser", bound="TestPSFParser")


class TestPSFParser(ParserBase):
    """Based on small PDB with AdK (:data:`PDB_small`)."""

    parser = PSFParser.Reader
    ref_filename = PSF
    expected_attrs = "ids names masses charges resids resnames segids bonds angles dihedrals impropers".split()
    expected_n_atoms = 330
    expected_n_residues = 115
    expected_n_segments = 1

    def test_bonds_atom_counts(self: TTestPSFParser) -> None:
        """Test bond count.

        GIVEN a PSF file
        WHEN a universe is loaded
        THEN the number of bonds should be equivalent
        """
        u = mda.Universe(PSF)
        assert len(u.atoms[[0]].bonds) == 2
        assert len(u.atoms[[42]].bonds) == 2

    def test_angles_atom_counts(self: TTestPSFParser, filename: str) -> None:
        u = mda.Universe(filename)
        assert len(u.atoms[[0]].angles), 4
        assert len(u.atoms[[42]].angles), 6


class TestPSFWriter:
    @staticmethod
    @pytest.fixture()
    def universe() -> mda.Universe:
        """Create a universe.

        Returns
        -------
        Universe
            An all-atom universe
        """
        return mda.Universe(PSF, COR)

    @staticmethod
    @pytest.fixture()
    def filename(tmp_path) -> Path:
        """Return an topology filename.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory

        Returns
        -------
        Path
            Topology filename
        """
        return tmp_path / "temp.xplor.psf"

    def test_writer(self: TTestPSFParser, universe: mda.Universe, filename: Path, mocker: MockerFixture) -> None:
        """Test PSF writer.

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
        mocker.patch.object(fluctmatch.parsers.writers.PSF.Writer, "write")
        with mda.Writer(filename) as w:
            w.write(universe.atoms)

        fluctmatch.parsers.writers.PSF.Writer.write.assert_called()

    def test_roundtrip(self: TTestPSFParser, universe: mda.Universe, filename: Path) -> None:
        """Compare a written PSF with the original file.

        GIVEN a PSF file
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
        # as it checks all formatting
        with mda.Writer(filename.as_posix()) as w:
            w.write(universe.atoms)

        def psf_iter(fn: str) -> str:
            with open(fn) as inf:
                for line in inf:
                    if not line.startswith("*"):
                        yield line

        for ref, other in zip(psf_iter(PSF), psf_iter(filename), strict=True):
            assert ref == other

    def test_write_atoms(self: TTestPSFParser, universe: mda.Universe, filename: Path) -> None:
        """Compare a written PSF with the original file.

        GIVEN an all-atom universe
        WHEN a topology file is written
        THEN the two topologies should match

        Parameters
        ----------
        universe : Universe
            An all-atom universe
        filename : Path
            Topology file
        """
        # Test that written file when read gives same coordinates
        with mda.Writer(filename.as_posix()) as w:
            w.write(universe.atoms)

        u2 = mda.Universe(filename.as_posix(), COR)

        assert_equal(universe.atoms.charges, u2.atoms.charges)
