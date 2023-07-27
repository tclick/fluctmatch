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
# flake8: noqa
"""Test PSF reader and writer."""

from pathlib import Path
from typing import TypeVar
from unittest.mock import patch

import pytest
from numpy.testing import assert_equal

import fluctmatch.parsers.parsers.PSFParser as PSFParser
import MDAnalysis as mda
from MDAnalysis.core.topologyobjects import TopologyObject
from MDAnalysisTests.topology.base import ParserBase

# from MDAnalysisTests.datafiles import CRD, PSF

from ..datafile import COR, PSF

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

    def test_bonds_total_counts(self: TTestPSFParser, top: TopologyObject):
        assert len(top.bonds.values) == 429

    def test_bonds_atom_counts(self: TTestPSFParser, filename: str) -> None:
        u = mda.Universe(PSF)
        assert len(u.atoms[[0]].bonds) == 2
        assert len(u.atoms[[42]].bonds) == 2

    def test_bonds_identity(self: TTestPSFParser, top: TopologyObject) -> None:
        vals = top.bonds.values
        for b in ((0, 1), (0, 2)):
            assert (b in vals) or (b[::-1] in vals)

    def test_angles_total_counts(self: TTestPSFParser, top: TopologyObject) -> None:
        assert len(top.angles.values) == 726

    def test_angles_atom_counts(self: TTestPSFParser, filename: str) -> None:
        u = mda.Universe(filename)
        assert len(u.atoms[[0]].angles), 4
        assert len(u.atoms[[42]].angles), 6

    def test_angles_identity(self: TTestPSFParser, top: TopologyObject) -> None:
        vals = top.angles.values
        for b in ((1, 0, 2), (0, 1, 2), (0, 2, 3)):
            assert (b in vals) or (b[::-1] in vals)

    def test_dihedrals_total_counts(self: TTestPSFParser, top: TopologyObject) -> None:
        assert len(top.dihedrals.values) == 907

    def test_dihedrals_atom_counts(self: TTestPSFParser, filename: str) -> None:
        u = mda.Universe(filename)
        assert len(u.atoms[[0]].dihedrals) == 4

    def test_dihedrals_identity(self: TTestPSFParser, top: TopologyObject) -> None:
        vals = top.dihedrals.values
        for b in ((0, 1, 2, 3), (0, 2, 3, 4), (0, 2, 3, 5), (1, 0, 2, 3)):
            assert (b in vals) or (b[::-1] in vals)


class TestPSFWriter:
    @staticmethod
    @pytest.fixture()
    def universe() -> mda.Universe:
        return mda.Universe(PSF, COR)

    def test_writer(self: TTestPSFParser, universe: mda.Universe, tmp_path: Path):
        filename = tmp_path / "temp.xplor.psf"
        with patch("fluctmatch.parsers.writers.PSF.Writer.write") as writer, mda.Writer(filename) as w:
            w.write(universe.atoms)
            writer.assert_called()

    def test_roundtrip(self: TTestPSFParser, universe: mda.Universe, tmp_path: Path):
        # Write out a copy of the Universe, and compare this against the
        # original. This is more rigorous than simply checking the coordinates
        # as it checks all formatting
        filename = (tmp_path / "temp.xplor.psf").as_posix()
        with mda.Writer(filename) as w:
            w.write(universe.atoms)

        def psf_iter(fn: str) -> str:
            with open(fn) as inf:
                for line in inf:
                    if not line.startswith("*"):
                        yield line

        for ref, other in zip(psf_iter(PSF), psf_iter(filename), strict=True):
            assert ref == other

    def test_write_atoms(self: TTestPSFParser, universe: mda.Universe, tmp_path: Path):
        # Test that written file when read gives same coordinates
        filename = tmp_path / "temp.xplor.psf"
        with mda.Writer(filename) as w:
            w.write(universe.atoms)

        u2 = mda.Universe(filename, COR)

        assert_equal(universe.atoms.charges, u2.atoms.charges)
