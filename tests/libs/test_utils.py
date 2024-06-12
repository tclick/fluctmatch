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
"""Tests for various utility functions."""

from typing import Self

import MDAnalysis as mda
import pytest
from fluctmatch.libs.utils import compare_dict_keys, merge, rename_universe
from MDAnalysis.coordinates.base import ReaderBase
from MDAnalysisTests.datafiles import DCD2, PSF
from numpy import testing
from testfixtures import ShouldRaise

from tests.datafile import DCD_CG, DMA, PSF_CG


@pytest.fixture()
def aa_universe() -> mda.Universe:
    """Create an all-atom universe.

    Returns
    -------
    MDAnalysis.Universe
        An all-atom universe
    """
    return mda.Universe(PSF, DCD2)


@pytest.fixture()
def cg_universe() -> mda.Universe:
    """Create a coarse-grain universe.

    Returns
    -------
    MDAnalysis.Universe
        A coarse-grain universe
    """
    return mda.Universe(PSF_CG, DCD_CG)


class TestUtils:
    """Test utility functions."""

    def test_merge(self: Self, aa_universe: mda.Universe) -> None:
        """Test the merging of two universes.

        GIVEN a universe
        WHEN merging it with itself
        THEN check the universe is doubled in size with the same trajectory length

        Parameters
        ----------
        aa_universe : MDAnalysis.Universe
            An all-atom universe
        """
        atoms: mda.AtomGroup = aa_universe.atoms
        trajectory: ReaderBase = aa_universe.trajectory
        n_atoms: int = atoms.n_atoms
        n_frames: int = trajectory.n_frames

        merged: mda.Universe = merge(aa_universe, aa_universe)
        assert hasattr(merged, "atoms")
        assert hasattr(merged, "trajectory")
        assert merged.atoms.n_atoms == n_atoms * 2
        assert merged.trajectory.n_frames == n_frames

    def test_merge_fail(self: Self, aa_universe: mda.Universe) -> None:
        """Test the merging of two universes fails.

        GIVEN two universes with unequal length trajectories
        WHEN merging them together
        THEN a ValueError should be raised

        Parameters
        ----------
        aa_universe : MDAnalysis.Universe
            An all-atom universe
        """
        dma = mda.Universe(DMA)
        with ShouldRaise(ValueError):
            merge(aa_universe, dma)

    def test_rename_universe(self: Self, cg_universe: mda.Universe) -> None:
        """Test function for renaming atoms and residues of a universe.

        GIVEN a universe
        WHEN renaming atoms and residues of a universe
        THEN the original atom and residue names are changed.

        Parameters
        ----------
        cg_universe : MDAnalysis.Universe
            A coarse-grain universe
        """
        rename_universe(cg_universe)

        testing.assert_string_equal(cg_universe.atoms[0].name, "A00001")
        testing.assert_string_equal(cg_universe.atoms[-1].name, "A00622")
        testing.assert_string_equal(cg_universe.residues[0].resname, "A00001")
        testing.assert_string_equal(cg_universe.residues[-1].resname, "A00214")
        assert "4AKE" in cg_universe.segments.segids

    def test_key_comparison(self: Self, aa_universe: mda.Universe) -> None:
        """Test the key comparison between two dictionaries.

        GIVEN a universe
        WHEN comparing the bond dictionary to itself
        THEN the two dictionaries are equal.

        Parameters
        ----------
        aa_universe : MDAnalysis.Universe
            An all-atom universe
        """
        bond_dict = aa_universe.bonds.topDict
        assert compare_dict_keys(bond_dict, bond_dict) is None

    def test_compare_different_dict_keys(self: Self, aa_universe: mda.Universe, cg_universe: mda.Universe) -> None:
        """Test the key comparison between two dictionaries that are not equivalent.

        GIVEN two different universes
        WHEN comparing the bond dictionaries
        THEN an exception is raised.

        Parameters
        ----------
        aa_universe : MDAnalysis.Universe
            An all-atom universe
        cg_universe : MDAnalysis.Universe
            A coarse-grain universe
        """
        aa_bonds = aa_universe.bonds.topDict
        cg_bonds = cg_universe.bonds.topDict
        with ShouldRaise(ValueError):
            compare_dict_keys(aa_bonds, cg_bonds)
