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
