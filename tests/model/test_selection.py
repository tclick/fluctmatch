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
"""Test additional MDAnalysis selection options."""

from typing import Self

import MDAnalysis as mda
import pytest
from fluctmatch.model.selection import *

from tests.datafile import GRO


class TestProteinSelections:
    """Test keyword arguments for selecting atoms within a protein."""

    @staticmethod
    @pytest.fixture()
    def universe() -> mda.Universe:
        """Fixture for a universe.

        Returns
        -------
        Universe
            universe with protein, DNA, and water
        """
        return mda.Universe(GRO)

    def test_backbone(self: Self, universe: mda.Universe) -> None:
        """Test selection of backbone."""
        selection: mda.AtomGroup = universe.select_atoms("backbone")
        assert selection.n_atoms > 0, "Number of atoms don't match."

    def test_hbackbone(self: Self, universe: mda.Universe) -> None:
        """Test selection of backbone with hydrogen."""
        selection: mda.AtomGroup = universe.select_atoms("hbackbone")
        assert selection.n_atoms > 0, "Number of atoms don't match."

    def test_calpha(self: Self, universe: mda.Universe) -> None:
        """Test selection for C-alpha atoms."""
        selection: mda.AtomGroup = universe.select_atoms("calpha")
        assert selection.n_atoms > 0, "Number of atoms don't match."

    def test_hcalpha(self: Self, universe: mda.Universe) -> None:
        """Test selection for C-alpha atoms with hydrogen."""
        selection: mda.AtomGroup = universe.select_atoms("hcalpha")
        assert selection.n_atoms > 0, "Number of atoms don't match."

    def test_cbeta(self: Self, universe: mda.Universe) -> None:
        """Test selection for C-beta atoms."""
        selection: mda.AtomGroup = universe.select_atoms("cbeta")
        assert selection.n_atoms > 0, "Number of atoms don't match."

    def test_amine(self: Self, universe: mda.Universe) -> None:
        """Test selection for amine group."""
        selection: mda.AtomGroup = universe.select_atoms("amine")
        assert selection.n_atoms > 0, "Number of atoms don't match."

    def test_carboxyl(self: Self, universe: mda.Universe) -> None:
        """Test selection for carboxyl group."""
        selection: mda.AtomGroup = universe.select_atoms("carboxyl")
        assert selection.n_atoms > 0, "Number of atoms don't match."

    def test_hsidechain(self: Self, universe: mda.Universe) -> None:
        """Test selection for sidechain group with hydrogen."""
        selection: mda.AtomGroup = universe.select_atoms("hsidechain")
        assert selection.n_atoms > 0, "Number of atoms don't match."


class TestSolvent:
    """Test selection for solvent groups."""

    @staticmethod
    @pytest.fixture()
    def universe() -> mda.Universe:
        """Fixture for a universe.

        Returns
        -------
        Universe
            universe with protein, DNA, and water
        """
        return mda.Universe(GRO)

    def test_bioions(self: Self, universe: mda.Universe) -> None:
        """Test selection for various biological ions."""
        selection: mda.AtomGroup = universe.select_atoms("bioion")
        assert selection.n_atoms > 0, "Number of atoms don't match."

    def test_water(self: Self, universe: mda.Universe) -> None:
        """Test selection for water."""
        selection: mda.AtomGroup = universe.select_atoms("water")
        assert selection.n_atoms > 0, "Number of atoms don't match."


class TestNucleic:
    """Test selection for nucleic acids."""

    @staticmethod
    @pytest.fixture()
    def universe() -> mda.Universe:
        """Fixture for a universe.

        Returns
        -------
        Universe
            universe with protein, DNA, and water
        """
        return mda.Universe(GRO)

    def test_nucleic(self: Self, universe: mda.Universe) -> None:
        """Test selection for nucleic acids."""
        selection: mda.AtomGroup = universe.select_atoms("nucleic")
        assert selection.n_atoms > 0, "Number of atoms don't match."

    def test_hsugar(self: Self, universe: mda.Universe) -> None:
        """Test selection for sugar group of nucleic acids with hydrogen."""
        selection: mda.AtomGroup = universe.select_atoms("hnucleicsugar")
        assert selection.n_atoms > 0, "Number of atoms don't match."

    def test_hbase(self: Self, universe: mda.Universe) -> None:
        """Test selection for nucleic base group of nucleic acids with hydrogen."""
        selection: mda.AtomGroup = universe.select_atoms("hnucleicbase")
        assert selection.n_atoms > 0, "Number of atoms don't match."

    def test_hphosphate(self: Self, universe: mda.Universe) -> None:
        """Test selection for phosphate group including the C5' of nucleic acids."""
        selection: mda.AtomGroup = universe.select_atoms("nucleicphosphate")
        assert selection.n_atoms > 0, "Number of atoms don't match."

    def test_sugarc2(self: Self, universe: mda.Universe) -> None:
        """Test selection for sugar atom on C3' of nucleic acids."""
        selection: mda.AtomGroup = universe.select_atoms("sugarC2")
        assert selection.n_atoms > 0, "Number of atoms don't match."

    def test_sugarc4(self: Self, universe: mda.Universe) -> None:
        """Test selection for sugar atom on C4' of nucleic acids."""
        selection: mda.AtomGroup = universe.select_atoms("sugarC4")
        assert selection.n_atoms > 0, "Number of atoms don't match."

    def test_center(self: Self, universe: mda.Universe) -> None:
        """Test selection for the central atoms (C4' and C5') on the base of the nuleic acid."""
        selection: mda.AtomGroup = universe.select_atoms("nucleiccenter")
        assert selection.n_atoms > 0, "Number of atoms don't match."
