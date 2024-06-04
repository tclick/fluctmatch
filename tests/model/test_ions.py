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
# pyright: reportOptionalSubscript=false, reportGeneralTypeIssues=false, reportInvalidTypeVarUse=false
# pyright: reportWildcardImportFromLibrary=false, reportIncompatibleMethodOverride=false
"""Tests for various ion combinations."""

import itertools
from typing import Self

import MDAnalysis as mda
import pytest
from fluctmatch.model import solventions
from fluctmatch.model.base import CoarseGrainModel, coarse_grain
from numpy import testing

from tests.datafile import IONS


class TestSolventIons:
    """Test conversion of all-atom water to O-only."""

    @pytest.fixture()
    def universe(self: Self) -> mda.Universe:
        """Fixture for a universe.

        Returns
        -------
        Universe
            universe with protein, DNA, and water
        """
        return mda.Universe(IONS)

    @pytest.fixture()
    def atoms(self: Self, universe: mda.Universe) -> mda.AtomGroup:
        """Fixture for a O-only water from the all-atom model.

        Returns
        -------
        MDAnalysis.AtomGroup
            Atom group with O atoms
        """
        return universe.select_atoms("name K CL")

    @pytest.fixture()
    def model(self: Self, universe: mda.Universe) -> CoarseGrainModel:
        """Fixture for a water model.

        Parameters
        ----------
        universe : mda.Universe
            Universe with water

        Returns
        -------
        water only universe with coordinates
        """
        return coarse_grain.get("SOLVENTIONS", universe, guess_angles=True)

    def test_topology_creation(self: Self, atoms: mda.AtomGroup, model: solventions.SolventIonModel) -> None:
        """Test topology for C-alpha model.

        GIVEN an all-atom universe
        WHEN transformed into a coarse-grain model
        THEN number of atoms, residues, masses, and charges should be equal
        """
        model.create_topology()
        system: mda.Universe = model.universe

        try:
            charges = [
                residue.atoms.select_atoms(selection).total_charge()
                for residue, selection in itertools.product(atoms.residues, model._selection.values())
                if residue.atoms.select_atoms(selection)
            ]
        except mda.NoDataError:
            charges = [0.0] * system.atoms.n_atoms

        testing.assert_equal(system.atoms.n_atoms, atoms.n_atoms, err_msg="Number of atoms not equal")
        testing.assert_equal(system.residues.n_residues, atoms.n_residues, err_msg="Number of residues not equal")
        testing.assert_allclose(system.residues.masses, atoms.residues.masses, err_msg="Masses not equal")
        testing.assert_allclose(system.residues.charges, charges, err_msg="Charges not equal")

    def test_bond_generation(self: Self, model: solventions.SolventIonModel) -> None:
        """Test that no bonds are created between water molecules.

        GIVEN an all-atom universe
        WHEN transformed into a coarse-grain model
        THEN no bonds are formed between respective sites.
        """
        with pytest.raises(AttributeError):
            model.generate_bonds()

        model.create_topology().generate_bonds()
        system: mda.Universe = model.universe

        testing.assert_equal(len(system.bonds), 0, err_msg="Bonds generated")
        with pytest.raises(mda.NoDataError):
            _ = system.angles
        with pytest.raises(mda.NoDataError):
            _ = system.dihedrals
        with pytest.raises(mda.NoDataError):
            _ = system.impropers

    def test_trajectory_addition(
        self: Self, universe: mda.Universe, atoms: mda.AtomGroup, model: solventions.SolventIonModel
    ) -> None:
        """Ensure that positions match between the all-atom and C-alpha model.

        GIVEN an all-atom universe
        WHEN transformed into a coarse-grain model
        THEN trajectory is added to the universe with the same number of frames.
        """
        model.create_topology().add_trajectory()
        system: mda.Universe = model.universe

        atom_positions = [atoms.positions for _ in universe.trajectory]
        model_positions = [system.atoms.positions for _ in system.trajectory]

        testing.assert_equal(
            system.trajectory.n_frames, universe.trajectory.n_frames, err_msg="Number of frames not equal"
        )
        testing.assert_allclose(model_positions, atom_positions, err_msg="Positions not equal")
