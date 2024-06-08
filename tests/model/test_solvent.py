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
"""Tests for different solvent core."""

import itertools
from typing import Self

import MDAnalysis as mda
import numpy as np
import pytest
from fluctmatch.model import dma, tip3p, water
from fluctmatch.model.base import CoarseGrainModel, coarse_grain
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysisTests.datafiles import DCD_TRICLINIC, PSF_TRICLINIC
from numpy import testing
from testfixtures.mock import Mock
from testfixtures.replace import Replacer

from tests.datafile import DMA

# Number of residues to test
N_RESIDUES: int = 5


@pytest.fixture()
def universe() -> mda.Universe:
    """Fixture for a universe.

    Returns
    -------
    MDAnalysis.Universe
        Universe with protein, DNA, and water
    """
    u = mda.Universe(PSF_TRICLINIC, DCD_TRICLINIC)
    new = mda.Merge(u.residues[:N_RESIDUES].atoms)
    n_atoms = new.atoms.n_atoms
    pos = np.array([u.atoms.positions[:n_atoms] for _ in u.trajectory])
    return new.load_new(pos, format=MemoryReader, order="fac")


class TestWater:
    """Test conversion of all-atom water to O-only."""

    @pytest.fixture()
    def atoms(self: Self, universe: mda.Universe) -> mda.AtomGroup:
        """Fixture for a O-only water from the all-atom model.

        Returns
        -------
        MDAnalysis.AtomGroup
            Atom group with O atoms
        """
        return universe.select_atoms("water and name OW OH2")

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
        return coarse_grain.get("WATER", universe)

    def test_topology_creation(self: Self, atoms: mda.AtomGroup, model: water.WaterModel) -> None:
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

    def test_bond_generation(self: Self, model: water.WaterModel) -> None:
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
        self: Self, universe: mda.Universe, atoms: mda.AtomGroup, model: water.WaterModel
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

    def test_transformation(self: Self, model: water.WaterModel) -> None:
        """Ensure that the all-atom model is transformed into a C-alpha model.

        GIVEN an all-atom universe
        WHEN transformed into a coarse-grain model
        THEN trajectory is added to the universe with the same number of frames.

        Parameters
        ----------
        model : MDAnalysis.Universe
            Coarse-grain model
        """
        with Replacer() as replace:
            mock_bonds = replace("fluctmatch.model.base.CoarseGrainModel.generate_bonds", Mock())
            mock_traj = replace("fluctmatch.model.base.CoarseGrainModel.add_trajectory", Mock())
            model.transform(guess=True)

            mock_bonds.assert_called_once()
            mock_traj.assert_called_once()


class TestTip3p(TestWater):
    """Test conversion of all-atom water to a water with three bonds."""

    @pytest.fixture()
    def atoms(self: Self, universe: mda.Universe) -> mda.AtomGroup:
        """Fixture for a O-only water from the all-atom model.

        Returns
        -------
        MDAnalysis.AtomGroup
            Atom group with O atoms
        """
        return universe.select_atoms("all")

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
        return coarse_grain.get("TIP3P", universe, guess_angles=True)

    def test_topology_creation(self: Self, atoms: mda.AtomGroup, model: tip3p.Tip3pModel) -> None:
        """Test topology for C-alpha model.

        GIVEN an all-atom universe
        WHEN transformed into a coarse-grain model
        THEN number of atoms, residues, masses, and charges should be equal
        """
        model.create_topology()
        system: mda.Universe = model.universe

        testing.assert_equal(system.atoms.n_atoms, atoms.n_atoms, err_msg="Number of atoms not equal")
        testing.assert_equal(system.residues.n_residues, atoms.n_residues, err_msg="Number of residues not equal")
        testing.assert_allclose(system.residues.masses, atoms.residues.masses, err_msg="Masses not equal")

    def test_bond_generation(self: Self, model: tip3p.Tip3pModel) -> None:
        """Test that no bonds are created between water molecules.

        GIVEN an all-atom universe
        WHEN transformed into a coarse-grain model
        THEN no bonds are formed between respective sites.
        """
        with pytest.raises(AttributeError):
            model.generate_bonds(guess=True)

        model.create_topology().generate_bonds(guess=True)
        system: mda.Universe = model.universe

        testing.assert_equal(len(system.bonds), model.atoms.n_atoms, err_msg="Bonds not generated")
        testing.assert_equal(len(system.angles), model.atoms.n_atoms, err_msg="Angles not generated")
        testing.assert_equal(len(system.dihedrals), 0, err_msg="Dihedral angles generated")
        testing.assert_equal(len(system.impropers), 0, err_msg="Improper dihedral angles generated")

    def test_trajectory_addition(
        self: Self, universe: mda.Universe, atoms: mda.AtomGroup, model: tip3p.Tip3pModel
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


class TestDma:
    """Test the conversion of an all-atom dimethylamide to a 4-atom model."""

    @pytest.fixture()
    def universe(self: Self) -> mda.Universe:
        """Fixture for a universe.

        Returns
        -------
        Universe
            universe with dimethylamide (DMA)
        """
        u = mda.Universe(DMA)
        new = mda.Merge(u.residues[:N_RESIDUES].atoms)
        n_atoms = new.atoms.n_atoms
        pos = np.array([u.atoms.positions[:n_atoms] for _ in u.trajectory])
        return new.load_new(pos, format=MemoryReader, order="fac")

    @pytest.fixture()
    def atoms(self: Self, universe: mda.Universe) -> mda.AtomGroup:
        """Fixture for a O-only water from the all-atom model.

        Returns
        -------
        MDAnalysis.AtomGroup
            Atom group with O atoms
        """
        return universe.select_atoms("name C1 C2 C3 N")

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
        return coarse_grain.get("DMA", universe, guess_angles=True)

    def test_topology_creation(self: Self, atoms: mda.AtomGroup, model: dma.DmaModel) -> None:
        """Test topology for C-alpha model.

        GIVEN an all-atom universe
        WHEN transformed into a coarse-grain model
        THEN number of atoms, residues, masses, and charges should be equal
        """
        model.create_topology()
        system: mda.Universe = model.universe

        testing.assert_equal(system.atoms.n_atoms, atoms.n_atoms, err_msg="Number of atoms not equal")
        testing.assert_equal(system.residues.n_residues, atoms.n_residues, err_msg="Number of residues not equal")
        testing.assert_allclose(system.residues.masses, atoms.residues.masses, err_msg="Masses not equal")

    def test_bond_generation(self: Self, model: dma.DmaModel) -> None:
        """Test that no bonds are created between water molecules.

        GIVEN an all-atom universe
        WHEN transformed into a coarse-grain model
        THEN no bonds are formed between respective sites.
        """
        with pytest.raises(AttributeError):
            model.generate_bonds()

        model.create_topology().generate_bonds(guess=True)
        system: mda.Universe = model.universe

        testing.assert_equal(len(system.bonds), system.residues.n_residues * 3, err_msg="Bonds not generated")
        testing.assert_equal(len(system.angles), system.residues.n_residues * 3, err_msg="Angles not generated")
        testing.assert_equal(len(system.dihedrals), 0, err_msg="Dihedral angles generated")
        testing.assert_equal(
            len(system.impropers), system.residues.n_residues * 3, err_msg="Improper dihedral angles generated"
        )

    def test_trajectory_addition(self: Self, universe: mda.Universe, atoms: mda.AtomGroup, model: dma.DmaModel) -> None:
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

    def test_transformation(self: Self, model: dma.DmaModel) -> None:
        """Ensure that the all-atom model is transformed into a C-alpha model.

        GIVEN an all-atom universe
        WHEN transformed into a coarse-grain model
        THEN trajectory is added to the universe with the same number of frames.

        Parameters
        ----------
        model : MDAnalysis.Universe
            Coarse-grain model
        """
        with Replacer() as replace:
            mock_bonds = replace("fluctmatch.model.base.CoarseGrainModel.generate_bonds", Mock())
            mock_traj = replace("fluctmatch.model.base.CoarseGrainModel.add_trajectory", Mock())
            model.transform(guess=True)

            mock_bonds.assert_called_once()
            mock_traj.assert_called_once()
