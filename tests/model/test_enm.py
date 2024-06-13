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
"""Test elastic network model."""

from typing import Self

import MDAnalysis as mda
import numpy as np
import pytest
from fluctmatch.model import enm
from fluctmatch.model.base import coarse_grain
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysisTests.datafiles import DCD2, PSF
from numpy import testing
from testfixtures.mock import Mock
from testfixtures.replace import Replacer

# Number of residues to test
N_RESIDUES: int = 5
N_FRAMES: int = 5


@pytest.fixture(autouse=True, scope="class")
def universe() -> mda.Universe:
    """Fixture for a universe.

    Returns
    -------
    Universe
        universe with protein, DNA, and water
    """
    u = mda.Universe(PSF, DCD2)
    new = mda.Merge(u.residues[:N_RESIDUES].atoms)
    n_atoms = new.atoms.n_atoms
    pos = np.array([u.atoms.positions[:n_atoms] for _ in u.trajectory[:N_FRAMES]])
    return new.load_new(pos, format=MemoryReader, order="fac")


class TestElasticNetworkModel:
    """Test elastic network model."""

    @pytest.fixture(scope="class")
    def atoms(self: Self, universe: mda.Universe) -> mda.AtomGroup:
        """Fixture for a coarse-grain model.

        Returns
        -------
        MDAnalysis.AtomGroup
            Atom group of a coarse-grain model
        """
        return universe.select_atoms("all")

    @pytest.fixture()
    def model(self: Self, universe: mda.Universe) -> enm.ElasticModel:
        """Fixture for an elastic network model.

        Parameters
        ----------
        universe : mda.Universe
            Coarse-grain model

        Returns
        -------
        Elastic network model
        """
        return coarse_grain.get("ENM", universe)

    def test_topology_creation(self: Self, atoms: mda.AtomGroup, model: enm.ElasticModel) -> None:
        """Test topology for elastic network model.

        GIVEN an all-atom universe
        WHEN transformed into a coarse-grain model
        THEN number of atoms, residues, masses, and charges should be equal
        """
        model.create_topology()
        system: mda.Universe = model.universe

        testing.assert_equal(system.atoms.n_atoms, atoms.n_atoms, err_msg="Number of atoms not equal")
        testing.assert_equal(system.residues.n_residues, atoms.n_residues, err_msg="Number of residues not equal")
        testing.assert_allclose(system.residues.masses, atoms.residues.masses, err_msg="Masses not equal")
        testing.assert_allclose(system.residues.charges, 0.0, err_msg="Charges not equal")

    def test_bond_generation(self: Self, model: enm.ElasticModel) -> None:
        """Test the creation of intramolecular bonds between elastic network atoms.

        GIVEN an all-atom universe
        WHEN transformed into a coarse-grain model
        THEN bonds are formed between respective sites.
        """
        model.create_topology().generate_bonds(guess=False)
        system: mda.Universe = model.universe

        assert len(system.bonds) > 0, "Bonds not generated"
        assert len(system.angles) == 0, "Angles generated"
        assert len(system.dihedrals) == 0, "Dihedrals angles generated"
        assert len(system.impropers) == 0, "Improper dihedral angles generated"

    def test_trajectory_addition(self: Self, atoms: mda.AtomGroup, model: enm.ElasticModel) -> None:
        """Ensure that positions match between the all-atom and elastic network model.

        GIVEN an all-atom universe
        WHEN transformed into a coarse-grain model
        THEN trajectory is added to the universe with the same number of frames.
        """
        model.create_topology().add_trajectory(com=True)
        u = atoms.universe
        system: mda.Universe = model.universe

        atom_positions = [atoms.positions for _ in u.trajectory]
        model_positions = [system.atoms.positions for _ in system.trajectory]

        testing.assert_equal(system.trajectory.n_frames, u.trajectory.n_frames, err_msg="Number of frames not equal")
        testing.assert_allclose(model_positions, atom_positions, err_msg="Positions not equal")

    def test_transformation(self: Self, model: enm.ElasticModel) -> None:
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
            mock_traj.assert_not_called()
