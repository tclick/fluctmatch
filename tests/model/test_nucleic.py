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
"""Test various coarse-grain models developed for proteins."""

import itertools
from types import MappingProxyType
from typing import Self

import MDAnalysis as mda
import numpy as np
import pytest
from fluctmatch.model import nucleic3, nucleic4
from fluctmatch.model.base import coarse_grain
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysisTests.datafiles import RNA_PDB, RNA_PSF
from numpy import testing
from numpy.typing import NDArray
from testfixtures.mock import Mock
from testfixtures.replace import Replacer

# Number of residues to test
N_RESIDUES: int = 5


@pytest.fixture(autouse=True, scope="class")
def universe() -> mda.Universe:
    """Fixture for a universe.

    Returns
    -------
    Universe
        universe with protein, DNA, and water
    """
    u = mda.Universe(RNA_PSF, RNA_PDB)
    new = mda.Merge(u.residues[:N_RESIDUES].atoms)
    n_atoms = new.atoms.n_atoms
    pos = np.array([u.atoms.positions[:n_atoms] for _ in u.trajectory])
    return new.load_new(pos, format=MemoryReader, order="fac")


class TestNucleic3:
    """Test 3-bead nucleic acid model."""

    @pytest.fixture(scope="class")
    def atoms(self: Self, universe: mda.Universe) -> mda.AtomGroup:
        """Fixture for a C-alpha model from the all-atom model.

        Returns
        -------
        MDAnalysis.AtomGroup
            Atom group with C-alpha atoms and bioions
        """
        return universe.select_atoms("nucleic")

    @pytest.fixture()
    def model(self: Self, universe: mda.Universe) -> nucleic3.NucleicModel:
        """Fixture for a C-alpha model.

        Parameters
        ----------
        universe : mda.Universe
            Universe with protein, DNA, and water

        Returns
        -------
        C-alpha only universe with coordinates
        """
        return coarse_grain.get("NUCLEIC3", universe)

    def test_topology_creation(self: Self, atoms: mda.AtomGroup, model: nucleic3.NucleicModel) -> None:
        """Test topology for C-alpha model.

        GIVEN an all-atom universe
        WHEN transformed into a coarse-grain model
        THEN number of atoms, residues, masses, and charges should be equal
        """
        model.create_topology()
        system: mda.Universe = model.universe

        testing.assert_equal(system.atoms.n_atoms, atoms.residues.n_residues * 3, err_msg="Number of atoms not equal")
        testing.assert_equal(
            system.residues.n_residues, atoms.residues.n_residues, err_msg="Number of residues not equal"
        )
        testing.assert_allclose(system.residues.masses, atoms.residues.masses, err_msg="Masses not equal", rtol=1e-01)

    def test_bond_generation(self: Self, model: nucleic3.NucleicModel) -> None:
        """Test the creation of intramolecular bonds between C-alpha atoms.

        GIVEN an all-atom universe
        WHEN transformed into a coarse-grain model
        THEN bonds are formed between respective sites.
        """
        with pytest.raises(AttributeError):
            model.generate_bonds()

        model.create_topology().generate_bonds(guess=True)
        system: mda.Universe = model.universe

        assert len(system.bonds) > 0, "Bonds not generated"
        assert len(system.angles) > 0, "Angles not generated"
        assert len(system.dihedrals) > 0, "Dihedral angles not generated"
        assert len(system.dihedrals) > 0, "Improper dihedral angles not generated"

    def test_trajectory_addition(self: Self, atoms: mda.AtomGroup, model: nucleic3.NucleicModel) -> None:
        """Ensure that positions match between the all-atom and C-alpha model.

        GIVEN an all-atom universe
        WHEN transformed into a coarse-grain model
        THEN trajectory is added to the universe with the same number of frames.
        """
        model.create_topology().add_trajectory()

        system: mda.Universe = model.universe

        positions: list[list[NDArray]] = []
        for residue, selection in itertools.product(atoms.residues, model._mapping.values()):
            value = (
                selection.get(residue.resname, "hsidechain and not name H*")
                if isinstance(selection, MappingProxyType)
                else selection
            )
            if residue.atoms.select_atoms(value):
                positions.append(residue.atoms.select_atoms(value).center_of_mass())
        testing.assert_allclose(positions, system.atoms.positions, err_msg="The coordinates do not match.")

    def test_transformation(self: Self, model: nucleic3.NucleicModel) -> None:
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


class TestNucleic4(TestNucleic3):
    """Test 4-bead nucleic acid model."""

    @pytest.fixture()
    def model(self: Self, universe: mda.Universe) -> nucleic4.NucleicModel:
        """Fixture for a C-alpha model.

        Parameters
        ----------
        universe : mda.Universe
            Universe with protein, DNA, and water

        Returns
        -------
        C-alpha only universe with coordinates
        """
        return coarse_grain.get("NUCLEIC4", universe)

    def test_topology_creation(self: Self, atoms: mda.AtomGroup, model: nucleic4.NucleicModel) -> None:
        """Test topology for C-alpha model.

        GIVEN an all-atom universe
        WHEN transformed into a coarse-grain model
        THEN number of atoms, residues, masses, and charges should be equal
        """
        model.create_topology()
        system: mda.Universe = model.universe

        testing.assert_equal(system.atoms.n_atoms, atoms.residues.n_residues * 4, err_msg="Number of atoms not equal")
        testing.assert_equal(
            system.residues.n_residues, atoms.residues.n_residues, err_msg="Number of residues not equal"
        )
        testing.assert_allclose(system.residues.masses, atoms.residues.masses, err_msg="Masses not equal", rtol=1e-01)
