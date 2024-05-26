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
"""Test various coarse-grain models developed for proteins."""

import itertools
from types import MappingProxyType
from typing import Self

import MDAnalysis as mda
import pytest
from fluctmatch.model import nucleic3
from fluctmatch.model.base import coarse_grain
from numpy import testing
from numpy.typing import NDArray

from tests.datafile import DNA


@pytest.fixture(autouse=True, scope="class")
def universe() -> mda.Universe:
    """Fixture for a universe.

    Returns
    -------
    Universe
        universe with DNA
    """
    return mda.Universe(DNA)


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
        return universe.select_atoms("all")

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
        return coarse_grain.get("NUCLEIC3", universe, guess_angles=True)

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

        model.create_topology()
        model.generate_bonds()
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
        model.create_topology()
        model.add_trajectory()

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

    def test_transformation(self: Self, atoms: mda.AtomGroup, model: nucleic3.NucleicModel) -> None:
        """Test transformation from an all-atom system to C-alpha model.

        GIVEN an all-atom universe
        WHEN transformed into a coarse-grain model
        THEN the system will have a proper topology and trajectory.
        """
        system = model.transform()

        # Test topology creation
        testing.assert_equal(system.atoms.n_atoms, atoms.n_residues * 3, err_msg="Number of atoms not equal")
        testing.assert_equal(system.residues.n_residues, atoms.n_residues, err_msg="Number of residues not equal")
        testing.assert_allclose(system.residues.masses, atoms.residues.masses, err_msg="Masses not equal", rtol=1e-01)

        # Test bond generation
        assert len(system.bonds) > 0, "Bonds not generated"
        assert len(system.angles) > 0, "Angles not generated"
        assert len(system.dihedrals) > 0, "Dihedral angles not generated"
        assert len(system.dihedrals) > 0, "Improper dihedral angles not generated"

        # Test trajectory addition
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


class TestNucleic4(TestNucleic3):
    """Test 4-bead nucleic acid model."""

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
        return coarse_grain.get("NUCLEIC4", universe, guess_angles=True)

    def test_topology_creation(self: Self, atoms: mda.AtomGroup, model: nucleic3.NucleicModel) -> None:
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

    def test_transformation(self: Self, atoms: mda.AtomGroup, model: nucleic3.NucleicModel) -> None:
        """Test transformation from an all-atom system to C-alpha model.

        GIVEN an all-atom universe
        WHEN transformed into a coarse-grain model
        THEN the system will have a proper topology and trajectory.
        """
        system = model.transform()

        # Test topology creation
        testing.assert_equal(system.atoms.n_atoms, atoms.n_residues * 4, err_msg="Number of atoms not equal")
        testing.assert_equal(system.residues.n_residues, atoms.n_residues, err_msg="Number of residues not equal")
        testing.assert_allclose(system.residues.masses, atoms.residues.masses, err_msg="Masses not equal", rtol=1e-01)

        # Test bond generation
        assert len(system.bonds) > 0, "Bonds not generated"
        assert len(system.angles) > 0, "Angles not generated"
        assert len(system.dihedrals) > 0, "Dihedral angles not generated"
        assert len(system.dihedrals) > 0, "Improper dihedral angles not generated"

        # Test trajectory addition
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
