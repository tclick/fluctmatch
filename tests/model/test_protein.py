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

from typing import Self

import MDAnalysis as mda
import numpy as np
import pytest
from fluctmatch.model import calpha, caside, ncsc, polar
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


class TestCalpha:
    """Test C-alpha model."""

    @pytest.fixture(scope="class")
    def atoms(self: Self, universe: mda.Universe) -> mda.AtomGroup:
        """Fixture for a C-alpha model from the all-atom model.

        Returns
        -------
        MDAnalysis.AtomGroup
            Atom group with C-alpha atoms and bioions
        """
        return universe.select_atoms("protein and name CA")

    @pytest.fixture()
    def model(self: Self, universe: mda.Universe) -> calpha.CalphaModel:
        """Fixture for a C-alpha model.

        Parameters
        ----------
        universe : mda.Universe
            Universe with protein, DNA, and water

        Returns
        -------
        C-alpha only universe with coordinates
        """
        return coarse_grain.get("CALPHA", universe)

    def test_topology_creation(self: Self, atoms: mda.AtomGroup, model: calpha.CalphaModel) -> None:
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
        testing.assert_allclose(system.residues.charges, atoms.residues.charges, err_msg="Charges not equal")

    def test_bond_generation(self: Self, model: calpha.CalphaModel) -> None:
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
        assert len(system.impropers) == 0, "Improper dihedral angles generated"

    def test_trajectory_addition(self: Self, atoms: mda.AtomGroup, model: calpha.CalphaModel) -> None:
        """Ensure that positions match between the all-atom and C-alpha model.

        GIVEN an all-atom universe
        WHEN transformed into a coarse-grain model
        THEN trajectory is added to the universe with the same number of frames.
        """
        model.create_topology().add_trajectory(com=True)
        u = atoms.universe
        system: mda.Universe = model.universe

        testing.assert_equal(system.trajectory.n_frames, u.trajectory.n_frames, err_msg="Number of frames not equal")

    def test_transformation(self: Self, model: calpha.CalphaModel) -> None:
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


class TestCaside(TestCalpha):
    """Test C-alpha/sidechain model."""

    @pytest.fixture(scope="class")
    def atoms(self: Self, universe: mda.Universe) -> mda.AtomGroup:
        """Fixture for a C-alpha model from the all-atom model.

        Returns
        -------
        MDAnalysis.AtomGroup
            Atom group with C-alpha atoms and bioions
        """
        return universe.select_atoms("protein and name CA CB")

    @pytest.fixture()
    def model(self: Self, universe: mda.Universe) -> caside.CasideModel:
        """Fixture for a C-alpha model.

        Parameters
        ----------
        universe : mda.Universe
            Universe with protein, DNA, and water

        Returns
        -------
        C-alpha only universe with coordinates
        """
        return coarse_grain.get("CASIDE", universe)

    def test_bond_generation(self: Self, model: calpha.CalphaModel) -> None:
        """Test the creation of intramolecular bonds between C-alpha atoms.

        GIVEN an all-atom universe
        WHEN transformed into a coarse-grain model
        THEN bonds are formed between respective sites.
        """
        testing.assert_raises(AttributeError, model.generate_bonds)

        model.create_topology().generate_bonds(guess=True)
        system: mda.Universe = model.universe

        assert len(system.bonds) > 0, "Bonds not generated"
        assert len(system.angles) > 0, "Angles not generated"
        assert len(system.dihedrals) > 0, "Dihedral angles not generated"
        assert len(system.impropers) > 0, "Improper dihedral angles not generated"


class TestNcsc(TestCaside):
    """Test C-alpha/sidechain model."""

    @pytest.fixture(scope="class")
    def atoms(self: Self, universe: mda.Universe) -> mda.AtomGroup:
        """Fixture for a C-alpha model from the all-atom model.

        Returns
        -------
        MDAnalysis.AtomGroup
            Atom group with C-alpha atoms and bioions
        """
        return universe.select_atoms("protein and name N CB O OT1")

    @pytest.fixture()
    def model(self: Self, universe: mda.Universe) -> ncsc.NcscModel:
        """Fixture for a C-alpha model.

        Parameters
        ----------
        universe : mda.Universe
            Universe with protein, DNA, and water

        Returns
        -------
        C-alpha only universe with coordinates
        """
        return coarse_grain.get("NCSC", universe)


class TestPolar(TestNcsc):
    """Test amino N, carboxyl O, and polar sidechain model."""

    @pytest.fixture()
    def model(self: Self, universe: mda.Universe) -> polar.PolarModel:
        """Fixture for a C-alpha model.

        Parameters
        ----------
        universe : mda.Universe
            Universe with protein, DNA, and water

        Returns
        -------
        C-alpha only universe with coordinates
        """
        return coarse_grain.get("POLAR", universe)
