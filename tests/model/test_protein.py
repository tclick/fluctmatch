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
from fluctmatch.model import calpha  # , caside, ncsc, polar
from numpy import testing
from numpy.typing import NDArray

from tests.datafile import TPR, XTC

# Number of residues to test
N_RESIDUES = 6


@pytest.fixture(autouse=True, scope="class")
def universe() -> mda.Universe:
    """Fixture for a universe.

    Returns
    -------
    Universe
        universe with protein, DNA, and water
    """
    u = mda.Universe(TPR, XTC)
    return mda.Merge(u.residues[:N_RESIDUES].atoms)


@pytest.fixture(autouse=True, scope="class")
def model(universe: mda.Universe) -> calpha.CalphaModel:
    """Fixture for a C-alpha model."""
    return calpha.CalphaModel(universe, guess_angles=True)


@pytest.fixture(autouse=True, scope="class")
def system(universe: mda.Universe) -> mda.Universe:
    """Fixture to transform an all-atom universe to a C-alpha model."""
    model = calpha.CalphaModel(universe, guess_angles=True)
    return model.transform()


class TestCalpha:
    """Test C-alpha model."""

    def test_topology(self: Self, universe: mda.Universe, system: mda.Universe) -> None:
        """Test topology for C-alpha model."""
        calpha = universe.select_atoms("protein and name CA")

        testing.assert_equal(system.atoms.n_atoms, calpha.n_atoms)
        testing.assert_allclose(system.residues.masses, calpha.residues.masses)
        testing.assert_allclose(system.residues.charges, calpha.residues.charges)

    def test_creation(self: Self, universe: mda.Universe, model: calpha.CalphaModel, system: mda.Universe) -> None:
        """Test creation of a C-alpha model."""
        n_atoms = 0
        for residue, selection in itertools.product(universe.residues, model._mapping.values()):
            value = selection.get(residue.resname) if isinstance(selection, MappingProxyType) else selection
            n_atoms += residue.atoms.select_atoms(value).residues.n_residues
        testing.assert_equal(system.atoms.n_atoms, n_atoms, err_msg="Number of sites don't match.")

    def test_positions(self: Self, universe: mda.Universe, model: calpha.CalphaModel, system: mda.Universe) -> None:
        """Ensure that positions match between the all-atom and C-alpha model."""
        positions: list[list[NDArray]] = []
        for residue, selection in itertools.product(universe.residues, model._mapping.values()):
            value = (
                selection.get(residue.resname, "hsidechain and not name H*")
                if isinstance(selection, MappingProxyType)
                else selection
            )
            if residue.atoms.select_atoms(value):
                positions.append(residue.atoms.select_atoms(value).center_of_mass())
        testing.assert_allclose(positions, system.atoms.positions, err_msg="The coordinates do not match.")

    def test_trajectory(self: Self, universe: mda.Universe, system: mda.Universe) -> None:
        """Compare the trajectory of the C-alpha model with the C-alpha atoms of the all-atom universe."""
        assert (
            system.trajectory.n_frames == universe.trajectory.n_frames
        ), "All-atom and coarse-grain trajectories unequal."

    def test_bonds(self: Self, system: mda.Universe) -> None:
        """Ensure that the number of bonds exists."""
        assert len(system.bonds) > 0, "Number of bonds should be > 0."

    def test_angles(self: Self, system: mda.Universe) -> None:
        """Ensure that the number of angles exists."""
        assert len(system.angles) > 0, "Number of angles should be > 0."

    def test_dihedrals(self: Self, system: mda.Universe) -> None:
        """Ensure that the number of dihedral angles exists."""
        assert len(system.dihedrals) > 0, "Number of dihedral angles should be > 0."

    def test_impropers(self: Self, system: mda.Universe) -> None:
        """Ensure that the number of improper dihedral angles exists."""
        assert len(system.impropers) == 0, "Number of improper angles should not be > 0."
