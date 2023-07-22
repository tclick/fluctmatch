# ------------------------------------------------------------------------------
#  fluctmatch
#  Copyright (c) 2023 Timothy H. Click, Ph.D.
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
# pyright: reportInvalidTypeVarUse=false, reportOptionalSubscript=false, reportGeneralTypeIssues=false
# flake8: noqa
"""Tests for the elastic network model."""

import itertools

import MDAnalysis as mda
import numpy as np
import pytest
from fluctmatch.core.models import enm
from numpy import testing

from ..datafile import DCD, PSF
from .. import Self


class TestEnm:
    """Tests for the elastic network model."""

    @pytest.fixture(scope="class")
    def universe(self: Self) -> mda.Universe:
        universe = mda.Universe(PSF, DCD)
        return mda.Merge(universe.residues[:6].atoms)

    @pytest.fixture(scope="class")
    def model(self: Self, universe: mda.Universe) -> enm.Model:
        return enm.Model()

    @pytest.fixture(scope="class")
    def model_nocharge(self: Self, universe: mda.Universe) -> enm.Model:
        return enm.Model(charges=False)

    @pytest.fixture(scope="class")
    def system(self: Self, universe: mda.Universe, model: enm.Model) -> mda.Universe:
        return model.transform(universe)

    def test_creation(self: Self, universe: mda.Universe, system: mda.Universe) -> None:
        n_atoms: int = universe.atoms.n_atoms
        assert system.atoms.n_atoms == n_atoms, "The number of beads don't match."

    def test_names(self: Self, universe: mda.Universe, system: mda.Universe) -> None:
        testing.assert_string_equal(system.atoms[0].name, "A001")
        testing.assert_string_equal(system.residues[0].resname, "A001")

    def test_positions(self: Self, universe: mda.Universe, system: mda.Universe) -> None:
        testing.assert_allclose(system.atoms.positions, universe.atoms.positions, err_msg="Coordinates don't match.")

    def test_trajectory(self: Self, universe: mda.Universe, system: mda.Universe) -> None:
        assert (
            system.trajectory.n_frames == universe.trajectory.n_frames
        ), "All-atom and coarse-grain trajectories unequal."

    def test_masses(self: Self, universe: mda.Universe, system: mda.Universe, model: enm.Model) -> None:
        testing.assert_allclose(system.atoms.masses, universe.atoms.masses, err_msg="Masses don't match.")

    def test_charges(self: Self, universe: mda.Universe, system: mda.Universe, model: enm.Model) -> None:
        testing.assert_allclose(system.atoms.charges, universe.atoms.charges, err_msg="Charges don't match.")

    def test_zero_charges(self: Self, universe: mda.Universe, model_nocharge: enm.Model) -> None:
        system = model_nocharge.transform(universe)
        testing.assert_allclose(
            system.atoms.charges, np.zeros_like(universe.atoms.charges), err_msg="Charges don't match."
        )

    def test_bonds(self: Self, universe: mda.Universe, system: mda.Universe) -> None:
        assert len(system.bonds) > len(
            universe.bonds
        ), "# of ENM bonds should be greater than the # of original CG bonds."

    def test_angles(self: Self, system: mda.Universe) -> None:
        assert len(system.angles) == 0, "Number of angles should not be > 0."

    def test_dihedrals(self: Self, system: mda.Universe) -> None:
        assert len(system.dihedrals) == 0, "Number of dihedral angles should not be > 0."

    def test_impropers(self: Self, system: mda.Universe) -> None:
        assert len(system.impropers) == 0, "Number of improper angles should not not be > 0."
