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
# pyright: reportOptionalIterable=false, reportGeneralTypeIssues=false, reportInvalidTypeVarUse=false
# pyright: reportOptionalSubscript=false
# flake8: noqa
"""Tests for various protein core."""

import itertools
from types import MappingProxyType

import MDAnalysis as mda
import numpy as np
import pytest
from fluctmatch.core.models import calpha, caside, ncsc, polar
from numpy import testing
from numpy.typing import NDArray

from .. import Self
from ..datafile import TPR, XTC

# Number of residues to test
N_RESIDUES = 6


class TestCalpha:
    @staticmethod
    @pytest.fixture()
    def universe() -> mda.Universe:
        u = mda.Universe(TPR, XTC)
        return mda.Merge(u.residues[:N_RESIDUES].atoms)

    @staticmethod
    @pytest.fixture()
    def model() -> calpha.Model:
        return calpha.Model(guess_angles=True)

    @staticmethod
    @pytest.fixture()
    def system(universe: mda.Universe, model: calpha.Model) -> mda.Universe:
        return model.transform(universe)

    def test_creation(
        self: Self,
        universe: mda.Universe,
        model: calpha.Model,
        system: mda.Universe,
    ) -> None:
        n_atoms = 0
        for residue, selection in itertools.product(universe.residues, model._mapping.values()):
            value = selection.get(residue.resname) if isinstance(selection, MappingProxyType) else selection
            n_atoms += residue.atoms.select_atoms(value).residues.n_residues
        testing.assert_equal(system.atoms.n_atoms, n_atoms, err_msg="Number of sites don't match.")

    def test_positions(self: Self, universe: mda.Universe, system: mda.Universe, model: calpha.Model) -> None:
        positions: list[list[NDArray]] = []
        for residue, selection in itertools.product(universe.residues, model._mapping.values()):
            value = (
                selection.get(residue.resname, "hsidechain and not name H*")
                if isinstance(selection, MappingProxyType)
                else selection
            )
            if residue.atoms.select_atoms(value):
                positions.append(residue.atoms.select_atoms(value).center_of_mass())
        testing.assert_allclose(
            positions,
            system.atoms.positions,
            err_msg="The coordinates do not match.",
        )

    def test_masses(self: Self, universe: mda.Universe, system: mda.Universe, model: calpha.Model) -> None:
        masses = [
            residue.atoms.select_atoms(selection).total_mass()
            for residue, selection in itertools.product(universe.residues, model._selection.values())
            if residue.atoms.select_atoms(selection)
        ]
        testing.assert_allclose(system.atoms.masses, masses, err_msg="The masses do not match.")

    def test_charges(self: Self, universe: mda.Universe, system: mda.Universe, model: calpha.Model) -> None:
        charges = np.asarray(
            [
                residue.atoms.select_atoms(selection).total_charge()
                for residue, selection in itertools.product(universe.residues, model._selection.values())
                if residue.atoms.select_atoms(selection)
            ],
            dtype=float,
        )
        testing.assert_allclose(system.atoms.charges, charges, err_msg="The charges do not match.")

    def test_trajectory(self: Self, universe: mda.Universe, system: mda.Universe) -> None:
        assert (
            system.trajectory.n_frames == universe.trajectory.n_frames
        ), "All-atom and coarse-grain trajectories unequal."

    def test_bonds(self: Self, system: mda.Universe) -> None:
        assert len(system.bonds) > 0, "Number of bonds should be > 0."

    def test_angles(self: Self, system: mda.Universe) -> None:
        assert len(system.angles) > 0, "Number of angles should be > 0."

    def test_dihedrals(self: Self, system: mda.Universe) -> None:
        assert len(system.dihedrals) > 0, "Number of dihedral angles should be > 0."

    def test_impropers(self: Self, system: mda.Universe) -> None:
        assert len(system.impropers) == 0, "Number of improper angles should not be > 0."


class TestCaside(TestCalpha):
    @staticmethod
    @pytest.fixture()
    def model() -> caside.Model:
        return caside.Model(guess_angles=True)

    def test_impropers(self: Self, system: mda.Universe) -> None:
        assert len(system.impropers) > 0, "Number of improper angles should be > 0."


class TestNcsc(TestCaside):
    @staticmethod
    @pytest.fixture()
    def model() -> ncsc.Model:
        return ncsc.Model(guess_angles=True)

    def test_masses(self: Self, universe: mda.Universe, system: mda.Universe, model: calpha.Model) -> None:
        ca_masses = {residue: 0.5 * residue.atoms.select_atoms("hcalpha").total_mass() for residue in universe.residues}
        masses = []
        for residue in universe.residues:
            ca_mass = ca_masses.get(residue)
            for key, selection in filter(lambda x: residue.atoms.select_atoms(x[1]), model._selection.items()):
                if key != "CB":
                    masses.append(residue.atoms.select_atoms(selection).total_mass() + ca_mass)
                else:
                    masses.append(residue.atoms.select_atoms(selection).total_mass())

        testing.assert_allclose(system.atoms.masses, masses, err_msg="The masses do not match.")

    def test_charges(self: Self, universe: mda.Universe, system: mda.Universe, model: calpha.Model) -> None:
        ca_charges = {
            residue: 0.5 * residue.atoms.select_atoms("hcalpha").total_charge() for residue in universe.residues
        }
        charges = []
        for residue in universe.residues:
            ca_charge = ca_charges.get(residue)
            for key, selection in filter(lambda x: residue.atoms.select_atoms(x[1]), model._selection.items()):
                if key != "CB":
                    charges.append(residue.atoms.select_atoms(selection).total_charge() + ca_charge)
                else:
                    charges.append(residue.atoms.select_atoms(selection).total_charge())

        testing.assert_allclose(system.atoms.charges, charges, err_msg="The charges do not match.")


class TestPolar(TestNcsc):
    @staticmethod
    @pytest.fixture()
    def model() -> polar.Model:
        return polar.Model(guess_angles=True)

    @staticmethod
    @pytest.fixture()
    def other() -> ncsc.Model:
        return ncsc.Model(guess_angles=True)

    @staticmethod
    @pytest.fixture()
    def other_system(universe: mda.Universe, other: ncsc.Model) -> mda.Universe:
        return other.transform(universe)

    def test_ncsc_polar_positions(self: Self, system: mda.Universe, other_system: mda.Universe) -> None:
        with pytest.raises(AssertionError):
            testing.assert_allclose(system.atoms.positions, other_system.atoms.positions)
