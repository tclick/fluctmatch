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
# pyright: reportOptionalSubscript=false, reportGeneralTypeIssues=false, reportInvalidTypeVarUse=false
# pyright: reportWildcardImportFromLibrary=false
# flake8: noqa

import itertools
from types import MappingProxyType

import MDAnalysis as mda
import numpy as np
import pytest
from numpy import testing
from numpy.typing import NDArray

from fluctmatch.core.models import united
from ..datafile import TPR, XTC
from .. import Self

# Number of residues to test
N_RESIDUES = 6


class TestUnited:
    @pytest.fixture()
    def universe(self: Self) -> mda.Universe:
        return mda.Universe(TPR, XTC)

    @pytest.fixture()
    def model(self: Self) -> united.Model:
        return united.Model(guess_angles=True)

    @pytest.fixture()
    def system(self: Self, universe: mda.Universe, model: united.Model) -> mda.Universe:
        return model.transform(universe)

    def test_creation(self: Self, universe: mda.Universe, model: united.Model, system: mda.Universe) -> None:
        n_atoms = 0
        for residue, selection in itertools.product(universe.residues, model._mapping.values()):
            value = selection.get(residue.resname) if isinstance(selection, MappingProxyType) else selection
            n_atoms += residue.atoms.select_atoms(value).n_atoms
        testing.assert_equal(
            system.atoms.n_atoms,
            n_atoms,
            err_msg="Number of sites don't match.",
        )

    def test_positions(self: Self, universe: mda.Universe, system: mda.Universe, model: united.Model) -> None:
        positions: list[NDArray] = []
        for residue, selection in itertools.product(universe.residues, model._mapping.values()):
            value = (
                selection.get(residue.resname, "hsidechain and not name H*")
                if isinstance(selection, MappingProxyType)
                else selection
            )
            if residue.atoms.select_atoms(value):
                positions.append(residue.atoms.select_atoms(value).positions)
        positon_array = np.concatenate(positions, axis=0)
        testing.assert_allclose(
            positon_array,
            system.atoms.positions,
            err_msg="The coordinates do not match.",
        )

    def test_masses(self: Self, universe: mda.Universe, system: mda.Universe, model: united.Model) -> None:
        masses = np.concatenate(
            [
                residue.atoms.select_atoms(selection).masses
                for residue, selection in itertools.product(universe.residues, model._selection.values())
                if residue.atoms.select_atoms(selection)
            ]
        )
        testing.assert_allclose(system.atoms.masses, masses, err_msg="The masses do not match.")

    def test_charges(self: Self, universe: mda.Universe, system: mda.Universe, model: united.Model) -> None:
        try:
            charges = np.concatenate(
                [
                    residue.atoms.select_atoms(selection).charges
                    for residue, selection in itertools.product(universe.residues, model._selection.values())
                    if residue.atoms.select_atoms(selection)
                ]
            )
        except mda.NoDataError:
            charges = [0.0] * system.atoms.n_atoms
        testing.assert_allclose(
            system.atoms.charges,
            charges,
            err_msg="The charges do not match.",
        )

    def test_trajectory(self: Self, universe: mda.Universe, system: mda.Universe) -> None:
        testing.assert_equal(
            system.trajectory.n_frames,
            universe.trajectory.n_frames,
            err_msg="All-atom and coarse-grain trajectories unequal.",
        )

    def test_bonds(self: Self, system: mda.Universe) -> None:
        assert len(system.bonds) > 0, "Number of bonds should be > 0."

    def test_angles(self: Self, system: mda.Universe) -> None:
        assert len(system.angles) > 0, "Number of angles should be > 0."

    def test_dihedrals(self: Self, system: mda.Universe) -> None:
        assert len(system.dihedrals) > 0, "Number of dihedral angles should be > 0."

    def test_impropers(self: Self, system: mda.Universe) -> None:
        assert len(system.impropers) > 0, "Number of improper angles should be > 0."
