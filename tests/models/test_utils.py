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

import MDAnalysis as mda
import numpy as np
import pytest
from fluctmatch.core import utils
from fluctmatch.core.models import polar
from numpy import testing

from .. import Self
from ..datafile import TPR, XTC


class TestMerge:
    @staticmethod
    @pytest.fixture()
    def universe() -> mda.Universe:
        return mda.Universe(TPR, XTC)

    def test_creation(self: Self, universe: mda.Universe) -> None:
        univ_tuple: tuple[mda.Universe, ...] = (universe, universe)
        new_universe: mda.Universe = utils.merge(*univ_tuple)

        n_atoms = universe.atoms.n_atoms * 2
        assert new_universe.atoms.n_atoms == n_atoms, "Number of sites don't match."

    def test_positions(self: Self, universe: mda.Universe) -> None:
        n_atoms = universe.atoms.n_atoms
        univ_tuple: tuple[mda.Universe, ...] = (universe, universe)
        new_universe: mda.Universe = utils.merge(*univ_tuple)

        positions = np.concatenate([universe.atoms.positions for u in univ_tuple])

        testing.assert_allclose(new_universe.atoms.positions, positions, err_msg="Coordinates don't match.")
        testing.assert_allclose(
            new_universe.atoms.positions[0],
            new_universe.atoms.positions[n_atoms],
            err_msg=f"Coordinates 0 and {n_atoms:d} " f"don't match.",
        )

    def test_topology(self: Self, universe: mda.Universe) -> None:
        new_universe: mda.Universe = utils.merge(universe)

        assert universe.atoms.n_atoms == new_universe.atoms.n_atoms
        assert new_universe.bonds == universe.bonds, "Bonds differ."
        assert new_universe.angles == universe.angles, "Angles differ."
        assert new_universe.dihedrals == universe.dihedrals, "Dihedrals differ."


class TestModeller:
    @staticmethod
    @pytest.fixture()
    def u() -> mda.Universe:
        return mda.Universe(TPR, XTC)

    @staticmethod
    @pytest.fixture()
    def u2() -> mda.Universe:
        return utils.modeller(TPR, XTC, "polar")

    @staticmethod
    @pytest.fixture()
    def system() -> polar.Model:
        return polar.Model()

    def test_creation(self: Self, u: mda.Universe, u2: mda.Universe, system: polar.Model):
        u3 = system.transform(u)

        testing.assert_raises(AssertionError, testing.assert_equal, (u.atoms.n_atoms,), (u2.atoms.n_atoms,))
        testing.assert_equal(u2.atoms.names, u3.atoms.names, err_msg="Universes don't match.")


class TestRename:
    @staticmethod
    @pytest.fixture()
    def universe() -> mda.Universe:
        return mda.Universe(TPR, XTC)

    def test_rename_universe(self: Self, universe: mda.Universe) -> None:
        utils.rename_universe(universe)

        testing.assert_string_equal(universe.atoms[0].name, "A001")
        testing.assert_string_equal(universe.atoms[-1].name, "F001")
        testing.assert_string_equal(universe.residues[0].resname, "A001")
