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
"""Tests for various utility functions."""

from typing import Self

import MDAnalysis as mda
import pytest
from fluctmatch.libs.utils import merge
from MDAnalysis.coordinates.base import ReaderBase
from MDAnalysisTests.datafiles import DCD2, PSF, TPR, XTC


class TestMerge:
    """Test utility functions."""

    @pytest.fixture(scope="class")
    def aa_universe(self: Self) -> mda.Universe:
        """Create an all-atom universe.

        Returns
        -------
        mda.Universe
            An all-atom universe
        """
        return mda.Universe(PSF, DCD2)

    @pytest.fixture(scope="class")
    def cg_universe(self: Self) -> mda.Universe:
        """Create a coarse-grain universe.

        Returns
        -------
        mda.Universe
            A coarse-grain universe
        """
        return mda.Universe(TPR, XTC)

    def test_merge(self: Self, aa_universe: mda.Universe) -> None:
        """Test the merging of two universes.

        GIVEN a universe
        WHEN merging it with itself
        THEN check the universe is doubled in size with the same trajectory length
        """
        atoms: mda.AtomGroup = aa_universe.atoms
        trajectory: ReaderBase = aa_universe.trajectory
        n_atoms: int = atoms.n_atoms
        n_frames: int = trajectory.n_frames

        merged: mda.Universe = merge(aa_universe, aa_universe)
        assert hasattr(merged, "atoms")
        assert hasattr(merged, "trajectory")
        assert merged.atoms.n_atoms == n_atoms * 2
        assert merged.trajectory.n_frames == n_frames

    def test_merge_fail(self: Self, aa_universe: mda.Universe, cg_universe: mda.Universe) -> None:
        """Test the merging of two universes fails.

        GIVEN two universes with unequal length trajectories
        WHEN merging them together
        THEN a ValueError should be raised
        """
        with pytest.raises(ValueError):
            merge(aa_universe, cg_universe)
