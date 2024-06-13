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
"""Tests for CHARMM stream file."""

from pathlib import Path
from typing import Self

import MDAnalysis as mda
import pytest
from fluctmatch.io.charmm.stream import CharmmStream
from testfixtures import ShouldRaise

from tests.datafile import DCD_CG, PSF_ENM


class TestCharmmStream:
    """Test CHARMM stream file object."""

    @pytest.fixture(scope="class")
    def universe(self: Self) -> mda.Universe:
        """Universe of an elastic network model.

        Returns
        -------
        MDAnalysis.Universe
            Elastic network model
        """
        return mda.Universe(PSF_ENM, DCD_CG)

    @pytest.fixture()
    def stream_file(self, tmp_path: Path) -> Path:
        """Return an empty file.

        Parameters
        ----------
        tmp_path : Path
            Filesystem

        Returns
        -------
        Path
            Empty file in memory
        """
        return tmp_path / "charmm.str"

    def test_initialize(self, universe: mda.Universe) -> None:
        """Test initialization of a stream file.

        GIVEN an elastic network model
        WHEN a stream object is initialized
        THEN segment IDs, residue numbers, and atom names are stored

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            Elastic network model
        """
        stream = CharmmStream().initialize(universe)

        assert len(stream._lines) == len(universe.bonds)
        assert "4AKE" in stream._lines[0]

    def test_initialize_error(self: Self) -> None:
        """Test if an error is raised when initializing an empty universe.

        GIVEN an empty universe with no bond information
        WHEN the stream object is initialized
        THEN an attribute error is raised.
        """
        universe = mda.Universe.empty(0)

        with pytest.raises(AttributeError):
            CharmmStream().initialize(universe)

    def test_write(self, universe: mda.Universe, stream_file: Path) -> None:
        """Test write method.

        GIVEN a universe and an initialized CHARMM stream object
        WHEN the write method is called
        THEN a stream file is written.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            Elastic network model
        stream_file : Path
            Filename
        """
        stream = CharmmStream()
        stream.initialize(universe)

        stream.write(stream_file)

        assert stream_file.exists()
        assert stream_file.stat().st_size > 0

    def test_read(self: Self) -> None:
        """Test read method.

        GIVEN an elastic network model
        WHEN the read method is called
        THEN an exception is raised.
        """
        with ShouldRaise(NotImplementedError):
            CharmmStream().read(DCD_CG)
