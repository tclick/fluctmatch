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
"""Tests for CHARMM stream file."""

from pathlib import Path
from typing import Self

import MDAnalysis as mda
import pytest
from fluctmatch.io.charmm.stream import CharmmStream

from tests.datafile import FLUCTDCD, FLUCTPSF


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
        return mda.Universe(FLUCTPSF, FLUCTDCD)

    @pytest.fixture()
    def stream_file(self: Self, tmp_path: Path) -> Path:
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

    def test_initialize(self: Self, universe: mda.Universe) -> None:
        """Test initialization of a stream file.

        GIVEN an elastic network model
        WHEN a stream object is initialized
        THEN segment IDs, residue numbers, and atom names are stored

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            Elastic network model
        """
        stream = CharmmStream()
        stream.initialize(universe)

        assert len(stream._lines) == len(universe.bonds)
        assert "PROA" in stream._lines[0]

    def test_initialize_error(self: Self) -> None:
        """Test if an error is raised when initializing an empty universe.

        GIVEN an empty universe with no bond information
        WHEN the stream object is initialized
        THEN an attribute error is raised.
        """
        universe = mda.Universe.empty(0)
        stream = CharmmStream()

        with pytest.raises(AttributeError):
            stream.initialize(universe)

    def test_write(self: Self, universe: mda.Universe, stream_file: Path) -> None:
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
