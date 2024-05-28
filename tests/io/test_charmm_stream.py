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
from fluctmatch.io.charmm_stream import CharmmStream
from pyfakefs import fake_file, fake_filesystem

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

    @pytest.fixture(scope="class")
    def stream_file(self: Self, fs_class: fake_filesystem.FakeFilesystem) -> fake_file.FakeFile:
        """Return an empty file.

        Parameters
        ----------
        fs_class : :class:`pyfakefs.fake_filesystem.FakeFileSystem`
            Filesystem

        Returns
        -------
        :class:`pyfakefs.fake_file.FakeFile
            Empty file in memory
        """
        return fs_class.create_file("charmm.str")

    def test_initialize(self: Self, universe: mda.Universe, stream_file: fake_file.FakeFile) -> None:
        """Test initialization of a stream file.

        GIVEN an elastic network model
        WHEN a stream object is initialized
        THEN segment IDs, residue numbers, and atom names are stored

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            Elastic network model
        stream_file : :class:`pyfakefs.fake_file.FakeFile
            Empty file in memory
        """
        stream = CharmmStream(filename=stream_file)
        stream.initialize(universe)

        assert len(stream._lines) == len(universe.bonds)
        assert "PROA" in stream._lines[0]

    def test_initialize_error(self: Self, stream_file: fake_file.FakeFile) -> None:
        """Test if an error is raised when initializing an empty universe.

        GIVEN an empty universe with no bond information
        WHEN the stream object is initialized
        THEN an attribute error is raised.

        Parameters
        ----------
        stream_file : :class:`pyfakefs.fake_file.FakeFile
            Empty file in memory
        """
        universe = mda.Universe.empty(0)
        stream = CharmmStream(filename=stream_file)

        with pytest.raises(AttributeError):
            stream.initialize(universe)

    @pytest.mark.asyncio()
    async def test_write(self: Self, universe: mda.Universe, tmp_path: Path) -> None:
        """Test write method.

        GIVEN a universe and an initialized CHARMM stream object
        WHEN the write method is called
        THEN a stream file is written.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            Elastic network model
        tmp_path : Path
            Location for temporary file
        """
        filename = tmp_path / "charmm.str"
        stream = CharmmStream(filename=filename)
        stream.initialize(universe)

        await stream.write()

        assert filename.exists()
        assert filename.stat().st_size > 0
