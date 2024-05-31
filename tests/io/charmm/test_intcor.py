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
"""Test I/O for CHARMM internal coordinate files."""

from pathlib import Path
from typing import Self

import MDAnalysis as mda
import numpy as np
import pytest
from fluctmatch.io.charmm.intcor import CharmmInternalCoordinates
from numpy import testing
from pyfakefs import fake_file, fake_filesystem

from tests.datafile import FLUCTDCD, FLUCTPSF, IC


class TestCharmmInternalCoordinates:
    """Test CHARMM internal coordinate file object."""

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
    def intcor_file(self: Self, fs_class: fake_filesystem.FakeFilesystem) -> fake_file.FakeFile:
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
        return fs_class.create_file("charmm.ic")

    def test_initialize(self: Self, universe: mda.Universe) -> None:
        """Test initialization of an internal coordinate file.

        GIVEN an elastic network model and bond lengths
        WHEN an internal coordinate object is provided
        THEN an internal coordinate object is initialized.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            Elastic network model
        """
        lengths = universe.bonds.values()
        intcor = CharmmInternalCoordinates()
        intcor.initialize(universe, data=lengths)

        testing.assert_equal(intcor._table[:, -5], lengths)

    def test_initialize_unequal_size(self: Self, universe: mda.Universe) -> None:
        """Test initialization of an internal coordinate file with unequal sizes of bonds and bond lengths.

        GIVEN an elastic network model with bonds and bond lengths
        WHEN the two variables are of unequal size
        THEN an ValueError is raised.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            Elastic network model
        """
        lengths = universe.bonds.values()[:-2]
        intcor = CharmmInternalCoordinates()

        with pytest.raises(ValueError):
            intcor.initialize(universe, data=lengths[:-5])

    def test_initialize_no_bonds(self: Self) -> None:
        """Test initialization of an internal coordinate file with no bond data.

        GIVEN an elastic network model with no bonds, bond force constants, and bond lengths
        WHEN the three variables are of unequal size
        THEN an internal coordinates object is initialized.
        """
        universe = mda.Universe.empty(0)
        lengths = np.zeros(5, dtype=float)
        intcor = CharmmInternalCoordinates()

        with pytest.raises(mda.NoDataError):
            intcor.initialize(universe, data=lengths)

    def test_write(self: Self, universe: mda.Universe, intcor_file: fake_file.FakeFile) -> None:
        """Test writing an internal coordinates file.

        GIVEN a universe and an internal coordinates file
        WHEN an internal coordinates object is initialized
        THEN an internal coordinates object is written.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            An elastic network model
        intcor_file : :class:`pyfakefs.fake_file.FakeFile
            Empty file in memory
        """
        filename = Path(intcor_file.name)
        lengths = universe.bonds.values()
        intcor = CharmmInternalCoordinates()
        intcor.initialize(universe, data=lengths)

        intcor.write(filename)
        assert filename.exists()
        assert filename.stat().st_size > 0

    def test_read(self: Self) -> None:
        """Test reading an internal coordinates file.

        GIVEN an internal coordinates file
        WHEN an internal coordinates object is initialized and read
        THEN an internal coordinates file is read and loaded into the parameter object.
        """
        intcor = CharmmInternalCoordinates()
        intcor.read(IC)

        assert intcor._table.size > 0, "Internal coordinates file is empty."

    def test_read_no_file(self: Self) -> None:
        """Test reading non-existent parameter file.

        GIVEN a non-existent parameter filename
        WHEN an internal coordinates object is initialized and read
        THEN a FileNotFoundError is raised.
        """
        intcor_file = "charmm.ic"
        intcor = CharmmInternalCoordinates()
        with pytest.raises(FileNotFoundError):
            intcor.read(intcor_file)

    def test_intcor_property(self: Self, universe: mda.Universe) -> None:
        """Test parameters property.

        GIVEN an elastic network model, equilibrium force constants, and bond lengths
        WHEN an internal coordinates object is initialized and the parameters property is called
        THEN the underlying parameter object is returned.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            An elastic network model
        """
        # Initialize parameters
        lengths = universe.bonds.values()
        intcor = CharmmInternalCoordinates()
        intcor.initialize(universe, data=lengths)

        # Test parameters getter
        table = intcor.table
        assert table.size > 0, "Internal coordinates file is empty."
        testing.assert_allclose(table[:, -5].astype(float), lengths, rtol=1e-05, atol=1e-08)

    def test_data_property(self: Self, universe: mda.Universe) -> None:
        """Test getter/setter of data property.

        GIVEN an elastic network model, equilibrium force constants, and bond lengths
        WHEN an internal coordinates object is initialized and the forces property is called
        THEN force constants are retrieved or set.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            An elastic network model
        """
        lengths = universe.bonds.values()
        intcor = CharmmInternalCoordinates()
        intcor.initialize(universe, data=lengths)

        # Test getter
        data = intcor.data
        testing.assert_allclose(data, lengths, rtol=1e-05, atol=1e-08, err_msg="Forces don't match.", verbose=True)

        # Test setter
        intcor.data = np.zeros_like(lengths)
        testing.assert_allclose(intcor.data, 0.0, rtol=1e-05, atol=1e-08, err_msg="Forces don't match.", verbose=True)

    def test_data_property_fail(self: Self, universe: mda.Universe) -> None:
        """Test setter of data property.

        GIVEN an elastic network model, equilibrium force constants, and bond lengths
        WHEN an internal coordinates object is initialized and a shorter forces array is given
        THEN a ValueError is raised.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            An elastic network model
        """
        lengths = universe.bonds.values()
        intcor = CharmmInternalCoordinates()
        intcor.initialize(universe, data=lengths)

        # Test setter
        with pytest.raises(IndexError):
            intcor.data = np.zeros_like(lengths)[:-1]
