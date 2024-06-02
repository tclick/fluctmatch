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

from collections import OrderedDict
from pathlib import Path
from typing import Self

import MDAnalysis as mda
import numpy as np
import pytest
from fluctmatch.io.charmm import BondData
from fluctmatch.io.charmm.intcor import CharmmInternalCoordinates
from numpy import testing

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
    def bonds(
        self: Self,
        universe: mda.Universe,
    ) -> BondData:
        """Define dictionary of bonds in CHARMM model.

        Parameters
        ----------
        MDAnalysis.Universe
            Elastic network model

        Returns
        -------
        OrderedDict[tuple[str, str], float]
            Bond data
        """
        return OrderedDict(dict(zip(universe.bonds.topDict.keys(), universe.bonds.values(), strict=True)))

    @pytest.fixture()
    def intcor_file(self: Self, tmp_path: Path) -> Path:
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
        return tmp_path / "charmm.ic"

    def test_initialize(self: Self, universe: mda.Universe, bonds: BondData) -> None:
        """Test initialization of an internal coordinate file.

        GIVEN an elastic network model and bond lengths
        WHEN an internal coordinate object is provided
        THEN an internal coordinate object is initialized.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            Elastic network model
        bonds : OrderedDict[tuple[str, str], float]
            Bond data
        """
        intcor = CharmmInternalCoordinates()
        intcor.initialize(universe, data=bonds)

        testing.assert_equal(intcor._table.keys(), bonds.keys())
        data = np.array(list(intcor._table.values()), dtype=object)[:, -5]
        testing.assert_allclose(data.astype(float), np.fromiter(list(bonds.values()), dtype=float), rtol=1e-05)

    def test_initialize_unequal_size(self: Self, universe: mda.Universe, bonds: BondData) -> None:
        """Test initialization of an internal coordinate file with unequal sizes of bonds and bond lengths.

        GIVEN an elastic network model with bonds and bond lengths
        WHEN the two variables are of unequal size
        THEN an ValueError is raised.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            Elastic network model
        bonds : OrderedDict[tuple[str, str], float]
            Bond data
        """
        distances = bonds.copy()
        distances.pop(("C00001", "C00002"))
        intcor = CharmmInternalCoordinates()

        with pytest.raises(ValueError, match="Provided data does not match bonds in universe."):
            intcor.initialize(universe, data=distances)

    def test_initialize_no_bonds(self: Self) -> None:
        """Test initialization of an internal coordinate file with no bond data.

        GIVEN an elastic network model with no bonds, bond force constants, and bond lengths
        WHEN the three variables are of unequal size
        THEN an internal coordinates object is initialized.
        """
        universe = mda.Universe.empty(0)
        intcor = CharmmInternalCoordinates()

        with pytest.raises(mda.NoDataError):
            intcor.initialize(universe)

    def test_write(self: Self, universe: mda.Universe, intcor_file: Path, bonds: BondData) -> None:
        """Test writing an internal coordinates file.

        GIVEN a universe and an internal coordinates file
        WHEN an internal coordinates object is initialized
        THEN an internal coordinates object is written.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            An elastic network model
        intcor_file : Path
            Empty file in memory
        bonds : OrderedDict[tuple[str, str], float]
            Bond data
        """
        intcor = CharmmInternalCoordinates()
        intcor.initialize(universe, data=bonds)

        intcor.write(intcor_file)
        assert intcor_file.exists()
        assert intcor_file.stat().st_size > 0

    def test_read(self: Self) -> None:
        """Test reading an internal coordinates file.

        GIVEN an internal coordinates file
        WHEN an internal coordinates object is initialized and read
        THEN an internal coordinates file is read and loaded into the parameter object.
        """
        intcor = CharmmInternalCoordinates()
        intcor.read(IC)

        assert len(intcor._table) > 0, "Internal coordinates file is empty."

    def test_read_no_file(self: Self) -> None:
        """Test reading non-existent parameter file.

        GIVEN a non-existent parameter filename
        WHEN an internal coordinates object is initialized and read
        THEN a FileNotFoundError is raised.
        """
        intcor_file = "fake.ic"
        intcor = CharmmInternalCoordinates()
        with pytest.raises(FileNotFoundError):
            intcor.read(intcor_file)

    def test_intcor_property(self: Self, universe: mda.Universe, bonds: BondData) -> None:
        """Test parameters property.

        GIVEN an elastic network model, equilibrium force constants, and bond lengths
        WHEN an internal coordinates object is initialized and the parameters property is called
        THEN the underlying parameter object is returned.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            An elastic network model
        intcor_file : Path
            Empty file in memory
        bonds : OrderedDict[tuple[str, str], float]
            Bond data
        """
        # Initialize parameters
        intcor = CharmmInternalCoordinates()
        intcor.initialize(universe, data=bonds)

        # Test parameters getter
        table = intcor.table
        assert table.size > 0, "Internal coordinates file is empty."
        testing.assert_allclose(
            table[:, -5].astype(float),
            np.fromiter(list(bonds.values()), dtype=float),
            rtol=1e-05,
            atol=1e-08,
            err_msg="Some bond values are not close.",
        )

    def test_data_property(self: Self, universe: mda.Universe, bonds: BondData) -> None:
        """Test getter/setter of data property.

        GIVEN an elastic network model, equilibrium force constants, and bond lengths
        WHEN an internal coordinates object is initialized and the forces property is called
        THEN force constants are retrieved or set.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            An elastic network model
        intcor_file : Path
            Empty file in memory
        bonds : OrderedDict[tuple[str, str], float]
            Bond data
        """
        intcor = CharmmInternalCoordinates()
        intcor.initialize(universe, data=bonds)

        # Test getter
        data = intcor.data
        testing.assert_allclose(
            list(data.values()),
            list(bonds.values()),
            rtol=1e-05,
            atol=1e-08,
            err_msg="Bond data does not match.",
            verbose=True,
        )

        # Test setter
        intcor.data = OrderedDict({k: 0.0 for k in bonds})
        testing.assert_allclose(
            list(intcor.data.values()), 0.0, rtol=1e-05, atol=1e-08, err_msg="Forces don't match.", verbose=True
        )

    def test_data_property_fail(self: Self, universe: mda.Universe, bonds: BondData) -> None:
        """Test setter of data property.

        GIVEN an elastic network model, equilibrium force constants, and bond lengths
        WHEN an internal coordinates object is initialized and a shorter forces array is given
        THEN a ValueError is raised.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            An elastic network model
        bonds : OrderedDict[tuple[str, str], float]
            Bond data
        """
        intcor = CharmmInternalCoordinates()
        intcor.initialize(universe, data=bonds)

        # Test setter
        with pytest.raises(IndexError):
            intcor.data = np.zeros_like(bonds.values())[:-1]
