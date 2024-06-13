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

from tests.datafile import DCD_CG, IC_FLUCT, PSF_ENM


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
        return mda.Universe(PSF_ENM, DCD_CG)

    @pytest.fixture(scope="class")
    def bonds(
        self,
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
        return OrderedDict(dict(zip(universe.bonds.types(), universe.bonds.values(), strict=True)))

    @pytest.fixture()
    def intcor_file(self, tmp_path: Path) -> Path:
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

    def test_initialize(self, universe: mda.Universe, bonds: BondData) -> None:
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
        intcor = CharmmInternalCoordinates().initialize(universe, data=bonds)

        testing.assert_equal(intcor._table.keys(), bonds.keys())
        data = np.array(list(intcor._table.values()), dtype=object)[:, -5]
        testing.assert_allclose(data.astype(float), np.fromiter(list(bonds.values()), dtype=float), rtol=1e-05)

    def test_initialize_unequal_size(self, universe: mda.Universe, bonds: BondData) -> None:
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
        distances.pop(("A00001", "A00002"))

        with pytest.raises(ValueError, match="Provided data does not match bonds in universe."):
            CharmmInternalCoordinates().initialize(universe, data=distances)

    def test_initialize_no_bonds(self: Self) -> None:
        """Test initialization of an internal coordinate file with no bond data.

        GIVEN an elastic network model with no bonds, bond force constants, and bond lengths
        WHEN the three variables are of unequal size
        THEN an internal coordinates object is initialized.
        """
        universe = mda.Universe.empty(0)

        with pytest.raises(mda.NoDataError):
            CharmmInternalCoordinates().initialize(universe)

    def test_write(self, universe: mda.Universe, intcor_file: Path, bonds: BondData) -> None:
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
        intcor = CharmmInternalCoordinates().initialize(universe, data=bonds)

        intcor.write(intcor_file)
        assert intcor_file.exists()
        assert intcor_file.stat().st_size > 0

    def test_read(self: Self) -> None:
        """Test reading an internal coordinates file.

        GIVEN an internal coordinates file
        WHEN an internal coordinates object is initialized and read
        THEN an internal coordinates file is read and loaded into the parameter object.
        """
        intcor = CharmmInternalCoordinates().read(IC_FLUCT)

        assert len(intcor._table) > 0, "Internal coordinates file is empty."

    def test_read_no_file(self: Self) -> None:
        """Test reading non-existent parameter file.

        GIVEN a non-existent parameter filename
        WHEN an internal coordinates object is initialized and read
        THEN a FileNotFoundError is raised.
        """
        intcor_file = "fake.ic"
        with pytest.raises(FileNotFoundError):
            CharmmInternalCoordinates().read(intcor_file)

    def test_intcor_property(self, universe: mda.Universe, bonds: BondData) -> None:
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
        intcor = CharmmInternalCoordinates().initialize(universe, data=bonds)

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

    def test_data_property(self, universe: mda.Universe, bonds: BondData) -> None:
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
        intcor = CharmmInternalCoordinates().initialize(universe, data=bonds)

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

    def test_data_property_fail(self, universe: mda.Universe, bonds: BondData) -> None:
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
        intcor = CharmmInternalCoordinates().initialize(universe, data=bonds)

        # Test setter
        with pytest.raises(IndexError):
            intcor.data = np.zeros_like(bonds.values())[:-1]

    def test_dataframe(self: Self) -> None:
        """Test the conversion of an internal coordinates file to a `pandas.DataFrame`.

        GIVEN an internal coordinates filename
        WHEN an internal coordinates object is read
        THEN a `pandas.DataFrame` is created.
        """
        ic = CharmmInternalCoordinates().read(IC_FLUCT)
        table = ic.to_dataframe()

        assert table.reset_index().shape == ic.table.shape, "The DataFrame does not have the expected shape."
        testing.assert_equal(table, ic.table[:, 1:], err_msg="DataFrame does not match the IC table.")

    def test_series(self: Self) -> None:
        """Test the conversion of an internal coordinates file to a `pandas.Series`.

        GIVEN an internal coordinates filename
        WHEN an internal coordinates object is read
        THEN a `pandas.Series` is created.
        """
        ic = CharmmInternalCoordinates().read(IC_FLUCT)
        table = ic.to_series()

        assert table.size == ic.table.shape[0], "The Series does not have the expected size."
        testing.assert_allclose(table, ic.table[:, -5].astype(float), err_msg="Series does not match the IC table.")
