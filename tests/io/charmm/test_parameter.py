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
"""Test I/O for CHARMM parameter files."""

from collections import OrderedDict
from pathlib import Path
from typing import Self

import MDAnalysis as mda
import numpy as np
import pytest
from fluctmatch.io.charmm import BondData
from fluctmatch.io.charmm.parameter import CharmmParameter
from numpy import testing
from testfixtures import ShouldRaise

from tests.datafile import DCD_CG, PRM, PSF_ENM, RTF, STR


class TestCharmmParameter:
    """Test CHARMM parameter file object."""

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
    def param_file(self, tmp_path: Path) -> Path:
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
        return tmp_path / "charmm.prm"

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

    def test_initialize(self, universe: mda.Universe, bonds: BondData) -> None:
        """Test initialization of a parameter file.

        GIVEN an elastic network model, bond force constants, and bond lengths
        WHEN a parameter object is provided
        THEN a parameter object is initialized.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            Elastic network model
        bonds : OrderedDict[tuple[str, str], float]
            Bond data
        """
        atom_types = OrderedDict({atom.type: "" for atom in universe.atoms})
        param = CharmmParameter().initialize(universe, forces=bonds, lengths=bonds)

        testing.assert_equal(param._parameters.atom_types.keys(), atom_types.keys())
        assert all(key in param._parameters.bond_types for key in universe.bonds.topDict)
        testing.assert_allclose(
            [at.mass for at in param._parameters.atom_types.values()], universe.atoms.masses, rtol=1e-4
        )
        testing.assert_allclose(
            [bt.k for bt in param._parameters.bond_types.values()], np.fromiter(bonds.values(), dtype=float), rtol=1e-4
        )
        testing.assert_allclose(
            [bt.req for bt in param._parameters.bond_types.values()],
            np.fromiter(bonds.values(), dtype=float),
            rtol=1e-4,
        )

    def test_no_forces_or_distances(self: Self, universe: mda.Universe) -> None:
        """Test initialization of a parameter file when no force constants or distances are provided.

        GIVEN an elastic network model
        WHEN no force constants or distances are provided
        THEN a parameter object is initialized with default values of 0.0 for both.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            Elastic network model
        """
        atom_types = OrderedDict({atom.type: "" for atom in universe.atoms})
        param = CharmmParameter().initialize(universe)

        testing.assert_equal(param._parameters.atom_types.keys(), atom_types.keys())
        assert all(key in param._parameters.bond_types for key in universe.bonds.topDict)
        testing.assert_allclose(
            [at.mass for at in param._parameters.atom_types.values()], universe.atoms.masses, rtol=1e-4
        )
        testing.assert_allclose([bt.k for bt in param._parameters.bond_types.values()], 0.0, rtol=1e-4)
        testing.assert_allclose([bt.req for bt in param._parameters.bond_types.values()], 0.0, rtol=1e-4)

    def test_initialize_unequal_size(self, universe: mda.Universe, bonds: BondData) -> None:
        """Test initialization of a parameter file with unequal sizes of forces and bond lengths.

        GIVEN an elastic network model with bonds, bond force constants, and bond lengths
        WHEN the three variables are of unequal size
        THEN a ValueError is raised.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            Elastic network model
        bonds : OrderedDict[tuple[str, str], float]
            Bond data
        """
        distances = bonds.copy()
        distances.pop(("A00001", "A00002"))

        with ShouldRaise(ValueError):
            CharmmParameter().initialize(universe, forces=bonds, lengths=distances)

        with ShouldRaise(ValueError):
            CharmmParameter().initialize(universe, forces=distances, lengths=distances)

    def test_initialize_no_bonds(self: Self) -> None:
        """Test initialization of a parameter file with no bond data.

        GIVEN an elastic network model with no bonds, bond force constants, and bond lengths
        WHEN the three variables are of unequal size
        THEN a parameter object is initialized.

        """
        n_atoms = 5
        universe = mda.Universe.empty(n_atoms)
        forces = np.zeros(n_atoms, dtype=float)

        with ShouldRaise(mda.NoDataError):
            CharmmParameter().initialize(universe, forces=forces, lengths=forces)

    def test_write(self, universe: mda.Universe, param_file: Path, bonds: BondData) -> None:
        """Test writing a parameter file.

        GIVEN a universe and a parameter file
        WHEN a parameter object is initialized
        THEN a parameter object is written.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            An elastic network model
        param_file : Path
            Empty file in memory
        bonds : OrderedDict[tuple[str, str], float]
            Bond data
        """
        param = CharmmParameter().initialize(universe, forces=bonds, lengths=bonds)

        param.write(param_file)
        assert param_file.exists()
        assert param_file.stat().st_size > 0
        assert param_file.with_suffix(".rtf").exists()
        assert param_file.with_suffix(".rtf").stat().st_size > 0

    def test_write_empty(self, param_file: Path, caplog) -> None:
        """Test writing an empty parameter file.

        GIVEN a parameter file
        WHEN a parameter object is initialized
        THEN a parameter object is written with no atom or bond types.

        Parameters
        ----------
        param_file : Path
            Empty file in memory
        caplog
            Fixture to capture log messages
        """
        param = CharmmParameter()
        param.write(param_file)

        warning = "No atom types or bond types were provided. The parameter file will be empty."
        assert warning in caplog.text
        assert param_file.exists()
        assert param_file.stat().st_size > 0

    def test_write_fail(self, universe: mda.Universe, bonds: BondData) -> None:
        """Test writing a parameter file if no filenames are provided.

        GIVEN a universe and no parameter file
        WHEN a parameter object is initialized
        THEN a ValueError is raised.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            An elastic network model
        bonds : OrderedDict[tuple[str, str], float]
            Bond data
        """
        param = CharmmParameter().initialize(universe, forces=bonds, lengths=bonds)

        with ShouldRaise(TypeError):
            param.write()

    def test_read_prm(self: Self) -> None:
        """Test reading a parameter file.

        GIVEN a parameter file
        WHEN a parameter object is initialized and read
        THEN a parameter file is read and loaded into the parameter object.
        """
        param = CharmmParameter().read(PRM)

        assert len(param._parameters.atom_types) > 0, "Atom type definitions don't exist."
        assert len(param._parameters.bond_types) > 0, "Bond definitions don't exist."

    def test_read_rtf(self: Self) -> None:
        """Test reading a topology file.

        GIVEN a topology file
        WHEN a parameter object is initialized and read
        THEN a parameter file is read and loaded into the parameter object.
        """
        param = CharmmParameter().read(RTF)

        assert len(param._parameters.atom_types) > 0, "Atom type definitions don't exist."

    def test_read_no_file(self: Self) -> None:
        """Test reading non-existent parameter file.

        GIVEN a non-existent parameter filename
        WHEN a parameter object is initialized and read
        THEN a FileNotFoundError is raised.
        """
        param_file = "charmm.str"
        with ShouldRaise(FileNotFoundError):
            CharmmParameter().read(param_file)

    def test_read_empty_file(self: Self) -> None:
        """Test reading stream file without parameter information.

        GIVEN a stream file without parameter information
        WHEN a parameter object is initialized and read
        THEN an OSError is raised.
        """
        with ShouldRaise(OSError):
            CharmmParameter().read(STR)

    def test_parameters_property(self, universe: mda.Universe, bonds: BondData) -> None:
        """Test parameters property.

        GIVEN an elastic network model, equilibrium force constants, and bond lengths
        WHEN a parameter object is initialized and the parameters property is called
        THEN the underlying parameter object is returned.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            An elastic network model
        bonds : OrderedDict[tuple[str, str], float]
            Bond data
        """
        # Initialize parameters
        param = CharmmParameter().initialize(universe, forces=bonds, lengths=bonds)

        # Test parameters getter
        parameters = param.parameters
        assert parameters.atom_types == param._parameters.atom_types
        assert parameters.bond_types == param._parameters.bond_types

    def test_forces_property(self, universe: mda.Universe, bonds: BondData) -> None:
        """Test getter/setter of forces property.

        GIVEN an elastic network model, equilibrium force constants, and bond lengths
        WHEN a parameter object is initialized and the forces property is called
        THEN force constants are retrieved or set.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            An elastic network model
        bonds : OrderedDict[tuple[str, str], float]
            Bond data
        """
        param = CharmmParameter().initialize(universe, forces=bonds, lengths=bonds)

        # Test getter
        data = param.forces
        testing.assert_allclose(
            list(data.values()),
            list(bonds.values()),
            rtol=1e-05,
            atol=1e-08,
            err_msg="Forces don't match.",
            verbose=True,
        )

        # Test setter
        param.forces = OrderedDict({k: 0.0 for k in bonds})
        testing.assert_allclose(
            list(param.forces.values()), 0.0, rtol=1e-05, atol=1e-08, err_msg="Forces don't match.", verbose=True
        )

    def test_forces_property_fail(self, universe: mda.Universe, bonds: BondData) -> None:
        """Test setter of forces property.

        GIVEN an elastic network model, equilibrium force constants, and bond lengths
        WHEN a parameter object is initialized and a shorter forces array is given
        THEN a ValueError is raised.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            An elastic network model
        bonds : OrderedDict[tuple[str, str], float]
            Bond data
        """
        param = CharmmParameter().initialize(universe, forces=bonds, lengths=bonds)

        # Test setter
        bad_bonds = bonds.copy()
        bad_bonds.pop(("A00001", "A00002"))
        with ShouldRaise(ValueError):
            param.forces = OrderedDict({k: 0.0 for k in bad_bonds})

    def test_distances_property(self, universe: mda.Universe, bonds: BondData) -> None:
        """Test getter/setter of distances property.

        GIVEN an elastic network model, equilibrium force constants, and bond lengths
        WHEN a parameter object is initialized and the distances property is called
        THEN equilibrium bond distances are retrieved or set.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            An elastic network model
        bonds : OrderedDict[tuple[str, str], float]
            Bond data
        """
        param = CharmmParameter().initialize(universe, forces=bonds, lengths=bonds)

        # Test getter
        data = param.distances
        testing.assert_allclose(
            list(data.values()),
            list(bonds.values()),
            rtol=1e-05,
            atol=1e-08,
            err_msg="Distances don't match.",
            verbose=True,
        )

        # Test setter
        param.distances = OrderedDict({k: 0.0 for k in bonds})
        testing.assert_allclose(
            list(param.distances.values()), 0.0, rtol=1e-05, atol=1e-08, err_msg="Distances don't match.", verbose=True
        )

    def test_distancess_property_fail(self, universe: mda.Universe, bonds: BondData) -> None:
        """Test setter of distances property.

        GIVEN an elastic network model, equilibrium force constants, and bond lengths
        WHEN a parameter object is initialized and a shorter distances array is given
        THEN a ValueError is raised.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            An elastic network model
        bonds : OrderedDict[tuple[str, str], float]
            Bond data
        """
        param = CharmmParameter().initialize(universe, forces=bonds, lengths=bonds)

        # Test setter
        bad_bonds = bonds.copy()
        bad_bonds.pop(("A00001", "A00002"))
        with ShouldRaise(ValueError):
            param.distances = OrderedDict({k: 0.0 for k in bad_bonds})
