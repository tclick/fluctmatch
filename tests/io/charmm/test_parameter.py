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

        with pytest.raises(ValueError, match="Bond force constants and bond distances do not match."):
            CharmmParameter().initialize(universe, forces=bonds, lengths=distances)

        with pytest.raises(ValueError, match="force constants and distances do not match bonds in universe."):
            CharmmParameter().initialize(universe, forces=distances, lengths=distances)

    def test_initialize_no_bonds(self: Self) -> None:
        """Test initialization of a parameter file with no bond data.

        GIVEN an elastic network model with no bonds, bond force constants, and bond lengths
        WHEN the three variables are of unequal size
        THEN a parameter object is initialized.

        """
        universe = mda.Universe.empty(0)
        forces = np.zeros(5, dtype=float)

        with pytest.raises(mda.NoDataError):
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

        with pytest.raises(TypeError):
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
        with pytest.raises(FileNotFoundError):
            CharmmParameter().read(param_file)

    def test_read_empty_file(self: Self) -> None:
        """Test reading stream file without parameter information.

        GIVEN a stream file without parameter information
        WHEN a parameter object is initialized and read
        THEN an OSError is raised.
        """
        with pytest.raises(OSError):
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
        with pytest.raises(ValueError):
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
        with pytest.raises(ValueError):
            param.distances = OrderedDict({k: 0.0 for k in bad_bonds})
