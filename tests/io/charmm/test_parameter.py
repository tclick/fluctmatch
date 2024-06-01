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
from fluctmatch.io.charmm.parameter import CharmmParameter
from numpy import testing
from pyfakefs import fake_file, fake_filesystem

from tests.datafile import FLUCTDCD, FLUCTPSF, PRM, STR


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
        return mda.Universe(FLUCTPSF, FLUCTDCD)

    @pytest.fixture(scope="class")
    def param_file(self: Self, fs_class: fake_filesystem.FakeFilesystem) -> fake_file.FakeFile:
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
        return fs_class.create_file("charmm.prm")

    def test_initialize(self: Self, universe: mda.Universe) -> None:
        """Test initialization of a parameter file.

        GIVEN an elastic network model, bond force constants, and bond lengths
        WHEN a parameter object is provided
        THEN a parameter object is initialized.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            Elastic network model
        """
        rng = np.random.default_rng()
        forces = rng.random(len(universe.bonds), dtype=universe.bonds.values().dtype)
        lengths = universe.bonds.values()
        atom_types = OrderedDict({atom.type: "" for atom in universe.atoms})
        param = CharmmParameter()
        param.initialize(universe, forces=forces, lengths=lengths)

        testing.assert_equal(param._parameters.atom_types.keys(), atom_types.keys())
        assert all(key in param._parameters.bond_types for key in universe.bonds.topDict)
        testing.assert_allclose(
            [at.mass for at in param._parameters.atom_types.values()], universe.atoms.masses, rtol=1e-4
        )
        testing.assert_allclose([bt.k for bt in param._parameters.bond_types.values()], forces, rtol=1e-4)
        testing.assert_allclose([bt.req for bt in param._parameters.bond_types.values()], lengths, rtol=1e-4)

    def test_initialize_unequal_size(self: Self, universe: mda.Universe) -> None:
        """Test initialization of a parameter file with unequal sizes of forces and bond lengths.

        GIVEN an elastic network model with bonds, bond force constants, and bond lengths
        WHEN the three variables are of unequal size
        THEN a ValueError is raised.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            Elastic network model
        """
        rng = np.random.default_rng()
        forces = rng.random(len(universe.bonds) - 1, dtype=universe.bonds.values().dtype)
        lengths = universe.bonds.values()[:-2]
        param = CharmmParameter()

        with pytest.raises(ValueError):
            param.initialize(universe, forces=forces, lengths=lengths)

    def test_initialize_no_bonds(self: Self) -> None:
        """Test initialization of a parameter file with no bond data.

        GIVEN an elastic network model with no bonds, bond force constants, and bond lengths
        WHEN the three variables are of unequal size
        THEN a parameter object is initialized.

        """
        universe = mda.Universe.empty(0)
        forces = np.zeros(5, dtype=float)
        param = CharmmParameter()

        with pytest.raises(mda.NoDataError):
            param.initialize(universe, forces=forces, lengths=forces)

    def test_write(self: Self, universe: mda.Universe, param_file: fake_file.FakeFile) -> None:
        """Test writing a parameter file.

        GIVEN a universe and a parameter file
        WHEN a parameter object is initialized
        THEN a parameter object is written.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            An elastic network model
        param_file : :class:`pyfakefs.fake_file.FakeFile
            Empty file in memory
        """
        filename = Path(param_file.name)
        rng = np.random.default_rng()
        forces = rng.random(len(universe.bonds), dtype=universe.bonds.values().dtype)
        lengths = universe.bonds.values()
        param = CharmmParameter()
        param.initialize(universe, forces=forces, lengths=lengths)

        param.write(par=param_file.name)
        assert filename.exists()
        assert filename.stat().st_size > 0

    def test_write_fail(self: Self, universe: mda.Universe) -> None:
        """Test writing a parameter file if no filenames are provided.

        GIVEN a universe and no parameter file
        WHEN a parameter object is initialized
        THEN a ValueError is raised.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            An elastic network model
        """
        rng = np.random.default_rng()
        forces = rng.random(len(universe.bonds), dtype=universe.bonds.values().dtype)
        lengths = universe.bonds.values()
        param = CharmmParameter()
        param.initialize(universe, forces=forces, lengths=lengths)

        with pytest.raises(ValueError):
            param.write()

    def test_read(self: Self) -> None:
        """Test reading a parameter file.

        GIVEN a parameter file
        WHEN a parameter object is initialized and read
        THEN a parameter file is read and loaded into the parameter object.
        """
        param = CharmmParameter()
        param.read(PRM)

        assert len(param._parameters.atom_types) > 0, "Atom type definitions don't exist."
        assert len(param._parameters.bond_types) > 0, "Bond definitions don't exist."

    def test_read_no_file(self: Self) -> None:
        """Test reading non-existent parameter file.

        GIVEN a non-existent parameter filename
        WHEN a parameter object is initialized and read
        THEN a FileNotFoundError is raised.
        """
        param_file = "charmm.str"
        param = CharmmParameter()
        with pytest.raises(FileNotFoundError):
            param.read(param_file)

    def test_read_empty_file(self: Self) -> None:
        """Test reading stream file without parameter information.

        GIVEN a stream file without parameter information
        WHEN a parameter object is initialized and read
        THEN an OSError is raised.
        """
        param = CharmmParameter()
        with pytest.raises(OSError):
            param.read(STR)

    def test_parameters_property(self: Self, universe: mda.Universe) -> None:
        """Test parameters property.

        GIVEN an elastic network model, equilibrium force constants, and bond lengths
        WHEN a parameter object is initialized and the parameters property is called
        THEN the underlying parameter object is returned.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            An elastic network model
        """
        # Initialize parameters
        rng = np.random.default_rng()
        forces = rng.random(len(universe.bonds), dtype=universe.bonds.values().dtype)
        lengths = universe.bonds.values()
        param = CharmmParameter()
        param.initialize(universe, forces=forces, lengths=lengths)

        # Test parameters getter
        parameters = param.parameters
        assert parameters.atom_types == param._parameters.atom_types
        assert parameters.bond_types == param._parameters.bond_types

    def test_forces_property(self: Self, universe: mda.Universe) -> None:
        """Test getter/setter of forces property.

        GIVEN an elastic network model, equilibrium force constants, and bond lengths
        WHEN a parameter object is initialized and the forces property is called
        THEN force constants are retrieved or set.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            An elastic network model
        """
        rng = np.random.default_rng()
        forces = rng.random(len(universe.bonds), dtype=universe.bonds.values().dtype)
        lengths = universe.bonds.values()
        param = CharmmParameter()
        param.initialize(universe, forces=forces, lengths=lengths)

        # Test getter
        data = param.forces
        testing.assert_allclose(data, forces, rtol=1e-05, atol=1e-08, err_msg="Forces don't match.", verbose=True)

        # Test setter
        param.forces = np.zeros_like(forces)
        testing.assert_allclose(param.forces, 0.0, rtol=1e-05, atol=1e-08, err_msg="Forces don't match.", verbose=True)

    def test_forces_property_fail(self: Self, universe: mda.Universe) -> None:
        """Test setter of forces property.

        GIVEN an elastic network model, equilibrium force constants, and bond lengths
        WHEN a parameter object is initialized and a shorter forces array is given
        THEN a ValueError is raised.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            An elastic network model
        """
        rng = np.random.default_rng()
        forces = rng.random(len(universe.bonds), dtype=universe.bonds.values().dtype)
        lengths = universe.bonds.values()
        param = CharmmParameter()
        param.initialize(universe, forces=forces, lengths=lengths)

        # Test setter
        with pytest.raises(ValueError):
            param.forces = np.zeros_like(forces)[:-1]

    def test_distances_property(self: Self, universe: mda.Universe) -> None:
        """Test getter/setter of distances property.

        GIVEN an elastic network model, equilibrium force constants, and bond lengths
        WHEN a parameter object is initialized and the distances property is called
        THEN equilibrium bond distances are retrieved or set.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            An elastic network model
        """
        rng = np.random.default_rng()
        forces = rng.random(len(universe.bonds), dtype=universe.bonds.values().dtype)
        lengths = universe.bonds.values()
        param = CharmmParameter()
        param.initialize(universe, forces=forces, lengths=lengths)

        # Test getter
        data = param.distances
        testing.assert_allclose(data, lengths, rtol=1e-05, atol=1e-08, err_msg="Distances don't match.", verbose=True)

        # Test setter
        param.distances = np.zeros_like(lengths)
        testing.assert_allclose(
            param.distances, 0.0, rtol=1e-05, atol=1e-08, err_msg="Distances don't match.", verbose=True
        )

    def test_distancess_property_fail(self: Self, universe: mda.Universe) -> None:
        """Test setter of distances property.

        GIVEN an elastic network model, equilibrium force constants, and bond lengths
        WHEN a parameter object is initialized and a shorter distances array is given
        THEN a ValueError is raised.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            An elastic network model
        """
        rng = np.random.default_rng()
        forces = rng.random(len(universe.bonds), dtype=universe.bonds.values().dtype)
        lengths = universe.bonds.values()
        param = CharmmParameter()
        param.initialize(universe, forces=forces, lengths=lengths)

        # Test setter
        with pytest.raises(ValueError):
            param.forces = np.zeros_like(lengths)[:-1]