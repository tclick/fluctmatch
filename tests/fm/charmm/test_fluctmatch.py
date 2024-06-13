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
"""Tests for fluctuation matching using CHARMM."""

import shutil
import subprocess
from collections import OrderedDict
from pathlib import Path
from subprocess import CalledProcessError
from typing import Self

import MDAnalysis as mda
import numpy as np
import pytest
from fluctmatch.fm.charmm.fluctmatch import CharmmFluctuationMatching
from testfixtures import Replacer, ShouldRaise
from testfixtures.mock import Mock

from tests.datafile import DCD_CG, IC_FLUCT, PRM, PSF_ENM


@pytest.fixture(scope="class")
def universe() -> mda.Universe:
    """Universe of an elastic network model.

    Returns
    -------
    MDAnalysis.Universe
        Elastic network model
    """
    return mda.Universe(PSF_ENM, DCD_CG)


@pytest.fixture()
def fluctmatch(universe: mda.Universe, tmp_path: Path) -> CharmmFluctuationMatching:
    """Object for fluctuation matching using CHARMM.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe of an elastic network model
    tmp_path : Path
        Temporary path

    Returns
    -------
    CharmmFluctuationMatching
        Object for fluctuation matching using CHARMM
    """
    return CharmmFluctuationMatching(universe, output_dir=tmp_path, prefix="fluctmatch")


class TestCharmmFluctuationMatching:
    """Tests for CharmmFluctuationMatching."""

    def test_invalid_temperature(self: Self, universe: mda.Universe, tmp_path: Path) -> None:
        """Construct an object with an invalid temperature.

        GIVEN an invalid temperature
        WHEN constructing the object
        THEN an exception is raised.

        Parameters
        ----------
        universe : MDAnalysis.Universe
            Universe of an elastic network model
        tmp_path : Path
            Temporary path
        """
        with ShouldRaise(ValueError):
            CharmmFluctuationMatching(universe, output_dir=tmp_path, prefix="fluctmatch", temperature=0)

    @pytest.mark.slow()
    def test_initialize(self: Self, fluctmatch: CharmmFluctuationMatching) -> None:
        """Test initialization.

        GIVEN a CharmmFluctuationMatching object
        WHEN the object is initialized
        THEN check the object is initialized.

        Parameters
        ----------
        fluctmatch : CharmmFluctuationMatching
            Object for fluctuation matching using CHARMM
        """
        stem = fluctmatch._output_dir.joinpath(fluctmatch._prefix)
        fm = fluctmatch.initialize()
        assert len(fm._parameters.parameters.bond_types) > 0, "Bond data not found."
        assert stem.with_suffix(".str").exists(), "Parameter file not found."
        assert stem.with_suffix(".inp").exists(), "CHARMM input file not found."

    @pytest.mark.slow()
    def test_simulate(self: Self, fluctmatch: CharmmFluctuationMatching) -> None:
        """Test simulation.

        GIVEN a CharmmFluctuationMatching object
        WHEN  the object is initialized and `simulate` is called
        THEN a subprocess call to `charmm` is called.

        Parameters
        ----------
        fluctmatch : CharmmFluctuationMatching
            Object for fluctuation matching using CHARMM
        """
        shutil.copy(IC_FLUCT, fluctmatch._output_dir.joinpath(fluctmatch._prefix).with_suffix(".ic"))
        with Replacer() as replace:
            mock_path = replace("pathlib.Path.exists", Mock(autospec=Path))
            mock_path.return_value = True
            mock_result = Mock(spec=subprocess.CompletedProcess, autospec=True)
            mock_run = replace("subprocess.run", mock_result)
            fluctmatch.initialize().simulate(executable=Path("charmm"))
            mock_run.assert_called_once()

    @pytest.mark.slow()
    def test_simulate_no_executable(self: Self, fluctmatch: CharmmFluctuationMatching) -> None:
        """Test simulation with no executable file.

        GIVEN a CharmmFluctuationMatching object
        WHEN  the object is initialized and `simulate` is called
        THEN an exception is raised when no executable is found.

        Parameters
        ----------
        fluctmatch : CharmmFluctuationMatching
            Object for fluctuation matching using CHARMM
        """
        with Replacer() as replace:
            mock_path = replace("pathlib.Path.exists", Mock(autospec=Path))
            mock_path.return_value = False
            mock_result = Mock(spec=subprocess.CompletedProcess, autospec=True)
            replace("subprocess.run", mock_result)
            with ShouldRaise(FileNotFoundError):
                fluctmatch.initialize().simulate(executable=Path("charmm"))

    @pytest.mark.slow()
    def test_simulate_fail(self: Self, fluctmatch: CharmmFluctuationMatching) -> None:
        """Test simulation failure if executable fails during run.

        GIVEN a CharmmFluctuationMatching object
        WHEN  the object is initialized and `simulate` is called
        THEN an exception is raised when executable fails during run.

        Parameters
        ----------
        fluctmatch : CharmmFluctuationMatching
            Object for fluctuation matching using CHARMM
        """
        with Replacer() as replace:
            mock_path = replace("pathlib.Path.exists", Mock(autospec=Path))
            mock_path.return_value = True
            mock_result = Mock(spec=subprocess.CompletedProcess, autospec=True)
            mock_result.side_effect = CalledProcessError(returncode=1, cmd="charmm")
            replace("subprocess.run", mock_result)
            with ShouldRaise(CalledProcessError):
                fluctmatch.initialize().simulate(executable=Path("charmm"))

    def test_load_target(self: Self, fluctmatch: CharmmFluctuationMatching) -> None:
        """Test loading a file containing target bond fluctuations.

        GIVEN a CharmmFluctuationMatching object
        WHEN `load_target` is called with an internal coordinate file
        THEN a file is read and the information loaded.

        Parameters
        ----------
        fluctmatch : CharmmFluctuationMatching
            Object for fluctuation matching using CHARMM
        """
        fm = fluctmatch.load_target(IC_FLUCT)
        assert len(fm._target.table) > 0

    @pytest.mark.slow()
    def test_load_parameters(self: Self, fluctmatch: CharmmFluctuationMatching) -> None:
        """Test loading a parameter file.

        GIVEN a CharmmFluctuationMatching object
        WHEN `load_parameter` is called with a CHARMM parameter file
        THEN a file is read and the information loaded.

        Parameters
        ----------
        fluctmatch : CharmmFluctuationMatching
            Object for fluctuation matching using CHARMM
        """
        fm = fluctmatch.load_parameters(PRM)
        assert len(fm._parameters.parameters.bond_types) > 0
        assert len(fm._param_ddist.parameters.bond_types) > 0

    @pytest.mark.slow()
    def test_calculate(self: Self, fluctmatch: CharmmFluctuationMatching) -> None:
        """Test calculations from fluctuation matching to get updated force constants.

        GIVEN a CharmmFluctuationMatching object
        WHEN  the object is initialized and `calculate` is called
        THEN a new set of force constants are calculated.

        Parameters
        ----------
        fluctmatch : CharmmFluctuationMatching
            Object for fluctuation matching using CHARMM
        """
        rng = np.random.default_rng()
        fluctmatch._parameters.read(PRM)
        forces = fluctmatch._parameters.forces.copy()
        fluctmatch._target.read(IC_FLUCT)
        fluctmatch._optimized.data.update(fluctmatch._target.data)
        fluctmatch._optimized.data = OrderedDict({key: rng.random() for key in fluctmatch._optimized.data})
        error = fluctmatch.calculate()

        assert set(fluctmatch._parameters.forces.values()) != set(forces.values()), "Calculation didn't work"
        assert error > 0.0
