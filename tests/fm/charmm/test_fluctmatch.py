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
"""Tests for fluctuation matching using CHARMM."""

import subprocess
from pathlib import Path
from subprocess import CalledProcessError
from typing import Self

import MDAnalysis as mda
import pytest
from fluctmatch.fm.charmm.fluctmatch import CharmmFluctuationMatching
from testfixtures.mock import Mock
from testfixtures.replace import Replacer

from tests.datafile import DCD_CG, PSF_ENM


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
        with Replacer() as replace:
            mock_path = replace("pathlib.Path.exists", Mock(autospec=Path))
            mock_path.return_value = True
            mock_result = Mock(spec=subprocess.CompletedProcess, autospec=True)
            mock_run = replace("subprocess.run", mock_result)
            fluctmatch.initialize().simulate(executable=Path("charmm"))
            mock_run.assert_called_once()

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
            with pytest.raises(FileNotFoundError):
                fluctmatch.initialize().simulate(executable=Path("charmm"))

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
            with pytest.raises(CalledProcessError):
                fluctmatch.initialize().simulate(executable=Path("charmm"))
