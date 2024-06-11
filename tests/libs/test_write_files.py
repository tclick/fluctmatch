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
"""Tests for write_files module."""

from pathlib import Path
from typing import Self

import MDAnalysis as mda
import numpy as np
import pytest
from fluctmatch.libs import write_files
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysisTests.datafiles import DCD2, PSF
from testfixtures import ShouldRaise

N_RESIDUES: int = 5
N_FRAMES: int = 5


@pytest.fixture(scope="class")
def universe() -> mda.Universe:
    """Return a universe with a trajectory.

    Returns
    -------
    MDAnalysis.Universe
        Universe with a trajectory.
    """
    u = mda.Universe(PSF, DCD2)
    new_u = mda.Merge(u.residues[:N_RESIDUES].atoms)
    positions = np.array([new_u.atoms.positions for _ in new_u.trajectory[:N_FRAMES]])
    return new_u.load_new(positions, format=MemoryReader, orient="fac")


class TestWriteFiles:
    """Tests for write_files module."""

    def test_write_trajectory(self: Self, universe: mda.Universe, tmp_path: Path) -> None:
        """Test writing a trajectory from an all-atom universe.

        GIVEN a universe and a tempory directory
        WHEN writing a trajectory from an all-atom universe
        THEN a trajectory file is written.

        Parameters
        ----------
        universe : MDAnalysis.Universe
            Universe with trajectory
        tmp_path : Path
            Temporary directory.
        """
        filename = tmp_path.joinpath("trajectory.pdb")
        write_files.write_trajectory(universe, filename=filename)
        assert filename.exists()
        assert filename.stat().st_size > 0

    def test_write_average_structure(self: Self, universe: mda.Universe, tmp_path: Path) -> None:
        """Test writing an average structure from an all-atom universe.

        GIVEN a universe and a tempory directory
        WHEN writing an average structure from an all-atom universe
        THEN a coordinate file is written.

        Parameters
        ----------
        universe : MDAnalysis.Universe
            Universe with trajectory
        tmp_path : Path
            Temporary directory.
        """
        filename = tmp_path.joinpath("trajectory.crd")
        write_files.write_average_structure(universe, filename=filename)
        assert filename.exists()
        assert filename.stat().st_size > 0

    @pytest.mark.parametrize("sim_type", ["fluctmatch", "thermodynamics"])
    def test_write_input(self: Self, tmp_path: Path, sim_type: str) -> None:
        """Test writing a CHARMM input file.

        GIVEN a tempory directory
        WHEN writing a CHARMM input file
        THEN a file is written.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory
        sim_type : str
            Simulation type
        """
        prefix = "4AKE"
        input_file = tmp_path.joinpath(sim_type).with_suffix(".inp")
        write_files.write_charmm_input(
            topology=PSF, trajectory=DCD2, directory=tmp_path, prefix=prefix, sim_type=sim_type
        )
        assert input_file.exists()
        assert input_file.stat().st_size > 0
        with input_file.open() as f:
            lines = f.readlines()
            assert "vibran nmode" in "".join(lines)

    def test_invalid_simtype(self: Self, tmp_path: Path) -> None:
        """Test writing a CHARMM input file given an invalid `sim_type` choice.

        GIVEN an invalid `sim_type` choice
        WHEN writing a CHARMM input file
        THEN an exception is raised.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory.
        """
        prefix = "4AKE"
        sim_type = "no_template"
        with ShouldRaise(FileNotFoundError):
            write_files.write_charmm_input(
                topology=PSF, trajectory=DCD2, directory=tmp_path, prefix=prefix, sim_type=sim_type
            )
