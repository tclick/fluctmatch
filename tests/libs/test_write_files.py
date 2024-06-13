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
