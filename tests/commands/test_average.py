# ---------------------------------------------------------------------------------------------------------------------
#  fluctmatch
#  Copyright (c) 2013-2024 Timothy H. Click, Ph.D.
#
#  This file is part of fluctmatch.
#
#  Fluctmatch is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
#  License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
#  later version.
#
#  Fluctmatch is distributed in the hope that it will be useful, # but WITHOUT ANY WARRANTY; without even the implied
#  warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with this program.
#  If not, see <[1](https://www.gnu.org/licenses/)>.
#
#  Reference:
#  Timothy H. Click, Nixon Raj, and Jhih-Wei Chu. Simulation. Meth Enzymology. 578 (2016), 327-342,
#  Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics doi:10.1016/bs.mie.2016.05.024.
# ---------------------------------------------------------------------------------------------------------------------
"""Tests whether an average structure can be determined from a trajectory."""

import os
import shutil
from typing import Self

import MDAnalysis as mda
import pytest
from click.testing import CliRunner
from fluctmatch.cli import main
from MDAnalysisTests.datafiles import DCD2, PSF
from testfixtures import TempDirectory


@pytest.fixture(scope="class")
def cli_runner() -> CliRunner:
    """Return CLI runner."""
    return CliRunner()


@pytest.fixture()
def universe() -> mda.Universe:
    """Return an all-atom universe."""
    return mda.Universe(PSF, DCD2)


class TestAverage:
    """Run tests for `average` subcommand."""

    def test_help(self: Self, cli_runner: CliRunner) -> None:
        """Test help output.

        GIVEN the init subcommand
        WHEN the help option is invoked
        THEN the help output should be displayed
        """
        result = cli_runner.invoke(main, "average -h")

        assert "Usage:" in result.output
        assert result.exit_code == os.EX_OK

    def test_single_average(self: Self, cli_runner: CliRunner) -> None:
        """Test whether average coordinates can be saved from a single trajectory.

        GIVEN the topology and trajectory files
        WHEN the `average` subcommand is invoked with a coordinae filename
        THEN an average structure should be written.
        """
        with TempDirectory() as tempdir:
            crdout = tempdir.as_path("average.crd")
            log_file = tempdir.as_path("average.log")

            result = cli_runner.invoke(
                main,
                f"average -s {PSF} -f {DCD2} -o {crdout} -l {log_file}",
                catch_exceptions=False,
            )

            # Check outcome of CLI including file creation.
            assert result.exit_code == os.EX_OK
            assert log_file.is_file(), f"{log_file} does not exist"
            assert crdout.is_file(), f"{crdout} does not exist"

    def test_multiple_averages(self: Self, cli_runner: CliRunner) -> None:
        """Test whether average coordinates can be saved from multiple trajectories.

        GIVEN the topology and trajectory files with a parent directory
        WHEN the `average` subcommand is invoked with a coordinae filename
        THEN an average structure should be written within each subdirectory.
        """
        n_dirs = 2
        with TempDirectory() as tempdir:
            directory = tempdir.as_path()
            crdout = "average.crd"
            log_file = tempdir.as_path("average.log")
            for i in range(n_dirs):
                subdir = tempdir.as_path(f"{i}")
                subdir.mkdir(exist_ok=True)
                shutil.copy(PSF, subdir)
                shutil.copy(DCD2, subdir)

            result = cli_runner.invoke(
                main,
                f"average -s {PSF} -f {DCD2} -o {crdout} -d {directory} -l {log_file}",
                catch_exceptions=False,
            )

            # Check outcome of CLI including file creation.
            assert result.exit_code == os.EX_OK
            assert log_file.is_file(), f"{log_file} does not exist"
            assert subdir.joinpath(crdout).is_file(), f"{subdir.joinpath(crdout)} does not exist"

    def test_invalid_filename(self: Self, cli_runner: CliRunner) -> None:
        """Test whether an exception is raised if an invalid filename is provided.

        GIVEN an invalid filename
        WHEN the `average` subcommand is invoked with a coordinae filename
        THEN an exception is raised.
        """
        with TempDirectory() as tempdir:
            crdout = tempdir.as_path("average.crd")
            log_file = tempdir.as_path("average.log")
            trajectory = "cg.dcd"

            result = cli_runner.invoke(
                main,
                f"average -s {PSF} -f {trajectory} -o {crdout} -l {log_file}",
                catch_exceptions=False,
            )
            assert result.exit_code != os.EX_OK
