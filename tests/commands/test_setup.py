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
# pyright: reportArgumentType = false
"""Test for mdsetup.commands.cmd_setup subcommand."""

import os
from typing import Self

import MDAnalysis as mda
import pytest
from click.testing import CliRunner
from fluctmatch.cli import main
from MDAnalysisTests.datafiles import DCD2, PSF
from testfixtures import ShouldRaise, TempDirectory


class TestSetup:
    """Run test for setup subcommand."""

    @pytest.fixture(scope="class")
    def cli_runner(self: Self) -> CliRunner:
        """Fixture for CLI runner.

        Returns
        -------
        CliRunner
            Click CLI runner
        """
        return CliRunner()

    def test_help(self: Self, cli_runner: CliRunner) -> None:
        """Test help output.

        GIVEN the init subcommand
        WHEN the help option is invoked
        THEN the help output should be displayed
        """
        result = cli_runner.invoke(main, "setup -h")

        assert "Usage:" in result.output
        assert result.exit_code == os.EX_OK

    @pytest.mark.parametrize("winsize", [2, 6, 17, 102])
    def test_setup(self: Self, cli_runner: CliRunner, winsize: int) -> None:
        """Test subcommand in an isolated filesystem.

        GIVEN an output subdirectory
        WHEN invoking the setup subcommand
        THEN the subcommand will complete successfully.
        """
        n_frames: int = mda.Universe(PSF, DCD2).trajectory.n_frames
        half_size = winsize // 2
        total_windows = (n_frames // half_size) - 1

        with TempDirectory(create=True) as tempdir:
            outdir = tempdir.as_path("test")
            log_file = tempdir.as_path("setup.log")
            json_file = log_file.with_suffix(".json")

            result = cli_runner.invoke(
                main,
                f"setup -s {PSF} -f {DCD2} -d {outdir} -o {json_file}  -w {winsize} -l {log_file}",
                catch_exceptions=False,
            )
            n_subdirs = len([_ for _ in outdir.glob("*") if _.is_dir()])

            assert result.exit_code == os.EX_OK
            assert log_file.is_file()
            assert json_file.is_file()
            assert (
                n_subdirs == total_windows
            ), f"{n_subdirs} created subdirectories not equal to expected {total_windows} subdirectories"

    def test_wrong_winsize(self: Self, cli_runner: CliRunner) -> None:
        """Test subcommand in an isolated filesystem.

        GIVEN a large window size
        WHEN invoking the setup subcommand
        THEN an exception is raised
        """
        n_frames: int = mda.Universe(PSF, DCD2).trajectory.n_frames * 2
        with TempDirectory() as tempdir:
            outdir = tempdir.as_path("test")
            log_file = tempdir.as_path("setup.log")
            json_file = log_file.with_suffix(".json")

            with ShouldRaise(ValueError):
                cli_runner.invoke(
                    main,
                    f"setup -s {PSF} -f {DCD2} -d {outdir} -o {json_file}  -w {n_frames} -l {log_file}",
                    catch_exceptions=False,
                )
