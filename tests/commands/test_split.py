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
"""Tests for split command."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Self

import pytest
from click.testing import CliRunner
from fluctmatch.cli import main
from MDAnalysisTests.datafiles import DCD2, PSF
from testfixtures.mock import Mock
from testfixtures.replace import Replacer

from tests.datafile import JSON


class TestSplit:
    """Run test for split subcommand."""

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

        Parameters
        ----------
        cli_runner : CliRunner
            CLI runner
        """
        result = cli_runner.invoke(main, "split -h")

        assert "Usage:" in result.output
        assert result.exit_code == os.EX_OK

    def test_split(self: Self, cli_runner: CliRunner) -> None:
        """Test subcommand in an isolated filesystem.

        GIVEN an output subdirectory
        WHEN invoking the setup subcommand
        THEN the subcommand will complete successfully.


        Parameters
        ----------
        cli_runner : CliRunner
            CLI runner
        """
        with cli_runner.isolated_filesystem() as ifs, Replacer() as replace:
            tmp_path = Path(ifs)
            outdir = tmp_path.joinpath("test")
            log_file = outdir.joinpath("split.log")
            crd_file = outdir.joinpath(PSF).with_suffix(".crd")
            traj_file = outdir.joinpath(PSF).with_suffix(".dcd")
            mock_average = replace("fluctmatch.libs.write_files.write_average_structure", Mock())

            args = f"split -s {PSF} -f {DCD2} --json {JSON} -o {traj_file} -c {crd_file} -l {log_file} --average"
            result = cli_runner.invoke(main, args)

            assert result.exit_code == os.EX_OK
            assert log_file.exists()
            mock_average.assert_called()
