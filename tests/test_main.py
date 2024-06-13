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
"""Test cases for the __main__ module."""

import os

import pytest
from click.testing import CliRunner
from fluctmatch.cli import main


class TestMain:
    """Run test for main command."""

    @pytest.fixture()
    def cli_runner(self) -> CliRunner:
        """Fixture for running the main command."""
        return CliRunner()

    def test_help(self, cli_runner: CliRunner) -> None:
        """Test help output.

        GIVEN the main command
        WHEN the help option is invoked
        THEN the help output should be displayed

        Parameters
        ----------
        runner : CliRunner
            Command-line runner
        """
        result = cli_runner.invoke(main, ["-h"])

        assert "Usage:" in result.output
        assert result.exit_code == os.EX_OK

    def test_main_succeeds(self, cli_runner: CliRunner) -> None:
        """Test main output.

        GIVEN the main command
        WHEN the help option is invoked
        THEN the help output should be displayed

        Parameters
        ----------
        runner : CliRunner
            Command-line runner
        """
        result = cli_runner.invoke(main)

        assert "Usage:" in result.output
        assert result.exit_code == os.EX_OK

    def test_main_fails(self, cli_runner: CliRunner) -> None:
        """Test main with invalid subcommand.

        GIVEN the main command
        WHEN an invalid subcommand is provided
        THEN an error will be issued.

        Parameters
        ----------
        cli_runner : CliRunner
            Command-line cli_runner
        """
        result = cli_runner.invoke(main, ["bad_subcommand"])

        assert "Error" in result.output
        assert result.exit_code != os.EX_OK
