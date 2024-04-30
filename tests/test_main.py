# ------------------------------------------------------------------------------
#  flucdtmatch
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
"""Test cases for the __main__ module."""
import os

import pytest
from click.testing import CliRunner
from fluctmatch.cli import main


class TestMain:
    """Run test for main command."""

    @pytest.fixture
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
