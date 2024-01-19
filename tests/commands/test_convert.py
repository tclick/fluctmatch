# ------------------------------------------------------------------------------
#  fluctmatch
#  Copyright (c) 2023 Timothy H. Click, Ph.D.
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
"""Test for mdsetup.commands.cmd_convert subcommand."""
import multiprocessing.pool
import os
from pathlib import Path
from typing import TypeVar
from unittest.mock import patch

from click.testing import CliRunner
from fluctmatch.commands.cmd_convert import cli

from ..datafile import CSV, TPR, XTC

TTestConvert = TypeVar("TTestConvert", bound="TestConvert")


class TestConvert:
    """Run test for setup subcommand."""

    def test_help(self: TTestConvert, cli_runner: CliRunner) -> None:
        """Test help output.

        GIVEN the convert subcommand
        WHEN the help option is invoked
        THEN the help output should be displayed

        Parameters
        ----------
        cli_runner : CliRunner
            Command-line cli_runner
        """
        result = cli_runner.invoke(cli, ["-h"])

        assert "Usage:" in result.output
        assert result.exit_code == os.EX_OK

    def test_commands(self: TTestConvert, cli_runner) -> None:
        """Test whether multiprocess is called.

        GIVEN an all-atom universe
        WHEN the `convert` subcommand is called and the multiprocess Pool is mocked
        THEN the command runs with an indication of the call
        """
        with cli_runner.isolated_filesystem() as ifs, patch.object(multiprocessing.pool.Pool, "imap_unordered") as imap:
            tmp_path = Path(ifs)
            logfile = tmp_path / "convert.log"
            outdir = tmp_path / "test"

            cli_runner.invoke(
                cli,
                [
                    "-s",
                    TPR,
                    "-f",
                    XTC,
                    "-o",
                    outdir.as_posix(),
                    "-l",
                    logfile.as_posix(),
                    "-f",
                    CSV,
                    "-v",
                    "DEBUG",
                ],
            )

        assert imap.assert_called()
        assert logfile.exists()
