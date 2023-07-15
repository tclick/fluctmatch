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
"""Test for mdsetup.commands.cmd_align subcommand."""
from __future__ import annotations

import os
from pathlib import Path

from click.testing import CliRunner
from fluctmatch.commands.cmd_align import cli

from ..datafile import TPR, XTC


class TestAlign:
    """Run test for align subcommand."""

    def test_help(self: TestAlign, cli_runner: CliRunner) -> None:
        """Test help output.

        GIVEN the align subcommand
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

    def test_align(self, cli_runner: CliRunner) -> None:
        """Test the alignment of a trajectory.

        GIVEN the align subcommand
        WHEN a topology and trajectory file are provided
        THEN a new trajectory file should be written

        Parameters
        ----------
        cli_runner : CliRunner
            Command-line cli_runner
        """
        with cli_runner.isolated_filesystem() as ifs:
            tmp_path = Path(ifs)
            logfile = tmp_path / "align.log"
            align = tmp_path / ("rmsfit_" + Path(XTC).name)

            result = cli_runner.invoke(
                cli, ["-s", TPR, "-f", XTC, "-o", ifs, "-l", logfile.as_posix(), "-t", "backbone", "--mass"]
            )

            assert result.exit_code == os.EX_OK
            assert logfile.exists()
            assert logfile.stat().st_size > 0
            assert align.exists()
            assert align.stat().st_size > 0
