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
# pyright: reportArgumentType = false
"""Test for mdsetup.commands.cmd_setup subcommand."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Self
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from fluctmatch.cli import main
from MDAnalysisTests.datafiles import DCD, PSF


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

        Parameters
        ----------
        cli_runner : CliRunner
            Command-line cli_runner
        """
        result = cli_runner.invoke(main, "setup -h")

        assert "Usage:" in result.output
        assert result.exit_code == os.EX_OK

    @pytest.mark.parametrize("winsize", [1000, 2000, 10000])
    def test_setup(self: Self, cli_runner: CliRunner, winsize: int) -> None:
        """Test subcommand in an isolated filesystem.

        GIVEN an output subdirectory
        WHEN invoking the setup subcommand
        THEN the subcommand will complete successfully.

        Parameters
        ----------
        cli_runner : CliRunner
            CLI runner
        winsize : int
            window size
        """
        with (
            cli_runner.isolated_filesystem() as ifs,
            patch.object(Path, "mkdir") as path_mkdir,
            patch.object(json, "dump") as json_dump,
        ):
            tmp_path = Path(ifs)
            outdir = tmp_path / "test"
            log_file = outdir / "setup.log"
            json_file = log_file.with_suffix(".json")

            result = cli_runner.invoke(
                main, f"setup -s {PSF} -f {DCD} -o {outdir} --json {json_file}  -w {winsize} -l {log_file}"
            )

            json_dump.assert_called()
            path_mkdir.assert_called()
            assert result.exit_code == os.EX_OK

    def test_wrong_winsize(self: Self, cli_runner: CliRunner) -> None:
        """Test subcommand in an isolated filesystem.

        GIVEN a large window size
        WHEN invoking the setup subcommand
        THEN an exception is raised

        Parameters
        ----------
        cli_runner : CliRunner
            CLI runner
        """
        with cli_runner.isolated_filesystem() as ifs:
            tmp_path = Path(ifs)
            outdir = tmp_path / "test"
            log_file = outdir / "setup.log"
            json_file = log_file.with_suffix(".json")

            with pytest.raises(ValueError):
                cli_runner.invoke(
                    main,
                    f"setup -s {PSF} -f {DCD} -o {outdir} --json {json_file}  -w 20000 -l {log_file}",
                    catch_exceptions=False,
                )
