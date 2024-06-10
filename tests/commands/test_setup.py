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

import os
from pathlib import Path
from typing import Self

import MDAnalysis as mda
import pytest
from click.testing import CliRunner
from fluctmatch.cli import main
from MDAnalysisTests.datafiles import DCD2, PSF
from testfixtures import Replacer, ShouldRaise, TempDirectory
from testfixtures.mock import Mock


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

    @pytest.mark.parametrize("winsize", [2, 6, 17, 102])
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
        n_frames: int = mda.Universe(PSF, DCD2).trajectory.n_frames
        half_size = winsize // 2
        total_windows = (n_frames // half_size) - 1

        with TempDirectory(create=True) as ifs, Replacer() as replace:
            mock_dump = replace("json.dump", Mock())
            tmp_path = ifs.as_path()
            outdir = tmp_path.joinpath("test")
            log_file = outdir.joinpath("setup.log")
            json_file = log_file.with_suffix(".json")

            result = cli_runner.invoke(
                main,
                f"setup -s {PSF} -f {DCD2} -o {outdir} --json {json_file}  -w {winsize} -l {log_file}",
                catch_exceptions=False,
            )
            n_subdirs = len([_ for _ in outdir.glob("*") if _.is_dir()])

            mock_dump.assert_called()
            assert result.exit_code == os.EX_OK
            assert (
                n_subdirs == total_windows
            ), f"{n_subdirs} created subdirectories not equal to expected {total_windows} subdirectories"

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
        n_frames: int = mda.Universe(PSF, DCD2).trajectory.n_frames * 2
        with cli_runner.isolated_filesystem() as ifs:
            tmp_path = Path(ifs)
            outdir = tmp_path / "test"
            log_file = outdir / "setup.log"
            json_file = log_file.with_suffix(".json")

            with ShouldRaise(ValueError):
                cli_runner.invoke(
                    main,
                    f"setup -s {PSF} -f {DCD2} -o {outdir} --json {json_file}  -w {n_frames} -l {log_file}",
                    catch_exceptions=False,
                )
