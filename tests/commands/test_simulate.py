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
"""Tests for simulate command."""

import os
from typing import Self

import numpy as np
import pytest
from click.testing import CliRunner
from fluctmatch.cli import main
from testfixtures import Replacer, TempDirectory
from testfixtures.mock import Mock

from tests.datafile import DCD_CG, IC_FLUCT, PSF_ENM, STR


class TestSimulate:
    """Tests for simulate command."""

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

        GIVEN the simulate subcommand
        WHEN the help option is invoked
        THEN the help output should be displayed

        Parameters
        ----------
        cli_runner : CliRunner
            Command-line cli_runner
        """
        result = cli_runner.invoke(main, "simulate -h")

        assert "Usage:" in result.output
        assert result.exit_code == os.EX_OK

    def test_simulate(self: Self, cli_runner: CliRunner) -> None:
        """Test simulate command.

        GIVEN the simulate subcommand
        WHEN additional arguments are provided
        THEN the simulation should run.

        Parameters
        ----------
        cli_runner : CliRunner
            Command-line cli_runner
        """
        with TempDirectory(create=True) as tempdir, Replacer() as replace:
            tmp_path = tempdir.as_path()
            log_file = tmp_path.joinpath("simulate.log")
            rmse_file = tmp_path.joinpath("rmse.csv")
            mock_simulate = replace("fluctmatch.fm.charmm.fluctmatch.CharmmFluctuationMatching.simulate", Mock())
            mock_calculate = replace("fluctmatch.fm.charmm.fluctmatch.CharmmFluctuationMatching.calculate", Mock())
            mock_calculate.return_value = np.random.default_rng().random(dtype=np.float64)

            result = cli_runner.invoke(
                main,
                f"simulate -s {PSF_ENM} -f {DCD_CG} -o {tmp_path}  -l {log_file} --target {IC_FLUCT} --param {STR} --max 5",
                catch_exceptions=False,
            )

            print(result.output)
            assert result.exit_code == os.EX_OK
            assert log_file.exists()
            assert log_file.stat().st_size > 0
            assert rmse_file.exists()
            assert rmse_file.stat().st_size > 0
            mock_simulate.assert_called()
            mock_calculate.assert_called()
