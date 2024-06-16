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
        """
        result = cli_runner.invoke(main, "simulate -h")

        assert "Usage:" in result.output
        assert result.exit_code == os.EX_OK

    def test_simulate(self: Self, cli_runner: CliRunner) -> None:
        """Test simulate command.

        GIVEN the simulate subcommand
        WHEN additional arguments are provided
        THEN the simulation should run.
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
                f"simulate -s {PSF_ENM} -f {DCD_CG} -d {tmp_path}  -l {log_file} --target {IC_FLUCT} --param {STR} --max 5",
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
