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
"""Tests for initialization before fluctuation matching."""

import os
from typing import Self

import pytest
from click.testing import CliRunner
from fluctmatch.cli import main
from testfixtures import Replacer, TempDirectory
from testfixtures.mock import Mock

from tests.datafile import DCD_CG, PSF_ENM


class TestInitialize:
    """Test file initialization subcommand."""

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
        result = cli_runner.invoke(main, "initialize -h")

        assert "Usage:" in result.output
        assert result.exit_code == os.EX_OK

    @pytest.mark.slow()
    def test_initialization(self: Self, cli_runner: CliRunner) -> None:
        """Test whether parameter, topology and stream files are written.

        GIVEN a coarse-grain model
        WHEN given an output directory and filename prefix
        THEN a CHARMM topology, parameter, and stream file are written
        """
        with TempDirectory() as tmpdir, Replacer() as replace:
            tmp_path = tmpdir.as_path()
            prefix = "fluctmatch"
            log_file = tmp_path.joinpath("initialize.log")
            inp_file = tmp_path.joinpath(f"{prefix}.inp")

            mock_param_init = replace("fluctmatch.io.charmm.parameter.CharmmParameter.initialize", Mock())
            mock_intcor_init = replace("fluctmatch.io.charmm.intcor.CharmmInternalCoordinates.initialize", Mock())
            mock_stream_init = replace("fluctmatch.io.charmm.stream.CharmmStream.initialize", Mock())
            result = cli_runner.invoke(
                main, f"initialize -s {PSF_ENM} -f {DCD_CG} -l {log_file} -d {tmp_path} -p {prefix}"
            )

            assert result.exit_code == os.EX_OK
            assert log_file.exists()
            assert inp_file.exists()
            assert inp_file.stat().st_size > 0
            mock_param_init.assert_called_once()
            mock_stream_init.assert_called_once()
            mock_intcor_init.assert_called()
