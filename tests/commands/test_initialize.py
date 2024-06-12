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
"""Tests for initialization before fluctuation matching."""

from __future__ import annotations

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

        Parameters
        ----------
        cli_runner : CliRunner
            Command-line cli_runner
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

        Parameters
        ----------
        cli_runner : CliRunner
            Command-line cli_runner
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
