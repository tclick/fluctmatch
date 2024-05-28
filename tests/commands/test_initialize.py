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
from pathlib import Path
from typing import Self
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from fluctmatch.cli import main

from tests.datafile import FLUCTDCD, FLUCTPSF


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

    def test_initiailization(self: Self, cli_runner: CliRunner) -> None:
        """Test whether parameter, topology and stream files are written.

        GIVEN a coarse-grain model
        WHEN given an output directory and filename prefix
        THEN a CHARMM topology, parameter, and stream file are written

        Parameters
        ----------
        cli_runner : CliRunner
            Command-line cli_runner
        """
        with (
            cli_runner.isolated_filesystem() as ifs,
            patch("fluctmatch.libs.write_files.write_parameters") as wp,
            patch("fluctmatch.libs.write_files.write_intcor") as wi,
            patch("fluctmatch.libs.write_files.write_stream") as ws,
        ):
            tmp_path = Path(ifs)
            prefix = "cg"
            log_file = tmp_path / "initialize.log"

            result = cli_runner.invoke(
                main, f"initialize -s {FLUCTPSF} -f {FLUCTDCD} -l {log_file} -d {tmp_path} -p {prefix}"
            )

            assert result.exit_code == os.EX_OK
            wp.assert_called_once()
            wp.assert_awaited()
            ws.assert_called_once()
            ws.assert_awaited()
            wi.assert_called()
            wi.assert_awaited()
