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
"""Tests for conversion of molecular dynamics trajectories to a coarse-grain model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Self
from unittest.mock import patch

import MDAnalysis as mda
import pytest
from click.testing import CliRunner
from fluctmatch.cli import main
from fluctmatch.model.base import CoarseGrainModel
from MDAnalysis.coordinates.base import WriterBase
from parmed import structure

from tests.datafile import TPR, XTC


class TestConvert:
    """Run test for convert subcommand."""

    @pytest.fixture(scope="class")
    def cli_runner(self: Self) -> CliRunner:
        """Fixture for CLI runner.

        Returns
        -------
        CliRunner
            Click CLI runner
        """
        return CliRunner()

    @pytest.fixture(scope="class")
    def universe(self: Self) -> mda.Universe:
        """Return all-atom universe."""
        return mda.Universe(TPR, XTC)

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
        result = cli_runner.invoke(main, "convert -h")

        assert "Usage:" in result.output
        assert result.exit_code == os.EX_OK

    def test_list(self: Self, cli_runner: CliRunner) -> None:
        """Test list model.

        GIVEN the convert subcommand
        WHEN the list option is invoked
        THEN the model list should be displayed

        Parameters
        ----------
        cli_runner : CliRunner
            Command-line cli_runner
        """
        result = cli_runner.invoke(main, "convert --list")
        assert "calpha:" in result.output
        assert result.exit_code == os.EX_OK

    def test_conversion(self: Self, cli_runner: CliRunner, universe: mda.Universe) -> None:
        """Test whether an all-atom trajectory is converted to a coarse-grain model.

        GIVEN an all-atom model
        WHEN flagged to convert to convert to a C-alpha model
        THEN a CHARMM PSF, CRD, and DCD file are written

        Parameters
        ----------
        cli_runner : CliRunner
            Command-line cli_runner
        """
        with (
            cli_runner.isolated_filesystem() as ifs,
            patch.object(CoarseGrainModel, "transform", return_value=universe) as cg,
            patch.object(WriterBase, "write") as writer,
            patch.object(structure.Structure, "save") as save,
        ):
            tmp_path = Path(ifs)
            prefix = "cg"
            log_file = tmp_path / "convert.log"

            result = cli_runner.invoke(
                main, f"convert -s {TPR} -f {XTC} -l {log_file} -o {tmp_path} -p {prefix} -m calpha --guess --write"
            )

            assert result.exit_code == os.EX_OK
            cg.assert_called()
            writer.assert_called()
            save.assert_called()
