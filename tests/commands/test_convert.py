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
"""Tests for conversion of molecular dynamics trajectories to a coarse-grain model."""

import os
from pathlib import Path
from typing import Self

import MDAnalysis as mda
import pytest
from click.testing import CliRunner
from fluctmatch.cli import main
from MDAnalysisTests.datafiles import DCD2, PSF
from testfixtures import Replacer
from testfixtures.mock import Mock


@pytest.fixture(scope="class")
def cli_runner() -> CliRunner:
    """Fixture for CLI runner.

    Returns
    -------
    CliRunner
        Click CLI runner
    """
    return CliRunner()


@pytest.fixture(scope="class")
def universe() -> mda.Universe:
    """Return all-atom universe."""
    return mda.Universe(PSF, DCD2)


class TestConvert:
    """Run test for convert subcommand."""

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
        with cli_runner.isolated_filesystem() as ifs, Replacer() as replace:
            replace("fluctmatch.model.base.CoarseGrainModel.transform", lambda self: universe)  # noqa: ARG005
            mock_write = replace("MDAnalysis.coordinates.base.WriterBase.write", Mock())
            mock_save = replace("parmed.structure.Structure.save", Mock())
            tmp_path = Path(ifs)
            prefix = "cg"
            log_file = tmp_path.joinpath("convert.log")

            result = cli_runner.invoke(
                main, f"convert -s {PSF} -f {DCD2} -l {log_file} -o {tmp_path} -p {prefix} -m calpha --guess --write"
            )

            assert result.exit_code == os.EX_OK
            mock_write.assert_called()
            mock_save.assert_called()
