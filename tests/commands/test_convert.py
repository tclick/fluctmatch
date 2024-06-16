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
from typing import Self

import MDAnalysis as mda
import pytest
from click.testing import CliRunner
from fluctmatch.cli import main
from MDAnalysisTests.datafiles import DCD2, PSF
from testfixtures import TempDirectory


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
        """
        result = cli_runner.invoke(main, "convert -h")

        assert "Usage:" in result.output
        assert result.exit_code == os.EX_OK

    def test_list(self: Self, cli_runner: CliRunner) -> None:
        """Test list model.

        GIVEN the convert subcommand
        WHEN the list option is invoked
        THEN the model list should be displayed
        """
        result = cli_runner.invoke(main, "convert --list")
        assert "calpha:" in result.output
        assert result.exit_code == os.EX_OK

    @pytest.mark.slow()
    @pytest.mark.parametrize("model", "calpha caside ncsc polar".split())
    def test_conversion(self: Self, cli_runner: CliRunner, model: str) -> None:
        """Test whether an all-atom topology is converted to a coarse-grain model.

        GIVEN an all-atom model
        WHEN flagged to convert to convert to a C-alpha model
        THEN a CHARMM PSF, CRD, and DCD file are written
        """
        with TempDirectory() as tempdir:
            tmp_path = tempdir.as_path()
            prefix = "cg"
            log_file = tempdir.as_path("convert.log")
            traj_file = tmp_path.joinpath(f"{prefix}").with_suffix(".dcd")
            psf_file = tmp_path.joinpath(f"{prefix}").with_suffix(".psf")
            crd_file = tmp_path.joinpath(f"{prefix}").with_suffix(".crd")

            result = cli_runner.invoke(
                main,
                f"convert -s {PSF} -f {DCD2} -l {log_file} -d {tmp_path} -p {prefix} -m {model} --guess",
                catch_exceptions=False,
            )

            assert result.exit_code == os.EX_OK
            assert psf_file.is_file()
            assert crd_file.is_file()
            assert not traj_file.is_file()

    @pytest.mark.slow()
    @pytest.mark.parametrize("model", "calpha caside ncsc polar".split())
    def test_conversion_with_trajectory(self: Self, cli_runner: CliRunner, model: str) -> None:
        """Test whether an all-atom topology and trajectory is converted to a coarse-grain model.

        GIVEN an all-atom model
        WHEN flagged to convert to convert to a C-alpha model
        THEN a CHARMM PSF, CRD, and DCD file are written
        """
        with TempDirectory() as tempdir:
            tmp_path = tempdir.as_path()
            prefix = "cg"
            log_file = tempdir.as_path("convert.log")
            traj_file = tmp_path.joinpath(f"{prefix}").with_suffix(".dcd")
            psf_file = tmp_path.joinpath(f"{prefix}").with_suffix(".psf")
            crd_file = tmp_path.joinpath(f"{prefix}").with_suffix(".crd")

            result = cli_runner.invoke(
                main,
                f"convert -s {PSF} -f {DCD2} -l {log_file} -d {tmp_path} -p {prefix} -m {model} --guess --write",
                catch_exceptions=False,
            )

            assert result.exit_code == os.EX_OK
            assert psf_file.is_file()
            assert crd_file.is_file()
            assert traj_file.is_file()
