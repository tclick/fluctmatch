# ---------------------------------------------------------------------------------------------------------------------
#  fluctmatch
#  Copyright (c) 2013-2024 Timothy H. Click, Ph.D.
#
#  This file is part of fluctmatch.
#
#  Fluctmatch is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
#  License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
#  later version.
#
#  Fluctmatch is distributed in the hope that it will be useful, # but WITHOUT ANY WARRANTY; without even the implied
#  warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with this program.
#  If not, see <[1](https://www.gnu.org/licenses/)>.
#
#  Reference:
#  Timothy H. Click, Nixon Raj, and Jhih-Wei Chu. Simulation. Meth Enzymology. 578 (2016), 327-342,
#  Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics doi:10.1016/bs.mie.2016.05.024.
# ---------------------------------------------------------------------------------------------------------------------
"""Test the combination of multiple parameter files."""

import os
import shutil
from typing import Self

import MDAnalysis as mda
import pandas as pd
import pytest
from click.testing import CliRunner
from fluctmatch.cli import main
from fluctmatch.commands import FILE_MODE
from testfixtures import ShouldRaise, TempDirectory

from tests.datafile import DCD_CG, IC_FLUCT, PRM, PSF_ENM


@pytest.fixture()
def cli_runner() -> CliRunner:
    """Fixture for CLI runner.

    Returns
    -------
    CliRunner
        Click CLI runner
    """
    return CliRunner()


class TestCombine:
    """Test `combine` subcommand.

    Methods
    -------
    test_help()
        Test whether the help message is displayed
    test_combine()
        Test whether the `combine` subcommand is executed successfully
    """

    def test_help(self: Self, cli_runner: CliRunner) -> None:
        """Test the --help option of the CLI using CliRunner.

        GIVEN the `combine` subcommand
        WHEN the help option is invoked
        THEN the help output should be displayed
        """
        result = cli_runner.invoke(main, "combine -h")

        assert result.exit_code == os.EX_OK
        assert "Usage: fluctmatch combine [OPTIONS]" in result.output
        assert "-o, --output FILE" in result.output
        assert "--filter / --no-filter" in result.output

    @pytest.mark.slow()
    def test_combine_all_only(self: Self, cli_runner: CliRunner) -> None:
        """Test that the `combine` subcommand is executed successfully.

        GIVEN the `combine` subcommand
        WHEN the parent directory and parameter file are provided
        THEN a CSV file with the force constants and a log file will be created
        """
        universe = mda.Universe(PSF_ENM, DCD_CG)
        n_bonds = len(universe.bonds)
        n_subdirs = 5
        subdirectories = (str(_) for _ in range(1, n_subdirs + 1))

        with TempDirectory() as tmp_path:
            output = tmp_path.as_path("combined_all.csv")
            print(output)
            log_file = tmp_path.as_path("combined.log")
            for directory in subdirectories:
                new_dir = tmp_path.as_path(directory)
                new_dir.mkdir(mode=FILE_MODE, exist_ok=True, parents=True)
                shutil.copy(PRM, new_dir)

            subcommand = f"combine -d {tmp_path.as_path()} -f {PRM} --ic {IC_FLUCT} -o {output} -l {log_file}"
            results = cli_runner.invoke(main, subcommand)
            assert results.exit_code == os.EX_OK, "'fluctmatch combine' failed to execute properly."

            # Test existence of CSV and log files.
            assert output.is_file(), "The CSV does not exist."
            assert log_file.is_file(), "The log file does not exist."

            # Compare shape of table with (n_bonds, n_subdirs)
            table = pd.read_csv(output, header=0, index_col=tuple(range(6)))
            assert table.shape == (n_bonds, n_subdirs), "The shape of the table does not match the expected shape."

    def test_combine_all_with_filter(self: Self, cli_runner: CliRunner) -> None:
        """Test that the `combine` subcommand with '--filter' option.

        GIVEN the `combine` subcommand with the '--filter' option
        WHEN the parent directory and parameter file are provided
        THEN a RuntimeWarning should be raised.
        """
        n_subdirs = 5
        subdirectories = (str(_) for _ in range(1, n_subdirs + 1))

        with TempDirectory() as tmp_path, ShouldRaise(RuntimeWarning):
            output = tmp_path.as_path("combined_all.csv")
            log_file = tmp_path.as_path("combined.log")
            for directory in subdirectories:
                new_dir = tmp_path.as_path(directory)
                new_dir.mkdir(mode=FILE_MODE, exist_ok=True, parents=True)
                shutil.copy(PRM, new_dir)

            subcommand = f"combine -d {tmp_path.as_path()} -f {PRM} --ic {IC_FLUCT} -o {output} -l {log_file} --filter"
            cli_runner.invoke(main, subcommand, catch_exceptions=False)

    @pytest.mark.slow()
    def test_combine_resij(self: Self, cli_runner: CliRunner) -> None:
        """Test that an interresidue CSV file is written when '--resij' is specified.

        GIVEN the `combine` subcommand
        WHEN the '--resij' option is invoked
        THEN a CSV file with the force constants and a log file will be created
        """
        n_subdirs = 5
        subdirectories = (str(_) for _ in range(1, n_subdirs + 1))

        with TempDirectory() as tmp_path:
            output = tmp_path.as_path("combined_all.csv")
            resij = tmp_path.as_path("resij.csv")
            log_file = tmp_path.as_path("combined.log")
            for directory in subdirectories:
                new_dir = tmp_path.as_path(directory)
                new_dir.mkdir(mode=FILE_MODE, exist_ok=True, parents=True)
                shutil.copy(PRM, new_dir)

            subcommand = f"combine -d {tmp_path.as_path()} -f {PRM} --ic {IC_FLUCT} -o {output} --resij -l {log_file}"
            results = cli_runner.invoke(main, subcommand)
            assert results.exit_code == os.EX_OK, "'fluctmatch combine' failed to execute properly."

            # Test existence of CSV and log files.
            assert output.is_file(), f"{output} does not exist."
            assert resij.is_file(), f"{resij} does not exist."
            assert log_file.is_file(), "The log file does not exist."

    @pytest.mark.slow()
    def test_combine_resi(self: Self, cli_runner: CliRunner) -> None:
        """Test that a residue CSV file is written when '--resi' is specified.

        GIVEN the `combine` subcommand
        WHEN the '--resi' option is invoked
        THEN a CSV file with the force constants and a log file will be created
        """
        n_subdirs = 5
        subdirectories = (str(_) for _ in range(1, n_subdirs + 1))

        with TempDirectory() as tmp_path:
            output = tmp_path.as_path("combined_all.csv")
            resi = tmp_path.as_path("resi.csv")
            log_file = tmp_path.as_path("combined.log")
            for directory in subdirectories:
                new_dir = tmp_path.as_path(directory)
                new_dir.mkdir(mode=FILE_MODE, exist_ok=True, parents=True)
                shutil.copy(PRM, new_dir)

            subcommand = f"combine -d {tmp_path.as_path()} -f {PRM} --ic {IC_FLUCT} -o {output} --resi -l {log_file}"
            results = cli_runner.invoke(main, subcommand, catch_exceptions=False)
            assert results.exit_code == os.EX_OK, "'fluctmatch combine' failed to execute properly."

            # Test existence of CSV and log files.
            assert output.is_file(), f"{output} does not exist."
            assert resi.is_file(), f"{resi} does not exist."
            assert log_file.is_file(), "The log file does not exist."
