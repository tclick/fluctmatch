# ---------------------------------------------------------------------------------------------------------------------
# fluctmatch
# Copyright (c) 2013-2024 Timothy H. Click, Ph.D.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
# later version.

# This program is distributed in the hope that it will be useful, # but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.

# You should have received a copy of the GNU General Public License along with this program.
# If not, see <[1](https://www.gnu.org/licenses/)>.
# ---------------------------------------------------------------------------------------------------------------------
# pyright: reportArgumentType = false
"""Tests for align of molecular dynamics trajectories."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Self

import MDAnalysis as mda
import numpy as np
import pytest
from click.testing import CliRunner
from fluctmatch.cli import main
from MDAnalysis.analysis.rms import RMSD
from MDAnalysisTests.datafiles import DCD2, PSF
from testfixtures import TempDirectory


@pytest.fixture()
def reference() -> mda.Universe:
    """Generate a reference structure.

    Returns
    -------
    MDAnalysis.Universe
        Reference structure
    """
    return mda.Universe(PSF, DCD2)


class TestAlign:
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
        result = cli_runner.invoke(main, "align -h")

        assert "Usage:" in result.output
        assert result.exit_code == os.EX_OK

    @pytest.mark.parametrize(
        "selection", [("all", "all"), ("ca", "name CA"), ("cab", "name CA CB"), ("backbone", "backbone")]
    )
    def test_align(self, cli_runner: CliRunner, selection: tuple[str, str], reference: mda.Universe) -> None:
        """Test the align function for proper trajectory alignment.

        GIVEN: Paths to the topology, trajectory, reference files, and output directory
        WHEN: The align function is invoked with the specified atom selection and options
        THEN: The output trajectory should be aligned according to the reference structure
              and a log file should be created with the specified verbosity level.

        This test will check if the align function correctly aligns the trajectory to the
        reference structure and creates the expected output files with the correct content.
        It will also verify that the log file is created and contains the expected messages.

        The test will cover:
        - Correct file creation for the aligned trajectory
        - Proper logging in the specified log file
        - Handling of the 'mass' option for mass-weighted alignment
        - Function's response to various 'select' options for atom selection

        Parameters
        ----------
        cli_runner : CliRunner
            Command-line cli_runner
        selection : str
            Atom selection used for alignment between the trajectory and the reference structure.
        reference : mda.Universe
            Reference structure
        """
        with TempDirectory(create=True) as tempdir:
            tmp_path = tempdir.as_path()
            log_file = tmp_path.joinpath("align.log")
            traj_file = tmp_path.joinpath(f"aligned_{Path(DCD2).name}")

            result = cli_runner.invoke(
                main, f"align -s {PSF} -f {DCD2} -o {tmp_path} -l {log_file} -t {selection[0]} --mass"
            )

            # Check outcome of CLI including file creation.
            assert result.exit_code == os.EX_OK
            assert log_file.is_file(), f"{log_file} does not exist"
            assert traj_file.is_file(), f"{traj_file} does not exist"

            # Compre the core of the protein as listed on the MDAnalysis example for r.m.s.d. calculations.
            # https://docs.mdanalysis.org/stable/documentation_pages/analysis/rms.html
            group_selection = [
                f"{selection[1]} and (resid 1-29 or resid 60-121 or resid 160-214)",
            ]
            max_rmsd = 4.0
            aligned = mda.Universe(PSF, traj_file.as_posix()).select_atoms(selection[1])
            rmsd = (
                RMSD(aligned, reference, select=selection[1], groupselections=group_selection, weights="mass")
                .run()
                .results.rmsd[:, 3]
            )
            assert np.all(rmsd < max_rmsd), f"r.m.s.d. should be less than {max_rmsd:.1f}."
