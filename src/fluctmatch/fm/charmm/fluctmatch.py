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
# pyright: reportInvalidTypeVarUse=false, reportArgumentType=false
"""Fluctuation matching using CHARMM.

Notes
-----
For CHARMM to work with the fluctuation matching code, it must be
recompiled with some modifications to the source code. `ATBMX`, `MAXATC`,
`MAXCB` (located in dimens.fcm [c35] or dimens_ltm.src [c39]) must
be increased. `ATBMX` determines the number of bonds allowed per
atom, `MAXATC` describes the maximum number of atom core, and `MAXCB`
determines the maximum number of bond parameters in the CHARMM parameter
file. Additionally, `CHSIZE` may need to be increased if using an earlier
version (< c36).
"""

import copy
import shlex
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Self

import MDAnalysis as mda
import numpy as np
import pandas as pd
from loguru import logger
from scipy.constants import Avogadro, Boltzmann, calorie, kilo
from sklearn.metrics import root_mean_squared_error

from fluctmatch.fm.base import FluctuationMatchingBase
from fluctmatch.io.charmm import BondData
from fluctmatch.io.charmm.intcor import CharmmInternalCoordinates
from fluctmatch.io.charmm.parameter import CharmmParameter
from fluctmatch.io.charmm.stream import CharmmStream
from fluctmatch.libs.bond_info import BondInfo
from fluctmatch.libs.utils import compare_dict_keys
from fluctmatch.libs.write_files import write_charmm_input


class CharmmFluctuationMatching(FluctuationMatchingBase):
    """Simulate fluctuation matching using CHARMM."""

    def __init__(
        self: Self,
        universe: mda.Universe,
        /,
        *,
        temperature: float = 300.0,
        output_dir: Path | str | None = None,
        prefix: Path | str = "fluctmatch",
    ) -> None:
        """Initialize of fluctuation matching.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            Elastic network model
        temperature : float (default: 300.0)
            Temperature in Kelvin
        output_dir, str, optional (default=None)
            Output directory
        prefix : str, optional (default "fluctmatch")
            Filename prefix

        Attributes
        ----------
        BOLTZMANN : float
            Boltzmann constant in kJ/mol

        Raises
        ------
        AttributeError
            Universe does not contain bond information
        MDAnalysis.NoDataError
            if no bond data exists
        ValueError
            if number of bonds, force constants, or bond lengths do not match
        """
        super().__init__(universe, temperature=temperature, output_dir=output_dir, prefix=prefix)

        self._stream: CharmmStream = CharmmStream().initialize(universe)
        self._average_dist: CharmmInternalCoordinates = CharmmInternalCoordinates().initialize(universe)
        self._optimized: CharmmInternalCoordinates = copy.deepcopy(self._average_dist)
        self._target: CharmmInternalCoordinates = copy.deepcopy(self._average_dist)
        self._parameters: CharmmParameter = CharmmParameter().initialize(universe)
        self._param_ddist: CharmmParameter = copy.deepcopy(self._parameters)

        self._stem: Path = self._output_dir.joinpath(self._prefix)
        self.BOLTZMANN: float = temperature * Boltzmann * Avogadro / (calorie * kilo)

    def initialize(self: Self) -> Self:
        """Initialize fluctuation matching.

        Determine the initial bond lengths and force constants to use within the simulations.

        Parameters
        ----------
        universe : :class:`mda.Universe`
            Elastic network model

        Returns
        -------
        CharmmFluctuationMatching
            Self
        """
        logger.info("Determining the average bond distance and the corresponding bond fluctuations.")
        bond_info = BondInfo(self._universe.atoms).run()
        lengths: BondData = bond_info.results.mean
        fluct: BondData = bond_info.results.std
        forces: BondData = (self.BOLTZMANN / pd.Series(fluct).apply(np.square)).to_dict(into=OrderedDict)

        # CHARMM parameter, topology, and stream files
        self._parameters.forces = forces
        self._parameters.distances = lengths
        self._parameters.write(self._stem, stream=True)
        self._param_ddist.forces = forces
        self._param_ddist.distances = lengths
        self._stream.write(self._stem.with_suffix(".bonds.str"))

        # Internal coordinate files
        self._average_dist.data = lengths
        self._average_dist.write(self._stem.with_suffix(".average.ic"))
        self._optimized.data = fluct
        self._target.data = fluct
        self._optimized.write(self._stem.with_suffix(".fluct.ic"))
        self._target.write(self._output_dir.joinpath("target").with_suffix(".ic"))

        write_charmm_input(
            topology=self._universe.filename,
            trajectory=self._universe.trajectory.filename,
            directory=self._output_dir,
            prefix=self._prefix,
            temperature=self._temperature,
        )
        return self

    def simulate(self: Self, executable: Path | str) -> None:
        """Run fluctuation matching.

        Parameters
        ----------
        executable : Path or str
            Path for executable file used for molecular dynamics program
        """
        _executable = Path(executable)
        if not _executable.exists():
            message = f"{_executable} does not exist."
            raise FileNotFoundError(message)

        logfile = self._stem.with_suffix(".log")
        input_file = self._stem.with_suffix(".inp")
        cmd: list[str] = shlex.split(f"{_executable} -i {input_file}")
        try:
            with input_file.open() as inp, logfile.open("w") as log:
                logger.debug(f"Running command {' '.join(cmd)}")
                subprocess.run(cmd, stdin=inp, stdout=subprocess.PIPE, stderr=log, check=True)  # noqa: S603
        except subprocess.CalledProcessError as e:
            e.add_note(f"{shlex.join(cmd)} failed.")
            logger.exception(e)
            raise

        self._optimized.read(self._stem.with_suffix(".fluct.ic"))
        self._average_dist.read(self._stem.with_suffix(".average.ic"))

    def load_target(self: Self, filename: str | Path, /) -> Self:
        """Load target bond fluctuations.

        Parameters
        ----------
        filename : path_like
            Name of internal coordinates file containing target bond fluctuations
        """
        self._target.read(filename)
        return self

    def load_parameters(self: Self, filename: str | Path, /) -> Self:
        """Load CHARMM parameter file.

        Parameters
        ----------
        filename : path_like
            Name of CHARMM parameter file
        """
        self._parameters.read(filename)
        self._param_ddist.parameters.atom_types.update(self._parameters.parameters.atom_types)
        self._param_ddist.parameters.bond_types.update(self._parameters.parameters.bond_types)
        return self

    def calculate(self: Self) -> float:
        """Calculate the new force constants from the optimized bond fluctuations and write the updated file.

        Returns
        -------
        float
            Root mean squared error between the target and the optimized bond fluctuations.

        Raises
        ------
        ValueError
            If the bond descriptions between the target and the optimized bond fluctuations are incorrect.
        """
        try:
            compare_dict_keys(self._target.data, self._optimized.data)
        except ValueError:
            message = "Target and optimized bonds do not match."
            logger.exception(message)
            raise

        logger.info("Calculating the new force constants.")
        target: pd.Series = pd.Series(self._target.data)
        optimized: pd.Series = pd.Series(self._optimized.data)
        forces: pd.Series = (optimized - target).multiply(self.BOLTZMANN * self.K_FACTOR)
        forces[forces < 0] = 0.0
        self._parameters.forces = forces.to_dict(into=OrderedDict)
        self._param_ddist.forces = forces.to_dict(into=OrderedDict)
        self._param_ddist.distances = pd.Series(self._average_dist.data).to_dict(into=OrderedDict)

        logger.info(f"Writing the updated parameter file to {self._stem.with_suffix('.str')}.")
        self._parameters.write(self._stem, stream=True)
        logger.info(f"Writing the updated parameter file to {self._stem.with_suffix('.str')}.")
        self._param_ddist.write(self._stem.as_posix() + "_dist", stream=True)

        rmse = root_mean_squared_error(target, optimized)
        logger.info(f"Root mean squared error: {rmse:.4f}")
        return rmse
