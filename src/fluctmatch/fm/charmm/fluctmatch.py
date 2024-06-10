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

from fluctmatch.fm.base import FluctuationMatchingBase
from fluctmatch.io.charmm import BondData
from fluctmatch.io.charmm.intcor import CharmmInternalCoordinates
from fluctmatch.io.charmm.parameter import CharmmParameter
from fluctmatch.io.charmm.stream import CharmmStream
from fluctmatch.libs.bond_info import BondInfo
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
        self._fluct_dist: CharmmInternalCoordinates = copy.deepcopy(self._average_dist)
        self._target: CharmmInternalCoordinates = copy.deepcopy(self._average_dist)

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
        self._parameters: CharmmParameter = CharmmParameter().initialize(self._universe, forces=forces, lengths=lengths)
        self._param_ddist: CharmmParameter = copy.deepcopy(self._parameters)
        self._parameters.write(self._stem, stream=True)
        self._stream.write(self._stem.with_suffix(".bonds.str"))

        # Internal coordinate files
        self._average_dist.data = lengths
        self._average_dist.write(self._stem.with_suffix(".average.ic"))
        self._fluct_dist.data = fluct
        self._target.data = fluct
        self._fluct_dist.write(self._stem.with_suffix(".fluct.ic"))
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
