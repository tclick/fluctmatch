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
# pyright: reportInvalidTypeVarUse=false
"""Base class for fluctuation matching."""

import abc
from pathlib import Path
from typing import Self

import MDAnalysis as mda
from scipy import constants

from fluctmatch.io.charmm import BondData


class FluctuationMatchingBase(abc.ABC):
    """Base class for fluctuation matching."""

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
        """
        self._universe = universe.copy()

        if temperature < 0:
            message = "Temperature cannot be negative."
            raise ValueError(message)

        self._temperature = temperature
        self._output_dir = Path.home() if output_dir is None else Path(output_dir)
        self._prefix = Path(prefix)

        # Attempt to create the necessary subdirectory
        self._output_dir.mkdir(exist_ok=True, parents=True)

        # Boltzmann constant (in kcal/mol)
        self.BOLTZMANN: float = temperature * (
            constants.Boltzmann * constants.Avogadro / (constants.calorie * constants.kilo)
        )

        # Bond factor mol^2-Ang./kcal^2
        self.K_FACTOR: float = 0.02

    @abc.abstractmethod
    def calculate(self: Self, *, target: BondData | None = None) -> BondData:
        """Calculate the force constants from the fluctuations.

        Parameters
        ----------
        target : OrderedDict[tuple[str, str], float]
            Target fluctuations

        Returns
        -------
        OrderedDict[tuple[str, str], float]
            Force constants
        """
        message = "Method 'calculate' not implemented."
        raise NotImplementedError(message)

    @abc.abstractmethod
    def initialize(self: Self) -> None:
        """Initialize data for fluctuation matching."""
        message = "Method 'initialize' not implemented."
        raise NotImplementedError(message)

    @abc.abstractmethod
    def load_target(self: Self, target: Path | str) -> None:
        """Load target fluctuation data.

        Parameters
        ----------
        target : Path or str
            Filename with target data
        """
        message = "Method 'load_target' not implemented."
        raise NotImplementedError(message)

    @abc.abstractmethod
    def simulate(
        self: Self,
        *,
        input_dir: Path | str | None = None,
        executable: Path | str | None = None,
        trajectory: Path | str | None = None,
        logfile: Path | str | None = None,
    ) -> BondData:
        """Perform normal mode analysis of the system.

        Parameters
        ----------
        input_dir : Path or str, optional
            Input directory
        executable : Path or str, optional
            Executable file for molecular dynamics program
        trajectory : Path or str, optional
            Trajectory file
        logfile : Path or str, optional
            Log file

        Returns
        -------
        OrderedDict[tuple[str, str], float]
            Bond fluctuations
        """
        message = "Method 'simulate' not implemented."
        raise NotImplementedError(message)
