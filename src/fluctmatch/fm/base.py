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
# pyright: reportInvalidTypeVarUse=false
"""Base class for fluctuation matching."""

import abc
from pathlib import Path
from typing import Self

import MDAnalysis as mda


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

        Raises
        ------
        ValueError
            If temperature is <= 0.0.
        """
        self._universe = universe.copy()

        if temperature <= 0:
            message = "Temperature cannot be negative or 0 K."
            raise ValueError(message)

        self._temperature = temperature
        self._output_dir = Path.home() if output_dir is None else Path(output_dir)
        self._prefix = Path(prefix)

        # Bond factor mol^2-Ang./kcal^2
        self.K_FACTOR: float = 0.02

    @abc.abstractmethod
    def initialize(self: Self) -> Self:
        """Initialize data for fluctuation matching."""
        message = "Method 'initialize' not implemented."
        raise NotImplementedError(message)

    @abc.abstractmethod
    def simulate(self: Self, executable: Path) -> None:
        """Run fluctuation matching.

        Parameters
        ----------
        executable : Path
            Path for executable file used for molecular dynamics program
        """
        message = "Method 'simulate' not implemented."
        raise NotImplementedError(message)

    @abc.abstractmethod
    def load_target(self: Self, filename: str | Path, /) -> Self:
        """Load target bond fluctuations.

        Parameters
        ----------
        filename : path_like
            Name of internal coordinates file containing target bond fluctuations
        """
        message = "Method 'load_target' not implemented."
        raise NotImplementedError(message)

    @abc.abstractmethod
    def load_parameters(self: Self, filename: str | Path, /) -> Self:
        """Load a parameter file.

        Parameters
        ----------
        filename : path_like
            Name of CHARMM parameter file
        """
        message = "Method 'load_parameters' not implemented."
        raise NotImplementedError(message)

    @abc.abstractmethod
    def calculate(self: Self) -> float:
        """Caluclate the new force constants from the optimized bond fluctuations."""
        message = "Method 'calculate' not implemented."
        raise NotImplementedError(message)
