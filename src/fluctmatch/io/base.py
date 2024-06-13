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
"""Base class for I/O."""

import abc
from pathlib import Path
from typing import Self

import MDAnalysis as mda


class IOBase(abc.ABC):
    """Base class for I/O."""

    def __init__(self: Self) -> None:  # noqa: B027
        """Construct an I/O class."""
        pass

    @abc.abstractmethod
    def initialize(self: Self, universe: mda.Universe, /) -> Self:
        """Initialize the I/O class.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            Universe

        Returns
        -------
        IOBase
            Self
        """
        message = "Method 'initiate' not implemented."
        raise NotImplementedError(message)

    @abc.abstractmethod
    def write(self: Self, filename: Path | str, /) -> None:
        """Write data to a file.

        Parameters
        ----------
        filename : Path or str
            Name of file
        """
        message = "Method 'write' not implemented."
        raise NotImplementedError(message)

    @abc.abstractmethod
    def read(self: Self, filename: Path | str, /) -> Self:
        """Read data from a file.

        Parameters
        ----------
        filename : Path or str
            Name of file

        Returns
        -------
        IOBase
            Self
        """
        message = "Method 'read' not implemented."
        raise NotImplementedError(message)
