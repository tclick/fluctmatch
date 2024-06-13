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
"""Initialize and write a stream file."""

import datetime
import getpass
from pathlib import Path
from typing import Self

import MDAnalysis as mda
from loguru import logger

from fluctmatch.io.base import IOBase


class CharmmStream(IOBase):
    """Initialize and write a CHARMM stream data from bond data."""

    def __init__(self: Self) -> None:
        """Prepare a CHARMM stream file."""
        super().__init__()
        self._lines: list[str] = []

    def initialize(self: Self, universe: mda.Universe, /) -> Self:
        """Initialize a stream file from bond data.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            Universe with bond data

        Returns
        -------
        CharmmStream
            Self

        Raises
        ------
        AttributeError
            Universe does not contain bond information
        """
        if not hasattr(universe, "bonds"):
            message = "The universe does not contain bond information."
            logger.exception(message)
            raise AttributeError(message, obj=universe, name="bonds")

        for bond in universe.bonds:
            atom1, atom2 = bond.atoms
            line = (
                "IC EDIT\n"
                f"DIST {atom1.segid:<8s} {atom1.resid:>8d} {atom1.name:<8s} "
                f"{atom2.segid:<8s} {atom2.resid:>8d} {atom2.name:<8s} {0.0:8.1f}\n"
                "END\n\n"
            )
            self._lines.append(line)

        return self

    def write(self: Self, filename: Path | str, /, title: list[str] | None = None) -> None:
        """Write bond data to stream file.

        Parameters
        ----------
        filename : Path or str
            Internal coordinate file
        title : list of str, optional
            Title lines at top of CHARMM file
        """
        now: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user: str = getpass.getuser()
        _title: list[str] = (
            title if title is not None else [f"* Created by fluctmatch on {now}.\n", f"* User: {user}\n"]
        )

        logger.info(f"Writing bond data to {filename}")
        with Path(filename).open(mode="w") as stream:
            stream.writelines(_title)
            stream.writelines(self._lines)
            stream.write("RETURN\n")

    def read(self: Self, filename: str | Path) -> Self:  # noqa: ARG002
        """Read bond data from stream file."""
        message = "Method 'read' not implemented."
        raise NotImplementedError(message)
