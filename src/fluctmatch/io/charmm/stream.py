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
