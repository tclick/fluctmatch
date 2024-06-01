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


class CharmmStream:
    """Initialize and write a CHARMM stream data from bond data."""

    def __init__(self: Self) -> None:
        """Prepare a CHARMM stream file."""
        self._lines: list[str] = []

    def initialize(self: Self, universe: mda.Universe) -> None:
        """Initialize a stream file from bond data.

        Parameters
        ----------
        universe : :class:`MDAnalysis.Universe`
            Universe with bond data

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
                f"DIST {atom1.segid:<8s} {atom1.resid:>4d} {atom1.name:<8s} "
                f"{atom2.segid:<8s} {atom2.resid:>4d} {atom2.name:<8s} {0.0:.1f}"
            )
            self._lines.append(line)

    def write(self: Self, filename: str | Path, /, title: list[str] | None = None) -> None:
        """Write bond data to stream file."""
        now: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user: str = getpass.getuser()
        _title: list[str] = title if title is not None else [f"* Created by fluctmatch on {now}.", f"* User: {user}"]

        logger.info(f"Writing bond data to {filename}")
        with Path(filename).open(mode="w") as stream:
            for _ in _title:
                stream.write(_ + "\n")

            for line in self._lines:
                stream.write(f"IC EDIT\n{line}\nEND\n\n")

            stream.write("RETURN\n")
