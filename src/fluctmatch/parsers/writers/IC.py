# ------------------------------------------------------------------------------
#  fluctmatch
#  Copyright (c) 2023 Timothy H. Click, Ph.D.
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
# pyright: reportInvalidTypeVarUse=false, reportInvalidTypeVarUse=false, reportGeneralTypeIssues=false
# pyright: reportIncompatibleMethodOverride=false
# flake8: noqa
"""Writer for internal coordinates."""

import textwrap
from io import StringIO
from pathlib import Path
from typing import ClassVar, TypeVar

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray

from ...libs.intcor import InternalCoord
from ...libs.safe_format import safe_format
from ..base import TopologyWriterBase

TWriter = TypeVar("TWriter", bound="Writer")


class Writer(TopologyWriterBase):
    """Write a CHARMM-formatted internal coordinate file.

    Parameters
    ----------
    filename : str or :class:`Path`
        Filename for output.
    n_atoms : int, optional
        The number of atoms in the output trajectory.
    extended : bool, optional
        Format with wider columns than the standard column width.
    resid : bool, optional
        Include segment names within each atom definition.
    title : str or list of str, optional
        A header section written at the beginning of the stream file.
        If no title is given, a default title will be written.
    """

    format: ClassVar[str] = "IC"
    units: ClassVar[dict[str, str | None]] = {"time": "picosecond", "length": "Angstrom"}
    fmt: ClassVar[dict[str, str]] = {
        "STANDARD": "%5d %3s %-4s%3s %-4%3s %-4%3s %-4%9.4f%8.2f%8.2f%8.2f%9.4f",
        "EXTENDED": "%10d %5s %-8s%5s %-8s%5s %-8s%5s %-8s%9.4f%8.2f%8.2f%8.2f%9.4f",
        "STANDARD_RESID": (
            "%5d %-4s %-4s %-4s: %-4s %-4s %-4s: %-4s %-4s %-4s: %-4s %-4s %-4s:%12.6f%12.4f%12.4f%12.4f%12.6f"
        ),
        "EXTENDED_RESID": (
            "%10d %-8s %-8s %-8s: %-8s %-8s %-8s: %-8s %-8s %-8s: %-8s %-8s %-8s:%12.6f%12.4f%12.4f%12.4f%12.6f"
        ),
    }

    def __init__(
        self: TWriter, filename: str | Path, *, extended: bool = True, resid: bool = True, n_atoms: int | None = None
    ) -> None:
        super().__init__()

        self.filename = Path(filename).with_suffix("." + self.format.lower())
        self._intcor: pd.DataFrame | None = None
        self._extended: bool = extended
        self._resid: bool = resid
        self.key: str = "EXTENDED" if self._extended else "STANDARD"
        self.key += "_RESID" if self._resid else ""
        self.n_atom: int | None = n_atoms

    def write(self: TWriter, intcor: InternalCoord, /) -> None:
        """Write an internal coordinates table.

        Parameters
        ----------
        intcor : InternalCoord
            A CHARMM-compliant internal coordinate table.
        """
        with open(self.filename, "w") as outfile:
            logger.info(safe_format("Writing to {}", self.filename))
            # Save the title lines
            print(textwrap.dedent(self.title).strip(), file=outfile)

            # Save the header information
            line: NDArray = np.zeros((1, 20), dtype=int)
            line[0, 0]: int = 30 if self._extended else 20
            line[0, 1]: int = 2 if self._resid else 1
            with StringIO() as output:
                np.savetxt(output, line, fmt="%4d", delimiter="")
                print(output.getvalue().rstrip(), file=outfile)

            # Save the internal coordinates
            line: NDArray = np.zeros((1, 2), dtype=int)
            line[0, 0] += intcor.data["r_IJ"].size
            line[0, 1] += 2 if self._resid else 1
            with StringIO() as output:
                np.savetxt(output, line, fmt="%5d", delimiter="")
                print(output.getvalue().rstrip(), file=outfile)

            table: pd.DataFrame = intcor.create_table()
            table.index += 1
            with StringIO() as output:
                np.savetxt(output, table.reset_index().values, fmt=self.fmt[self.key])
                print(output.getvalue().rstrip(), file=outfile)
            logger.info("Table successfully written.")
