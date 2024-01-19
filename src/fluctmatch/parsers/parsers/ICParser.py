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
# pyright: reportInvalidTypeVarUse=false, reportUnboundVariable=false, reportGeneralTypeIssues=false
# flake8: noqa
"""Internal coordinates reader."""

from typing import ClassVar, TypeVar

from loguru import logger
from MDAnalysis.lib.util import FORTRANReader
from MDAnalysis.topology.base import TopologyReaderBase

from fluctmatch.libs import intcor
from fluctmatch.libs.safe_format import safe_format

Self = TypeVar("Self", bound="Reader")


class Reader(TopologyReaderBase):
    """Read internal coordinate file."""

    format: ClassVar[str] = "IC"

    _lines: ClassVar[dict[str, str]] = {
        "STD": "6X,I3,1X,A4,I3,1X,A4,A3,1X,A4,A3,1X,A4,F9.4,3F8.2,F9.4",
        "STD_RESID": (
            "5X,1X,A4,1X,I4,1X,A4,1X,1X,A4,1X,I4,1X,A4,1X,1X,A4,1X,A4,1X,A4,1X,1X,A4,1X,A4,1X,A4,1X,F12.6,3F12.4,F12.6"
        ),
        "EXT": "10X,I5,1X,A8,I5,1X,A8,A5,1X,A8,A5,1X,A8,1X,F9.4,3F8.2,F9.4",
        "EXT_RESID": (
            "10X,1X,A8,1X,I8,1X,A8,1X,1X,A8,1X,I8,1X,A8,1X,1X,A8,1X,A8,1X,A8,1X,1X,A8,1X,A8,1X,A8,1X,F12.6,3F12.4,F12.6"
        ),
    }

    def parse(self: Self, **kwargs) -> intcor.InternalCoord:
        """Read an internal coordinate file.

        Returns
        -------
        InternalCoord
            Internal coordinates information
        """
        with open(self.filename) as infile:
            # Read title and header lines
            for line in infile:
                line = line.split("!")[0].strip()
                if line.startswith("*") or line.startswith("!") or not line:
                    continue  # ignore TITLE, comments, and empty lines
                break
            line: list[int] = list(map(int, line.split()))
            key: str = "EXT" if line[0] == 30 else "STD"
            key += "_RESID" if line[1] == 2 else ""
            logger.info(safe_format("Reading {} with {} format", self.filename, key))
            formatter = FORTRANReader(self._lines[key])
            n_lines, _ = list(map(int, next(infile).split()))

            internal = intcor.InternalCoord(n_lines=n_lines)
            for i, line in enumerate(infile):
                if "RESID" in key:
                    segidI, resI, I, segidJ, resJ, J, _, _, _, _, _, _, r_IJ, _, _, _, _ = formatter.read(line)
                    internal.data["segidI"][i], internal.data["resI"][i], internal.data["I"][i] = segidI, resI, I
                    internal.data["segidJ"][i], internal.data["resJ"][i], internal.data["J"][i] = segidJ, resJ, J
                    internal.data["r_IJ"][i] = r_IJ
                else:
                    internal.data["resI"][i], internal.data["I"][i] = resI, I
                    internal.data["resJ"][i], internal.data["J"][i] = resJ, J
                    internal.data["r_IJ"][i] = r_IJ

        return internal
