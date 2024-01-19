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
# pyright: reportOptionalIterable=false, reportInvalidTypeVarUse=false, reportUnboundVariable=false
# pyright: reportIncompatibleMethodOverride=false
# flake8: noqa
"""Class to read CHARMM parameter files."""

from pathlib import Path
from typing import ClassVar, TypeVar

import pandas as pd
from loguru import logger
from MDAnalysis.lib.util import FORTRANReader

from ...libs.parameters import Parameters
from ...libs.safe_format import safe_format
from ..base import TopologyReaderBase

TReader = TypeVar("TReader", bound="Reader")


class Reader(TopologyReaderBase):
    """Read a CHARMM-formated parameter file.

    Parameters
    ----------
    filename : str or :class:`~MDAnalysis.lib.util.NamedStream`
         name of the output file or a stream
    """

    format: ClassVar[str] = "PRM"
    units: ClassVar[dict[str, str | None]] = {"time": None, "length": "Angstrom"}

    _HEADERS: ClassVar[tuple[str, ...]] = (
        "ATOMS",
        "BONDS",
        "ANGLES",
        "DIHEDRALS",
        "IMPROPER",
    )
    _COLUMNS: ClassVar[dict[str, list]] = {
        "ATOMS": "type atom mass".split(),
        "BONDS": "I J Kb b0".split(),
        "ANGLES": "I J K Ktheta theta0 Kub S0".split(),
        "DIHEDRALS": "I J K L Kchi n delta".split(),
        "IMPROPER": "I J K L Kchi n delta".split(),
    }
    _FORMAT: ClassVar[dict[str, str]] = {
        "ATOMS": "4X,1X,I5,1X,A6,1X,F9.5",
        "BONDS": "A6,1X,A6,1X,F10.4,F10.4",
        "ANGLES": "A6,1X,A6,1X,A6,F8.2,F10.2,A10,A10",
        "DIHEDRALS": "A6,1X,A6,1X,A6,1X,A6,1X,F12.4,I3,F9.2",
        "IMPROPER": "A6,1X,A6,1X,A6,1X,A6,1X,F12.4,I3,F9.2",
        "NONBONDED": "A6,1X,F5.1,1XF13.4,1XF10.4",
    }

    def __init__(self: TReader, filename: str | Path) -> None:
        self.filename = Path(filename).with_suffix("." + self.format.lower()).as_posix()

    def read(self: TReader) -> Parameters:
        """Parse the parameter file.

        Returns
        -------
        Named tuple with CHARMM parameters per key.
        """
        buffers = {key: [] for key in self._HEADERS}
        logger.debug(safe_format("Reading {}", self.filename))
        with open(self.filename) as prmfile:
            section: str
            for line in prmfile:
                line: str = line.strip()
                if line.startswith("*") or line.startswith("!") or not line:
                    continue  # ignore TITLE and empty lines

                # Parse sections
                if line in self._HEADERS:
                    section = line
                    continue

                if (
                    line.startswith("NONBONDED")
                    or line.startswith("CMAP")
                    or line.startswith("END")
                    or line.startswith("end")
                ):
                    break

                buffers[section].append(FORTRANReader(self._FORMAT[section]).read(line))

        params = {key: pd.DataFrame(value, columns=self._COLUMNS[key]) for key, value in buffers.items() if value}
        parameters = Parameters()
        parameters.from_dataframe(**params)

        return parameters
