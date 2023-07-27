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
# pyright: reportInvalidTypeVarUse=false, reportGeneralTypeIssues=false
"""Internal coordinates based upon CHARMM."""

from collections.abc import Iterable
from typing import ClassVar, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Self = TypeVar("Self", bound="InternalCoord")


class InternalCoord:
    """Class containing information about internal coordinates of a system."""

    _header: ClassVar[list[str]] = [
        "segidI",
        "resI",
        "I",
        "segidJ",
        "resJ",
        "J",
        "segidK",
        "resK",
        "K",
        "segidL",
        "resL",
        "L",
        "r_IJ",
        "T_IJK",
        "P_IJKL",
        "T_JKL",
        "r_KL",
    ]

    def __init__(self: Self, n_lines: int = 1) -> None:
        """Initialize storage for internal coordinates with `n_lines`.

        Parameters
        ----------
        n_lines : int
            number of lines
        """
        extra: list[str] = ["??"] * n_lines

        self.data: dict[str, list[str] | NDArray] = dict.fromkeys(self._header)
        self.data["segidI"] = [""] * n_lines
        self.data["resI"] = np.zeros(n_lines, dtype=int)
        self.data["I"] = [""] * n_lines
        self.data["segidJ"] = [""] * n_lines
        self.data["resJ"] = np.zeros(n_lines, dtype=int)
        self.data["J"] = [""] * n_lines
        self.data["segidK"] = extra.copy()
        self.data["resK"] = extra.copy()
        self.data["K"] = extra.copy()
        self.data["segidL"] = extra.copy()
        self.data["resL"] = extra.copy()
        self.data["L"] = extra.copy()
        self.data["r_IJ"] = np.zeros(n_lines, dtype=float)
        self.data["T_IJK"] = np.zeros(n_lines, dtype=float)
        self.data["P_IJKL"] = np.zeros(n_lines, dtype=float)
        self.data["T_JKL"] = np.zeros(n_lines, dtype=float)
        self.data["r_KL"] = np.zeros(n_lines, dtype=float)

    def create_table(self: Self) -> pd.DataFrame:
        """Create a table of the internal coordinates.

        Returns
        -------
        DataFrame
            the internal coordinates
        """
        return pd.DataFrame.from_dict(self.data)

    def iterate(self: Self) -> Iterable[list[str | int | float]]:
        """Iterate over the internal coordinates in a line-by-line fashion.

        Returns
        -------
        Iterable
            internal coordinates by line
        """
        return zip(*(self.data.values()), strict=True)


def merge(*intcors: InternalCoord) -> pd.DataFrame:
    """Merge multiple internal coordinate tables.

    Tables will be merged using the segidI, resI, I, segidJ, resJ, J as the indices.

    Parameters
    ----------
    intcors : list of `InternalCoord`
        multiple internal coordinates

    Returns
    -------
    DataFrame
        A new table with multiple r_IJ
    """
    tables: list[pd.DataFrame] | pd.DataFrame = []
    header: list[str] = list(intcors[0].data.keys())

    for intcor in intcors:
        table: pd.DataFrame = intcor.create_table().set_index(header[:6])
        table: pd.DataFrame = table.drop(header[6:12], axis="columns").drop(header[13:], axis="columns")
        tables.append(table)

    tables: pd.DataFrame = pd.concat(tables, axis="columns").fillna(0.0)
    return tables
