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
# pyright: reportOptionalIterable=false, reportInvalidTypeVarUse=false, reportGeneralTypeIssues=false
# flake8: noqa
"""Class to write CHARMM parameter files with a prm extension."""

import textwrap
from io import StringIO
from pathlib import Path
from typing import ClassVar, TypeVar

import numpy as np
import pandas as pd

from ...libs.parameters import Parameters
from ..base import TopologyWriterBase

TWriter = TypeVar("TWriter", bound="Writer")


class Writer(TopologyWriterBase):
    """Write a parameter dictionary to a CHARMM-formatted parameter file.

    Parameters
    ----------
    filename : str or :class:`~MDAnalysis.lib.util.NamedStream`
         name of the output file or a stream
    title : str
        Title lines at beginning of the file.
    charmm_version
        Version of CHARMM for formatting (default: 41)
    nonbonded
        Add the nonbonded section. (default: False)
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
    _FORMAT: ClassVar[dict[str, str]] = {
        "ATOMS": "MASS %5d %-6s %9.5f",
        "BONDS": "%-6s %-6s %10.4f%10.4f",
        "ANGLES": "%-6s %-6s %-6s %8.2f%10.2f%10s%10s",
        "DIHEDRALS": "%-6s %-6s %-6s %-6s %12.4f%3d%9.2f",
        "IMPROPER": "%-6s %-6s %-6s %-6s %12.4f%3d%9.2f",
        "NONBONDED": "%-6s %5.1f %13.4f %10.4f",
    }

    def __init__(
        self: TWriter,
        filename: str | Path,
        *,
        charmm_version: int = 41,
        nonbonded: bool = False,
        n_atoms: int | None = None,
    ) -> None:
        super().__init__()

        self.filename: Path = Path(filename).with_suffix(".prm")
        self._version: int = charmm_version
        self._nonbonded: bool = nonbonded
        self.n_atoms: int = n_atoms

    def write(self: TWriter, parameters: Parameters) -> None:
        """Write a CHARMM-formatted parameter file.

        Parameters
        ----------
        parameters : dict
            Keys are the section names and the values are of class
            :class:`~pandas.DataFrame`, which contain the corresponding
            parameter data.
        """
        nb_header: str = """
            NONBONDED nbxmod  5 atom cdiel shift vatom vdistance vswitch -
            cutnb 14.0 ctofnb 12.0 ctonnb 10.0 eps 1.0 e14fac 1.0 wmin 1.5
            """
        data: dict[str, pd.DataFrame] = parameters.create_table()

        with open(self.filename, "w") as prmfile:
            print(textwrap.dedent(self.title).strip(), file=prmfile)
            print(file=prmfile)

            if self._version >= 39 and data["ATOMS"].size > 0:
                data["ATOMS"]["type"] = -1

            for key, section in data.items():
                if (self._version < 36 and key == "ATOMS") or section.size == 0:
                    continue

                print(key, file=prmfile)
                with StringIO() as output:
                    np.savetxt(output, section.values, fmt=self._FORMAT[key])
                    print(output.getvalue().rstrip(), file=prmfile)
                print(file=prmfile)

            print(textwrap.dedent(nb_header[1:])[:-1], file=prmfile)

            if self._nonbonded:
                atom_list = (
                    data["ATOMS"]["atom"]
                    if data["ATOMS"].size > 0
                    else pd.unique(pd.concat([data["BONDS"]["I"], data["BONDS"]["J"]], axis="rows"))
                )
                nb_list = pd.concat([atom_list, pd.DataFrame(np.zeros((atom_list.size, 3)))], axis="columns")
                with StringIO() as output:
                    np.savetxt(output, nb_list.values, fmt=self._FORMAT["NONBONDED"], delimiter="")
                    print(output.getvalue().rstrip(), file=prmfile)
            print("\nEND", file=prmfile)
