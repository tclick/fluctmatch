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
# pyright: reportInvalidTypeVarUse=false
"""Write CHARMM stream file."""

import itertools
import textwrap
from pathlib import Path
from typing import ClassVar, TypeVar

import MDAnalysis as mda
from loguru import logger
from MDAnalysis.core.topologyattrs import TopologyGroup

from ...libs.safe_format import safe_format
from .. import base as topbase

TWriter = TypeVar("TWriter", bound="Writer")


class Writer(topbase.TopologyWriterBase):
    """Write a stream file to define internal coordinates within CHARMM.

    Parameters
    ----------
    filename : str
        The filename to which the internal coordinate definitions are written.
    n_atoms : int, optional
        The number of atoms in the output trajectory.
    title
        A header section written at the beginning of the stream file.
        If no title is given, a default title will be written.
    """

    format: ClassVar[str] = "STREAM"  # noqa: A003
    units: ClassVar[dict[str, str | None]] = {"time": None, "length": "Angstrom"}

    def __init__(self: TWriter, filename: str | Path, n_atoms: int | None = None) -> None:
        super().__init__()

        self.filename: Path = Path(filename).with_suffix(".stream")
        self.n_atoms: int = n_atoms

        # Column width
        self.fmt = """
            IC EDIT
            DIST {segid1:<8} {resid1:8d} {name1:<8} {segid2:<8} {resid2:8d} {name2:<8}{dist:8.1f}
            END
            """

    def write(self: TWriter, universe: mda.Universe | mda.AtomGroup) -> None:
        """Write the bond information to a CHARMM-formatted stream file.

        Parameters
        ----------
        universe : :class:`~MDAnalysis.Universe` or :class:`~MDAnalysis.AtomGroup`
            A collection of atoms in a universe or atomgroup with bond
            definitions.
        """
        # Create the table
        try:
            bonds: TopologyGroup = universe.bonds
            atom1: mda.AtomGroup = bonds.atom1
            atom2: mda.AtomGroup = bonds.atom2

            data = zip(
                atom1.segids.tolist(),
                atom1.resids.tolist(),
                atom1.names.tolist(),
                atom2.segids.tolist(),
                atom2.resids.tolist(),
                atom2.names.tolist(),
                itertools.repeat(0.0, atom2.names.size),
                strict=True,
            )
        except AttributeError as err:
            msg = "No bonds were found."
            raise AttributeError(msg) from err

        # Write the data to the file.
        formatted: str = textwrap.dedent(self.fmt.strip("\n"))
        with open(self.filename, "w") as stream_file:
            logger.info(f"Writing a CHARMM stream file to {self.filename}.")
            print(textwrap.dedent(self.title).strip(), file=stream_file)
            for segid1, resid1, name1, segid2, resid2, name2, dist in data:
                values = {
                    "segid1": segid1,
                    "resid1": resid1,
                    "name1": name1,
                    "segid2": segid2,
                    "resid2": resid2,
                    "name2": name2,
                    "dist": dist,
                }
                print(safe_format(formatted, **values), file=stream_file)
            print("RETURN", file=stream_file)
