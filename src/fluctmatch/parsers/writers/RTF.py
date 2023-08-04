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
# pyright: reportInvalidTypeVarUse=false, reportGeneralTypeIssues=false, reportOptionalMemberAccess=false
# flake8: noqa
"""Write CHARMM residue topology file (RTF)."""

import textwrap
from io import StringIO
from pathlib import Path
from typing import ClassVar, TextIO, TypeVar

import MDAnalysis as mda
import numpy as np
from loguru import logger
from MDAnalysis.core import groups
from MDAnalysis.core.topologyobjects import TopologyObject
from numpy.typing import DTypeLike, NDArray

from ...libs.safe_format import safe_format
from .. import base as topbase

TWriter = TypeVar("TWriter", bound="Writer")


class Writer(topbase.TopologyWriterBase):
    """Write a CHARMM-formatted topology file.

    Parameters
    ----------
    filename : str
        Filename where to write the information.
    n_atoms : int, optional
        The number of atoms in the output trajectory.
    title
        A header section written at the beginning of the stream file.
        If no title is given, a default title will be written.
    charmm_version
        Version of CHARMM for formatting (default: 41)
    """

    format: ClassVar[str] = "RTF"
    units: ClassVar[dict[str, str | None]] = {"time": None, "length": None}
    fmt: ClassVar[dict[str, str]] = {
        "HEADER": "{:>5d}{:>5d}",
        "MASS": "MASS %5d %-6s%12.5f",
        "DECL": "DECL +%s\nDECL -%s",
        "RES": "RESI {:<4s} {:>12.4f}\nGROUP",
        "ATOM": "ATOM %-6s %-6s %7.4f",
        "IC": "IC %-4s %-4s %-4s %-4s %7.4f %8.4f %9.4f %8.4f %7.4f",
    }
    bonds: ClassVar[tuple[str, tuple[str, int]]] = (
        ("BOND", ("bonds", 8)),
        ("IMPH", ("impropers", 8)),
    )

    def __init__(
        self: TWriter,
        filename: str | Path,
        charmm_version: int = 41,
        n_atoms: int | None = None,
    ) -> None:
        super().__init__()
        self.filename: Path = Path(filename).with_suffix("." + self.format.lower())
        self._version: int = charmm_version
        self.n_atoms = n_atoms
        self._atoms: mda.AtomGroup | None = None
        self.rtffile: TextIO | None = None

    def _write_mass(self: TWriter) -> None:
        types, index = np.unique(self._atoms.names, return_index=True)
        masses = self._atoms.masses[index]

        columns = np.hstack(
            (
                index[:, np.newaxis],
                types[:, np.newaxis],
                masses[:, np.newaxis],
            )
        )
        if self._version >= 39:
            columns[:, 0] = -1
        with StringIO() as output:
            np.savetxt(output, columns, fmt=self.fmt["MASS"], delimiter="")
            print(output.getvalue().rstrip(), file=self.rtffile)

    def _write_decl(self: TWriter) -> None:
        names = np.unique(self._atoms.names)[:, np.newaxis]
        decl = np.hstack((names, names))
        with StringIO() as output:
            np.savetxt(output, decl, fmt=self.fmt["DECL"])
            print(output.getvalue().rstrip(), file=self.rtffile)
        print(file=self.rtffile)

    def _write_residues(self: TWriter, residue: groups.Residue) -> None:
        atoms: mda.AtomGroup = residue.atoms

        print(safe_format(self.fmt["RES"], residue.resname, residue.charge), file=self.rtffile)

        # Write the atom lines with site name, type, and charge.
        key = "ATOM"
        atoms = residue.atoms
        lines: tuple[NDArray, ...] = (
            (atoms.names[:, np.newaxis], atoms.types[:, np.newaxis], atoms.charges[:, np.newaxis])
            if not np.issubdtype(atoms.types.dtype, np.number)
            else (atoms.names[:, np.newaxis], atoms.names[:, np.newaxis], atoms.charges[:, np.newaxis])
        )
        with StringIO() as output:
            np.savetxt(output, np.hstack(lines), fmt=self.fmt[key])
            print(output.getvalue().rstrip(), file=self.rtffile)

        # Write the bond, angle, dihedral, and improper dihedral lines.
        for key, (attr, n_perline) in self.bonds:
            fmt = key + n_perline * "%10s"
            try:
                bonds: TopologyObject = getattr(atoms, attr)
                if len(bonds) == 0:
                    continue

                # Create list of atom names and include "+" for atoms not
                # within the residue.
                names: NDArray = np.vstack([_.atoms.names[np.newaxis, :] for _ in bonds])

                idx: list[NDArray] = [np.isin(_.atoms, atoms, invert=True) for _ in bonds]
                idx: NDArray = np.any(idx, axis=1)

                pos_names: NDArray = np.where(np.isin(bonds[idx], atoms, invert=True), "+", "")
                if pos_names.size == 0:
                    logger.warning(
                        "Please check that all bond definitions are valid. You may have some missing or broken bonds."
                    )
                else:
                    names[idx] = pos_names + names[idx]
                names: NDArray = names.astype(str)

                # Eliminate redundancies.
                # Code courtesy of Daniel F on
                # https://stackoverflow.com/questions/45005477/eliminating-redundant-numpy-rows/45006131?noredirect=1#comment76988894_45006131
                dtype: DTypeLike = np.dtype((np.void, names.dtype.itemsize * names.shape[1]))
                sorted_names: NDArray = np.ascontiguousarray(np.sort(names)).view(dtype)
                _, idx = np.unique(sorted_names, return_index=True)
                names: NDArray = names[idx]

                # Add padding for missing columns.
                n_rows, n_cols = names.shape
                n_values: int = n_perline // n_cols
                if n_rows % n_values > 0:
                    n_extra = n_values - (n_rows % n_values)
                    extras = np.full((n_extra, n_cols), "")
                    names = np.concatenate((names, extras))
                names: NDArray = names.reshape((names.shape[0] // n_values, n_perline))
                with StringIO() as output:
                    np.savetxt(output, names, fmt=fmt)
                    print(output.getvalue().rstrip(), file=self.rtffile)
            except AttributeError:
                continue
        print(file=self.rtffile)

    def write(self: TWriter, universe: mda.Universe | mda.AtomGroup, decl: bool = True) -> None:
        """Write a CHARMM-formatted RTF topology file.

        Parameters
        ----------
        universe : :class:`~MDAnalysis.Universe` or :class:`~MDAnalysis.AtomGroup`
            A collection of atoms in a universe or atomgroup with bond
            definitions.
        decl : boolean
            Include the declaration (DECL) statements
        """
        self._atoms: mda.AtomGroup = universe.atoms
        with self.filename.open(mode="w") as self.rtffile:
            # Write the title and header information.
            print(textwrap.dedent(self.title).strip(), file=self.rtffile)
            print(safe_format(self.fmt["HEADER"], 36, 1), file=self.rtffile)
            print(file=self.rtffile)

            # Write the atom mass and declaration sections
            self._write_mass()
            print(file=self.rtffile)
            if decl:
                self._write_decl()
            print("DEFA FIRS NONE LAST NONE", file=self.rtffile)
            print("AUTOGENERATE ANGLES DIHEDRAL\n", file=self.rtffile)

            # Write out the residue information
            _, idx = np.unique(self._atoms.residues.resnames, return_index=True)
            for residue in self._atoms.residues[idx]:
                self._write_residues(residue)
            print("END", file=self.rtffile)
