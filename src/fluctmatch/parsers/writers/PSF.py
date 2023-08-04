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
"""Write CHARMM PSF files (for CHARMM36 style)."""

import logging
import textwrap
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, TextIO, TypeVar
from collections.abc import Iterator
from io import StringIO

import MDAnalysis as mda
import numpy as np
from numpy.typing import NDArray

from ...libs.safe_format import safe_format
from .. import base

logger: logging.Logger = logging.getLogger(__name__)

TWriter = TypeVar("TWriter", bound="Writer")


class Writer(base.TopologyWriterBase):
    """PSF writer that implements the CHARMM PSF topology format.

    Requires the following attributes to be present:
    - ids
    - names
    - core
    - masses
    - charges
    - resids
    - resnames
    - segids
    - bonds

    versionchanged:: 3.0.0
       Uses numpy arrays for bond, angle, dihedral, and improper outputs.

    Parameters
    ----------
    filename : str or :class:`~MDAnalysis.lib.util.NamedStream`
         name of the output file or a stream
    n_atoms : int, optional
        The number of atoms in the output trajectory.
    extended
         extended format
    cmap
         include CMAP section
    cheq
         include charge equilibration
    title
         title lines at beginning of the file
    charmm_version
        Version of CHARMM for formatting (default: 41)
    """

    format: ClassVar[str] = "PSF"
    units: ClassVar[dict[str, str | None]] = {"time": None, "length": None}
    _fmt: ClassVar[MappingProxyType[str, str]] = MappingProxyType(
        {
            "STD": "{:>8d} {:<4} {:<4d} {:<4} {:<4} {:>4d} {:>14.6f}{:>14.6f}{:>8d}",
            "STD_XPLOR": "{:>8d} {:<4} {:<4d} {:<4} {:<4} {:<4} {:>14.6f}{:>14.6f}{:>8d}",
            "STD_XPLOR_C35": "{:>4d} {:<4} {:<4d} {:<4} {:<4} {:>4} {:>14.6f}{:>14.6f}{:>8d}",
            "EXT": "{:>10d} {:<8} {:<8d} {:<8} {:<8} {:>4d} {:>14.6f}{:>14.6f}{:>8d}",
            "EXT_XPLOR": "{:>10d} {:<8} {:<8d} {:<8} {:<8} {:<6} {:>14.6f}{:>14.6f}{:>8d}",
            "EXT_XPLOR_C35": "{:>10d} {:<8} {:<8d} {:<8} {:<8} {:<4} {:>14.6f}{:>14.6f}{:>8d}",
        }
    )

    def __init__(
        self: TWriter,
        filename: str | Path,
        *,
        extended: bool = True,
        xplor: bool = True,
        cmap: bool = True,
        cheq: bool = True,
        charmm_version: int = 41,
        n_atoms: int | None = None,
    ) -> None:
        super().__init__()

        self.filename = Path(filename).with_suffix(".psf").as_posix()
        self._extended = extended
        self._xplor = xplor
        self._cmap = cmap
        self._cheq = cheq
        self._version = charmm_version
        self._universe: mda.Universe | None = None
        self._fmtkey = "EXT" if self._extended else "STD"
        self.n_atoms = n_atoms

        self.col_width = 10 if self._extended else 8
        self.sect_hdr = "{:>10d} !{}" if self._extended else "{:>8d} !{}"
        self.sect_hdr2 = "{:>10d}{:>10d} !{}" if self._extended else "{:>8d}{:>8d} !{}"
        self.sections: tuple[tuple[str, str, int], ...] = (
            ("bonds", "NBOND: bonds", 8),
            ("angles", "NTHETA: angles", 9),
            ("dihedrals", "NPHI: dihedrals", 8),
            ("impropers", "NIMPHI: impropers", 8),
            ("donors", "NDON: donors", 8),
            ("acceptors", "NACC: acceptors", 8),
        )

    def write(self: TWriter, universe: mda.Universe | mda.AtomGroup) -> None:
        """Write universe to PSF format.

        Parameters
        ----------
        universe : :class:`~MDAnalysis.Universe` or :class:`~MDAnalysis.AtomGroup`
            A collection of atoms in a universe or atomgroup with bond
            definitions.
        """
        try:
            self._universe = universe.copy()
        except TypeError:
            self._universe = mda.Universe(universe.filename, universe.trajectory.filename)

        header = "PSF"
        if self._extended:
            header += " EXT"
        if self._cheq:
            header += " CHEQ"
        if self._xplor:
            header += " XPLOR"
            self._fmtkey += "_XPLOR"
            if self._version < 36:
                self._fmtkey += "_C35"
        if self._cmap:
            header += " CMAP"

        header += ""

        with open(self.filename, mode="w") as psffile:
            print(header, file=psffile)
            print(file=psffile)
            n_title = len(self.title.strip().split("\n"))
            print(safe_format(self.sect_hdr, n_title, "NTITLE"), file=psffile)
            print(textwrap.dedent(self.title).strip(), file=psffile)
            print(file=psffile)
            self._write_atoms(psffile)
            for section in self.sections:
                self._write_sec(psffile, section)
            self._write_other(psffile)

    def _write_atoms(self: TWriter, psffile: TextIO) -> None:
        """Write atom section in a Charmm PSF file.

        Normal (standard) and extended (EXT) PSF format are
        supported.


        CHARMM Format from ``source/psffres.src``:

        no CHEQ::
         standard format:
            (I8,1X,A4,1X,A4,1X,A4,1X,A4,1X,I4,1X,2G14.6,I8)
            (I8,1X,A4,1X,A4,1X,A4,1X,A4,1X,I4,1X,2G14.6,I8,2G14.6) CHEQ
            (I8,1X,A4,1X,A4,1X,A4,1X,A4,1X,A6,1X,2G14.6,I8)  XPLOR
            (I8,1X,A4,1X,A4,1X,A4,1X,A4,1X,A6,1X,2G14.6,I8,2G14.6)  XPLOR,CHEQ
            (I8,1X,A4,1X,A4,1X,A4,1X,A4,1X,A4,1X,2G14.6,I8)  XPLOR,c35
            (I8,1X,A4,1X,A4,1X,A4,1X,A4,1X,A4,1X,2G14.6,I8,2G14.6) XPLOR,c35,CHEQ
          expanded format EXT:
            (I10,1X,A8,1X,A8,1X,A8,1X,A8,1X,I4,1X,2G14.6,I8)
            (I10,1X,A8,1X,A8,1X,A8,1X,A8,1X,I4,1X,2G14.6,I8,2G14.6) CHEQ
            (I10,1X,A8,1X,A8,1X,A8,1X,A8,1X,A6,1X,2G14.6,I8) XPLOR
            (I10,1X,A8,1X,A8,1X,A8,1X,A8,1X,A6,1X,2G14.6,I8,2G14.6) XPLOR,CHEQ
            (I10,1X,A8,1X,A8,1X,A8,1X,A8,1X,A4,1X,2G14.6,I8) XPLOR,c35
            (I10,1X,A8,1X,A8,1X,A8,1X,A8,1X,A4,1X,2G14.6,I8,2G14.6) XPLOR,c35,CHEQ
        """
        fmt: str = self._fmt[self._fmtkey]

        print(safe_format(self.sect_hdr, self._universe.atoms.n_atoms, "NATOM"), file=psffile)
        atoms: mda.AtomGroup = self._universe.atoms
        atoms.charges[atoms.charges == -0.0] = 0.0

        atom_types: NDArray
        if not self._xplor:
            try:
                atom_types = atoms.types.astype(int)
            except ValueError:
                key, index = np.unique(atoms.types, return_index=True)
                int_types = dict(zip(key, index, strict=True))
                atom_types = np.asarray([int_types[_] for _ in atoms.types])
        else:
            try:
                _ = atoms.types.astype(int)
                atom_types = atoms.names
            except ValueError:
                atom_types = atoms.types

        num: list[int] = (np.arange(atoms.n_atoms) + 1).tolist()
        segids: list[str] = atoms.segids.tolist()
        resids: list[int] = atoms.resids.tolist()
        resnames: list[str] = atoms.resnames.tolist()
        names: list[str] = atoms.names.tolist()
        types: list[str | int] = atom_types.tolist()
        charges: list[float] = atoms.charges.tolist()
        masses: list[float] = atoms.masses.tolist()
        ids: list[int] = np.zeros_like(atoms.ids).tolist()

        data: Iterator
        if self._cheq:
            fmt += "{:>10.6f}{:>18}"
            cheq: list[float] = np.zeros_like(masses).tolist()
            cmap: list[str] = ["-0.301140E-02"] * len(cheq)
            data = zip(num, segids, resids, resnames, names, types, charges, masses, ids, cheq, cmap, strict=True)
        else:
            data = zip(num, segids, resids, resnames, names, types, charges, masses, ids, strict=True)

        for line in data:
            print(safe_format(fmt, *line).rstrip(), file=psffile)
        print(file=psffile)

    def _write_sec(self: TWriter, psffile: TextIO, section_info: tuple[str, str, int]) -> None:
        attr, header, n_perline = section_info

        if not hasattr(self._universe, attr) or len(getattr(self._universe, attr).to_indices()) < 2:
            print(safe_format(self.sect_hdr, 0, header), file=psffile)
            print("\n", file=psffile)
            return

        values: NDArray = np.asarray(getattr(self._universe, attr).to_indices()) + 1
        values: NDArray = values.astype(object)
        n_rows, n_cols = values.shape
        n_values: int = n_perline // n_cols
        if n_rows % n_values > 0:
            n_extra: int = n_values - (n_rows % n_values)
            values: NDArray = np.concatenate((values, np.full((n_extra, n_cols), fill_value="", dtype=object)))
        values: NDArray = values.reshape((values.shape[0] // n_values, n_perline))
        print(safe_format(self.sect_hdr, n_rows, header), file=psffile)
        with StringIO() as output:
            np.savetxt(output, values, fmt=f"%{self.col_width:d}s", delimiter="")
            print(output.getvalue().rstrip(), file=psffile)
        print(file=psffile)

    def _write_other(self: TWriter, psffile: TextIO) -> None:
        n_atoms: int = self._universe.atoms.n_atoms
        n_cols: int = 8
        dn_cols: int = n_atoms % n_cols
        missing: int = n_cols - dn_cols if dn_cols > 0 else dn_cols

        # NNB
        nnb: NDArray = np.full(n_atoms, "0", dtype=object)
        if missing > 0:
            nnb = np.concatenate([nnb, np.full(missing, "", dtype=object)])
        nnb: NDArray = nnb.reshape((nnb.size // n_cols, n_cols))

        print(safe_format(self.sect_hdr, 0, "NNB\n"), file=psffile)
        with StringIO() as output:
            np.savetxt(output, nnb, fmt=f"%{self.col_width:d}s", delimiter="")
            print(output.getvalue().rstrip(), file=psffile)
        print(file=psffile)

        # NGRP NST2
        print(safe_format(self.sect_hdr2, 1, 0, "NGRP NST2"), file=psffile)
        line: NDArray = np.zeros(3, dtype=int)
        line: NDArray = line.reshape((1, line.size))
        np.savetxt(psffile, line, fmt=f"%{self.col_width:d}d", delimiter="")
        print(file=psffile)

        # MOLNT
        if self._cheq:
            line: NDArray = np.full(n_atoms, "1", dtype=object)
            if dn_cols > 0:
                line: NDArray = np.concatenate([line, np.zeros(missing, dtype=object)])
            line: NDArray = line.reshape((line.size // n_cols, n_cols))
            print(safe_format(self.sect_hdr, 1, "MOLNT"), file=psffile)
            with StringIO() as output:
                np.savetxt(output, line, fmt=f"%{self.col_width:d}s", delimiter="")
                print(output.getvalue().rstrip(), file=psffile)
            print(file=psffile)
        else:
            print(safe_format(self.sect_hdr, 0, "MOLNT"), file=psffile)
            print("\n", file=psffile)

        # NUMLP NUMLPH
        print(safe_format(self.sect_hdr2, 0, 0, "NUMLP NUMLPH").rstrip(), file=psffile)
        print("\n", file=psffile)

        print(safe_format(self.sect_hdr, 0, "NCRTERM: cross-terms").rstrip(), file=psffile)
