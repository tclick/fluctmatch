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
# flake8: noqa
"""Test writer for COR files."""
from pathlib import Path
from typing import TypeVar
from unittest.mock import patch

import MDAnalysis as mda
import pytest
from numpy.testing import assert_equal

import fluctmatch.parsers.readers.COR
from ..datafile import COR

Self = TypeVar("Self")


class TestCORWriter:
    @staticmethod
    @pytest.fixture()
    def universe() -> mda.Universe:
        return mda.Universe(COR)

    def test_writer(self: Self, universe: mda.Universe, tmp_path: Path):
        filename: Path = tmp_path / "temp.cor"
        with patch("fluctmatch.parsers.writers.COR.Writer.write") as writer, mda.Writer(
            filename.as_posix(), n_atoms=universe.atoms.n_atoms
        ) as w:
            w.write(universe.atoms)
            writer.assert_called()

    def test_roundtrip(self: Self, universe: mda.Universe, tmp_path: Path):
        # Write out a copy of the Universe, and compare this against the
        # original. This is more rigorous than simply checking the coordinates
        # as it checks all formatting
        filename: Path = tmp_path / "temp.cor"
        crd: Path = filename.with_suffix(".crd")

        with mda.Writer(filename.as_posix(), n_atoms=universe.atoms.n_atoms) as w:
            w.write(universe.atoms)

        with mda.Writer(crd.as_posix(), n_atoms=universe.atoms.n_atoms, extended=True) as w:
            w.write(universe.atoms)

        def CRD_iter(fn: Path | str):
            with open(fn) as inf:
                for line in inf:
                    if not line.startswith("*"):
                        yield line

        for ref, other in zip(CRD_iter(crd), CRD_iter(filename)):
            assert ref == other

    def test_write_atoms(self: Self, universe: mda.Universe, tmp_path: Path):
        # Test that written file when read gives same coordinates
        filename: Path = tmp_path / "temp.cor"
        with mda.Writer(filename.as_posix(), n_atoms=universe.atoms.n_atoms) as w:
            w.write(universe.atoms)

        u2 = mda.Universe(filename)

        assert_equal(universe.atoms.positions, u2.atoms.positions)
