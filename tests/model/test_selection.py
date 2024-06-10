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
"""Test additional MDAnalysis selection options."""

from typing import Self

import fluctmatch.model.selection  # noqa: F401
import MDAnalysis as mda
import pytest
from MDAnalysisTests.datafiles import DCD2, DCD_TRICLINIC, PSF, PSF_TRICLINIC, RNA_PDB, RNA_PSF, PDB_elements


@pytest.fixture()
def universe() -> mda.Universe:
    """Fixture for a universe.

    Returns
    -------
    Universe
        universe with protein, DNA, and water
    """
    multiverse = [
        mda.Universe(PSF, DCD2),
        mda.Universe(PSF_TRICLINIC, DCD_TRICLINIC),
        mda.Universe(RNA_PSF, RNA_PDB),
        mda.Universe(PDB_elements),
    ]
    return mda.Merge(*[_.atoms for _ in multiverse])


class TestSelection:
    """Test new MDAnalysis selection options."""

    @pytest.mark.parametrize(
        "selection",
        """hbackbone calpha backbone hcalpha cbeta amine carboxyl hsidechain bioion water nucleic
                             hnucleicsugar hnucleicbase nucleicphosphate sugarC2 sugarC4 nucleiccenter""".split(),
    )
    def test_keywords(self: Self, universe: mda.Universe, selection: str) -> None:
        """Test selection keywords.

        GIVEN an MDANA universe and a selection keyword
        WHEN invoking the select_atoms method
        THEN the number of atoms selected is greater than 0.

        Parameters
        ----------
        universe : MDAnalysis.Universe
            Universe with protein, DNA, and water
        selection : str
            Selection keyword
        """
        selection: mda.AtomGroup = universe.select_atoms(selection)
        assert selection.n_atoms > 0, "Number of atoms don't match."
