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
"""Tests for fluctuation matching using CHARMM."""

from pathlib import Path
from typing import Self

import MDAnalysis as mda
import pytest
from fluctmatch.fm.charmm.fluctmatch import CharmmFluctuationMatching

from tests.datafile import FLUCTDCD, FLUCTPSF


@pytest.fixture(scope="class")
def universe() -> mda.Universe:
    """Universe of an elastic network model.

    Returns
    -------
    MDAnalysis.Universe
        Elastic network model
    """
    return mda.Universe(FLUCTPSF, FLUCTDCD)


class TestCharmmFluctuationMatching:
    """Tests for CharmmFluctuationMatching."""

    def test_initialize(self: Self, universe: mda.Universe, tmp_path: Path) -> None:
        """Test initialization.

        Parameters
        ----------
        universe : MDAnalysis.Universe
            Universe of an elastic network model
        tmp_path : Path
            Temporary path
        """
        prefix = "fluctmatch"
        stem = tmp_path.joinpath(prefix)
        fm = CharmmFluctuationMatching(universe, output_dir=tmp_path, prefix=prefix).initialize()
        assert len(fm._parameters.parameters.bond_types) > 0, "Bond data not found."
        assert stem.with_suffix(".str").exists(), "Parameter file not found."
        assert stem.with_suffix(".inp").exists(), "CHARMM input file not found."
