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
# flake8: noqa
"""Class to write CHARMM extended coordinate files with a cor extension."""

from pathlib import Path
from typing import ClassVar, TypeVar

from MDAnalysis.coordinates import CRD

TWriter = TypeVar("TWriter", bound="Writer")


class Writer(CRD.CRDWriter):
    """CRD writer that implements the CHARMM CRD coordinate format.

    It automatically writes the CHARMM EXT extended format if there
    are more than 99,999 atoms.

    Requires the following attributes to be present:
    - resids
    - resnames
    - names
    - chainIDs
    - tempfactors

    - versionchanged: 0.11.0
       Frames now 0-based instead of 1-based
    - versionchanged: 2.2.0
       CRD extended format can now be explicitly requested with the
       `extended` keyword
    """

    format: ClassVar[str] = "COR"
    units: ClassVar[dict[str, str | None]] = {"time": None, "length": "Angstrom"}

    def __init__(self: TWriter, filename: str | Path, **kwargs: str) -> None:
        """CRD writer that implements the CHARMM CRD coordinate format.

        Parameters
        ----------
        filename : str or :class:`~MDAnalysis.lib.util.NamedStream`
             name of the output file or a stream
        """
        super().__init__(filename, **kwargs)

        self.filename = Path(filename).with_suffix("." + self.format.lower())

        # account for explicit crd format, if requested
        self.extended = kwargs.pop("extended", True)
