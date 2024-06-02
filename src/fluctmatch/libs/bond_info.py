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
# pyright: reportInvalidTypeVarUse=false
"""Calculate the average bond distance and the corresponding standard deviation."""

from collections import OrderedDict
from typing import Any, Self

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from numpy.typing import NDArray


class BondInfo(AnalysisBase):
    """Calculate the average bond distance and the corresponding standard deviation."""

    def __init__(self: Self, atoms: mda.AtomGroup, verbose: bool = False, **kwargs: dict[str, Any]) -> None:
        """Construct the class.

        Parameters
        ----------
        universe : :class:`MDAnalysis.AtomGroup`
            The universe to be analyzed.
        verbose : bool, optional
            Turn on more logging and debugging

        Notes
        -----
        This class assumes that every bond type is different, which should be the case for an elastic network model. If
        this is not the case, this class will raise a ValueError because of the different number of the bond types
        compared with the bond distances.
        """
        super().__init__(atoms.universe.trajectory, verbose, **kwargs)

        self._atoms = atoms

    def _prepare(self: Self) -> None:
        self._bond_types = self._atoms.bonds.types()
        self._bonds: list[NDArray] = []
        self.results.std = OrderedDict.fromkeys(self._bond_types)
        self.results.mean = OrderedDict.fromkeys(self._bond_types)

    def _single_frame(self: Self) -> None:
        self._bonds.append(self._atoms.bonds.values())

    def _conclude(self: Self) -> None:
        bonds = np.array(self._bonds).T
        bond_std = np.std(bonds, axis=1)
        bond_mean = np.mean(bonds, axis=1)
        for bond_type, std, mean in zip(self._bond_types, bond_std, bond_mean, strict=True):
            self.results.std[bond_type] = std
            self.results.mean[bond_type] = mean

        del self._bonds
        del self._atoms
