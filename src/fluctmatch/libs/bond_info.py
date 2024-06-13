# ---------------------------------------------------------------------------------------------------------------------
# fluctmatch
# Copyright (c) 2013-2024 Timothy H. Click, Ph.D.
#
# This file is part of fluctmatch.
#
# Fluctmatch is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# Fluctmatch is distributed in the hope that it will be useful, # but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <[1](https://www.gnu.org/licenses/)>.
#
# Reference:
# Timothy H. Click, Nixon Raj, and Jhih-Wei Chu. Simulation. Meth Enzymology. 578 (2016), 327-342,
# Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics doi:10.1016/bs.mie.2016.05.024.
# ---------------------------------------------------------------------------------------------------------------------
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
