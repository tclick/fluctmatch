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
# pyright: reportInvalidTypeVarUse=false, reportOptionalIterable=false, reportAssignmentType=false
"""Elastic network model using C-alpha atoms of a protein."""

from typing import ClassVar, Self

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.align import AverageStructure
from MDAnalysis.core.topologyattrs import Angles, Bonds, Charges, Dihedrals, Impropers
from MDAnalysis.lib.distances import self_capped_distance
from numpy.typing import NDArray

from fluctmatch.libs.utils import rename_universe
from fluctmatch.model.base import CoarseGrainModel


class ElasticModel(CoarseGrainModel):
    """Convert a basic coarse-grain universe into an elastic-network model.

    Determines the interactions between beads via distance cutoffs `rmin` and
    `rmax`. The atoms and residues are also renamed to prevent name collision
    when working with fluctuation matching.
    """

    model: ClassVar[str] = "ENM"
    description: ClassVar[str] = "Elastic network model"

    def __init__(self: Self, mobile: mda.Universe, /, **kwargs: dict[str, bool]) -> None:
        super().__init__(mobile, **kwargs)
        self._universe = self._mobile.copy()

    def create_topology(self: Self) -> Self:
        """Determine the topology attributes and initialize the universe.

        Returns
        -------
        CoarseGrainModel
            Updated coarse-grain model
        """
        rename_universe(self._universe)

        self._universe.add_TopologyAttr(Charges(np.zeros_like(self.atoms.charges)))
        return self

    def generate_bonds(self: Self, rmin: float = 0.0, rmax: float = 10.0, guess: bool = False) -> Self:
        """Add bonds, angles, dihedrals, and improper dihedrals to the universe.

        Parameters
        ----------
        rmin : float
            Minimum bond distance
        rmax : bool
            Maximum bond distance
        guess : bool, optional
            Guess angles and dihedral and improper dihedral angles

        Returns
        -------
        CoarseGrainModel
            Updated coarse-grain model
        """
        super().generate_bonds(rmin, rmax, guess)
        if not guess:
            self._universe.add_TopologyAttr(Angles([]))
            self._universe.add_TopologyAttr(Dihedrals([]))
            self._universe.add_TopologyAttr(Impropers([]))

        return self

    def add_trajectory(
        self: Self,
        start: int | None = None,  # noqa: ARG002
        stop: int | None = None,  # noqa: ARG002
        step: int | None = None,  # noqa: ARG002
        com: bool = True,  # noqa: ARG002
    ) -> Self:
        """Add coordinates to the new system.

        Parameters
        ----------
        start : int, optional
            Beginning frame
        stop : int, optional
            Final frame
        step : int, optional
            Number of frames to skip
        com : bool, optional
            Define positions either by center of mass (default) or of geometry

        Returns
        -------
        CoarseGrainModel
            Updated coarse-grain model
        """
        return self

    def _add_bonds(self: Self, rmin: float, rmax: float) -> None:
        # Determine the average positions of the system
        average = AverageStructure(self._universe)
        average.run()

        # Find bonds with distance range of rmin <= r <= rmax
        pairs, _ = self_capped_distance(average.results.positions, rmax, min_cutoff=rmin)

        # Include predefined bonds
        bonds: NDArray = self._universe.bonds.dump_contents() if hasattr(self._universe, "bonds") else []
        pairs = np.concatenate([pairs, bonds], axis=0)

        bonds = np.unique(pairs, axis=0)

        # Add topology information
        self._universe.add_TopologyAttr(Bonds(bonds))
