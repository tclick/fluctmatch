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
"""Base classes for the model and the factory."""

import abc
from types import MappingProxyType
from typing import Self

import MDAnalysis as mda


class CoarseGrainModel(abc.ABC):
    """Base class for creating coarse-grain core.

    Parameters
    ----------
    mobile : Universe
        All-atom universe
    com : bool, optional
        Calculates the bead coordinates using either the center of mass
        (default) or center of geometry.
    guess_angles : bool, optional
        Once Universe has been created, attempt to guess the connectivity
        between atoms.  This will populate the .angles, .dihedrals, and
        .impropers attributes of the Universe.

    Attributes
    ----------
    _universe : :class:`~MDAnalysis.Universe`
        The transformed universe
    """

    def __init__(self: Self, mobile: mda.Universe, /, **kwargs: dict[str, object]) -> None:
        """Initialise like a normal MDAnalysis Universe but give the mapping and com keywords.

        Mapping must be a dictionary with atom names as keys.
        Each name must then correspond to a selection string,
        signifying how to split up a single residue into many beads.
        eg:
        mapping = {"CA":"protein and name CA",
                   "CB":"protein and not name N HN H HT* H1 H2 H3 CA HA* C O OXT
                   OT*"}
        would split residues into 2 beads containing the C-alpha atom and the
        sidechain.
        """
        # Coarse grained Universe
        # Make a blank Universe for myself.
        self._mobile = mobile
        self._universe: mda.Universe = mda.Universe.empty(0)
        self._com = kwargs.get("com", True)
        self._guess = kwargs.get("guess_angles", False)

        # Named tuple for specific bead selections. This is primarily used to
        # determine positions.
        self._mapping: MappingProxyType = MappingProxyType({})

        # Named tuple for all-atom to bead selection. This is particularly
        # useful for mass, charge, velocity, and force topology attributes.
        self._selection: MappingProxyType = MappingProxyType({})

        # Beads from all-atom system
        self._beads: list[mda.AtomGroup] = []
        self._mass_beads: list[mda.AtomGroup] = []

        # Residues with corresponding atoms and selection criteria.
        self._residues: tuple[tuple[str, str, object]] | None = None

    @abc.abstractmethod
    def _add_bonds(self: Self) -> None:
        pass


class CoarseGrainModelFactory(abc.ABC):
    """Abstract base class for the model factory."""

    @abc.abstractmethod
    def create_model(self: Self, model_type: str, **kwargs: dict[str, object]) -> CoarseGrainModel:
        """Create a model.

        Parameters
        ----------
        model_type: str
            The type of model to create.
        **kwargs
            Additional keyword arguments specific to the model type.

        Returns
        -------
        CoarseGrainModel
            An instance of the created coarse-grained model.

        Raises
        ------
        ValueError
            If an unsupported model type is provided
        """
        pass
