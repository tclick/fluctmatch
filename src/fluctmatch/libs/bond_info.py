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
"""Calculate the bond averages and fluctuations."""

from typing import TypeVar

import MDAnalysis as mda
import MDAnalysis.analysis.rms as rms
import numpy as np
from loguru import logger
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.exceptions import SelectionError
from numpy.typing import NDArray

from ..libs.safe_format import safe_format

TBondInfo = TypeVar("TBondInfo", bound="BondInfo")


class BondInfo(AnalysisBase):
    """Calculate the average bond length and the standard deviations of bonds.

    Parameters
    ----------
    mobile : Universe
        Universe containing trajectory to be fitted to reference
    reference : Universe, optional
        Universe containing trajectory frame to be used as reference
    select : str, optional
        Set as default to all, is used for Universe.select_atoms to choose subdomain to be fitted against
    filename : str, optional
        Provide a filename for results to be written to
    in_memory : bool, optional
        *Permanently* switch `mobile` to an in-memory trajectory so that alignment can be done in-place, which can
        improve performance substantially in some cases. In this case, no file is written out (`filename` and `prefix`
        are ignored) and only the coordinates of `mobile` are *changed in memory*.
    ref_frame : int, optional
        frame index to select frame from `reference`
    verbose : bool, optional
        Set logger to show more information and show detailed progress of the calculation if set to ``True``;
        the default is ``False``.

    Attributes
    ----------
    reference_atoms : AtomGroup
        Atoms of the reference structure to be aligned against
    mobile_atoms : AtomGroup
        Atoms inside each trajectory frame to be rmsd_aligned
    results.average : np.ndarray(dtype=float)
        Average bond distances
    results.fluctuation : np.ndarray(dtype=float)
        Fluctuations between bond distances

    Notes
    -----
    - If set to ``verbose=False``, it is recommended to wrap the statement in a ``try ...  finally`` to guarantee
      restoring of the log level in the case of an exception.
    - The ``in_memory`` option changes the `mobile` universe to an in-memory representation (see
      :mod:`MDAnalysis.coordinates.memory`) for the remainder of the Python session. If ``mobile.trajectory`` is
      already a :class:`MemoryReader` then it is *always* treated as if ``in_memory`` had been set to ``True``.
    """

    def __init__(
        self: TBondInfo,
        mobile: mda.Universe,
        reference: mda.Universe | None = None,
        select: str = "all",
        filename: str | None = None,
        in_memory: bool = False,
        ref_frame: int = 0,
        **kwargs: str,
    ) -> None:
        if in_memory or isinstance(mobile.trajectory, MemoryReader):
            mobile.transfer_to_memory()
            filename = None
            logger.info("Moved mobile trajectory to in-memory representation")

        # do this after setting the memory reader to have a reference to the right reader.
        super().__init__(mobile.trajectory, **kwargs)
        if not self._verbose:
            logger.disable("WARN")

        self.reference = reference if reference is not None else mobile

        selection = rms.process_selection(select)
        self.ref_atoms = self.reference.select_atoms(*selection["reference"])
        self.mobile_atoms = mobile.select_atoms(*selection["mobile"])

        if len(self.ref_atoms) != len(self.mobile_atoms):
            err = safe_format(
                "Reference and trajectory atom selections do not contain the same number of atoms: N_ref={0:d}, N_traj={1:d}",
                self.ref_atoms.n_atoms,
                self.mobile_atoms.n_atoms,
            )
            logger.exception(err)
            raise SelectionError(err)

        # store reference to mobile atoms
        self.mobile = mobile.atoms
        self.ref_frame = ref_frame

        self.filename = filename

    def _prepare(self: TBondInfo) -> None:
        current_frame: int = self.reference.universe.trajectory.ts.frame
        try:
            # Move to the ref_frame
            # (coordinates MUST be stored in case the ref traj is advanced
            # elsewhere or if ref == mobile universe)
            self.reference.universe.trajectory[self.ref_frame]
        finally:
            # Move back to the original frame
            self.reference.universe.trajectory[current_frame]

        # allocate the array for selection atom coords
        self.bonds: list[NDArray] = []
        self.results.average = np.zeros(len(self.mobile_atoms.bonds))
        self.results.fluctuation = np.zeros(len(self.mobile_atoms.bonds))

    def _single_frame(self: TBondInfo) -> None:
        self.bonds.append(self.mobile_atoms.bonds.bonds().copy())

    def _conclude(self: TBondInfo) -> None:
        self.results.average = np.mean(self.bonds, axis=0)
        self.results.fluctuation = np.std(self.bonds, axis=0)
        del self.bonds
