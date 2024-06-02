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
# pyright: reportInvalidTypeVarUse=false, reportGeneralTypeIssues=false
"""Initialize and write a stream file."""

from pathlib import Path
from typing import Self

import MDAnalysis as mda
import numpy as np
import parmed as pmd
import parmed.charmm as charmm
from loguru import logger
from numpy.typing import NDArray


class CharmmParameter:
    """Initialize, read, or write CHARMM parameter data."""

    def __init__(self: Self) -> None:
        """Prepare a CHARMM parameter file."""
        self._parameters = charmm.CharmmParameterSet()

    @property
    def parameters(self: Self) -> charmm.CharmmParameterSet:
        """Return the actual parameter object.

        Returns
        -------
        charmm.CharmmParameterSet
            Atom and bond types
        """
        return self._parameters

    @property
    def forces(self: Self) -> NDArray:
        """Return the force constant data.

        Returns
        -------
        numpy.ndarray
            Force constant data
        """
        return np.asarray([bond_type.k for bond_type in self._parameters.bond_types.values()])

    @forces.setter
    def forces(self: Self, k: NDArray) -> None:
        """Set the force constant data.

        Parameters
        ----------
        k : numpy.ndarray
            Force constant data
        """
        logger.debug("Updating the equilibrium force constants.")
        try:
            for bond_type, force in zip(self._parameters.bond_types, k, strict=True):
                self._parameters.bond_types[bond_type].k = force
        except ValueError as e:
            logger.exception(e)
            raise

    @property
    def distances(self: Self) -> NDArray:
        """Return the equilibrium bond distance data.

        Returns
        -------
        numpy.ndarray
            Equilibrium bond distance data
        """
        return np.asarray([bond_type.req for bond_type in self._parameters.bond_types.values()])

    @distances.setter
    def distances(self: Self, req: NDArray) -> None:
        """Set the equilibrium bond distance data.

        Parameters
        ----------
        req : numpy.ndarray
            Equilibrium bond distance data
        """
        logger.debug("Updating the equilibrium bond distances.")
        try:
            for bond_type, distance in zip(self._parameters.bond_types, req, strict=True):
                self._parameters.bond_types[bond_type].req = distance
        except ValueError as e:
            logger.exception(e)
            raise

    def initialize(
        self: Self, universe: mda.Universe, forces: NDArray | None = None, lengths: NDArray | None = None
    ) -> None:
        """Fill the parameters with atom types and bond information.

        Parameters
        ----------
        universe : :class:`mda.Universe`
            Universe with bond information
        forces : :class:`numpy.ndarray`, optional
            Force constants between bonded atoms
        lengths : :class:`numpy.ndarray`, optional
            Bond lengths between atoms

        Raises
        ------
        MDAnalysis.NoDataError
            if no bond data exists
        ValueError
            if number of bonds, force constants, or bond lengths do not match
        """
        # Prepare atom types
        logger.debug("Adding atom types to the parameter list.")
        atom_types: dict[str, pmd.AtomType] = {}
        for atom in universe.atoms:
            atom_type = pmd.AtomType(atom.type, -1, atom.mass, charge=atom.charge)
            atom_type.epsilon = 0.0
            atom_type.epsilon_14 = 0.0
            atom_type.sigma = 0.0
            atom_type.sigma_14 = 0.0
            atom_type.number = -1
            atom_types[atom.type] = atom_type
        self._parameters.atom_types.update(atom_types)

        # Setup atom bond types
        if forces is not None and lengths is not None:
            try:
                logger.debug("Adding bond types to the parameter list.")
                bond_keys = universe.bonds.topDict.keys()
                bond_type = {
                    key: pmd.BondType(k=force, req=length)
                    for key, force, length in zip(bond_keys, forces, lengths, strict=True)
                }
                self._parameters.bond_types.update(bond_type)
            except mda.NoDataError as e:
                e.add_note("No bond data found in the universe.")
                logger.exception(e)
                raise
            except ValueError as e:
                e.add_note("Forces or lengths are not equal to the number of bonds.")
                logger.exception(e)
                raise

    def write(
        self: Self, par: str | Path | None = None, top: str | Path | None = None, stream: str | Path | None = None
    ) -> None:
        """Write the parameter data to a parameter, topology, or stream file.

        Parameters
        ----------
        par : str or Path, optional
            Parameter file
        top : str or Path, optional
            Topology file
        stream : str or Path, optional
            Stream file

        Raises
        ------
        ValueError
            if neither the parameter or stream filename is provided
        """
        if par is not None:
            logger.info(f"Writing parameter file: {par}")
        if top is not None:
            logger.info(f"Writing topology file: {top}")
        if stream is not None:
            logger.info(f"Writing stream file: {stream}")

        if not self._parameters.atom_types and not self._parameters.bond_types:
            logger.warning("No atom types or bond types were provided. The parameter file will be empty.")

        par = Path(par).as_posix() if par is not None else None
        top = Path(top).as_posix() if top is not None else None
        stream = Path(stream).as_posix() if stream is not None else None
        self._parameters.write(par=par, top=top, stream=stream)

    def read(self: Self, filename: str | Path) -> None:
        """Read the parameters from a file.

        Parameters
        ----------
        filename : str or Path
            Parameter file

        Raises
        ------
        FileNotFoundError
            if parameter, topology, or stream file not found
        """
        param_file = Path(filename)
        try:
            if param_file.suffix == ".par" or param_file.suffix == ".prm":
                self._parameters.read_parameter_file(param_file.as_posix())
            elif param_file.suffix == ".rtf":
                self._parameters.read_topology_file(param_file.as_posix())
            else:
                self._parameters.read_stream_file(param_file.as_posix())
        except FileNotFoundError as e:
            logger.exception(e)
            raise

        if len(self._parameters.atom_types) == 0 and len(self._parameters.bond_types) == 0:
            message = "No atom types or bond types found in the parameter file"
            logger.exception(message)
            raise OSError(message)
