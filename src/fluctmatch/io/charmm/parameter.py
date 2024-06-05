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
# pyright: reportInvalidTypeVarUse=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false
"""Initialize and write a stream file."""

from collections import OrderedDict
from pathlib import Path
from typing import Self

import MDAnalysis as mda
import parmed as pmd
import parmed.charmm as charmm
from loguru import logger

from fluctmatch.io.base import IOBase
from fluctmatch.io.charmm import BondData
from fluctmatch.libs.utils import compare_dict_keys


class CharmmParameter(IOBase):
    """Initialize, read, or write CHARMM parameter data."""

    def __init__(self: Self) -> None:
        """Prepare a CHARMM parameter file."""
        super().__init__()
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
    def forces(self: Self) -> BondData:
        """Return the force constant data.

        Returns
        -------
        OrderedDict[tuple[str, str], float]
            Force constant data
        """
        return OrderedDict({key: bond_type.k for key, bond_type in self._parameters.bond_types.items()})

    @forces.setter
    def forces(self: Self, k: BondData) -> None:
        """Set the force constant data.

        Parameters
        ----------
        k : OrderedDict[tuple[str, str], float]
            Force constant data

        Raises
        ------
        ValueError
            if number or types of bonds, force constants, or bond lengths do not match
        """
        logger.debug("Updating the equilibrium force constants.")
        try:
            compare_dict_keys(self._parameters.bond_types, k, message="Bond types do not match.")
            for bond_type in self._parameters.bond_types:
                self._parameters.bond_types[bond_type].k = k[bond_type]
        except ValueError as e:
            logger.exception(e)
            raise

    @property
    def distances(self: Self) -> BondData:
        """Return the equilibrium bond distance data.

        Returns
        -------
        OrderedDict[tuple[str, str], float]
            Equilibrium bond distance data
        """
        return OrderedDict({key: bond_type.req for key, bond_type in self._parameters.bond_types.items()})

    @distances.setter
    def distances(self: Self, req: BondData) -> None:
        """Set the equilibrium bond distance data.

        Parameters
        ----------
        req : numpy.ndarray
            Equilibrium bond distance data

        Raises
        ------
        ValueError
            if number or types of bonds or bond lengths do not match
        """
        logger.debug("Updating the equilibrium bond distances.")
        try:
            compare_dict_keys(self._parameters.bond_types, req, message="Bond types do not match.")
            for bond_type in self._parameters.bond_types:
                self._parameters.bond_types[bond_type].req = req[bond_type]
        except ValueError as e:
            logger.exception(e)
            raise

    def initialize(
        self: Self, universe: mda.Universe, /, forces: BondData | None = None, lengths: BondData | None = None
    ) -> Self:
        """Fill the parameters with atom types and bond information.

        Parameters
        ----------
        universe : :class:`mda.Universe`
            Universe with bond information
        forces : OrderedDict[tuple[str, str], float], optional
            Force constants between bonded atoms
        lengths : OrderedDict[tuple[str, str], float], optional
            Bond lengths between atoms

        Returns
        -------
        CharmmParameter
            Self

        Raises
        ------
        MDAnalysis.NoDataError
            if no bond data exists
        ValueError
            if number or types of bonds, force constants, or bond lengths do not match
        """
        # Prepare atom types
        logger.debug("Adding atom types to the parameter list.")
        self._parameters.atom_types.update(self._parameters.atom_types.fromkeys(universe.atoms.names))
        for atom in universe.atoms:
            atom_type = pmd.AtomType(atom.type, -1, atom.mass, charge=atom.charge)
            atom_type.epsilon = 0.0
            atom_type.epsilon_14 = 0.0
            atom_type.sigma = 0.0
            atom_type.sigma_14 = 0.0
            atom_type.number = -1
            self._parameters.atom_types[atom.type] = atom_type

        # Setup atom bond types
        if forces is not None and lengths is not None:
            try:
                logger.debug("Adding bond types to the parameter list.")
                self._parameters.bond_types.update(self._parameters.bond_types.fromkeys(universe.bonds.types()))
                compare_dict_keys(forces, lengths, message="Bond force constants and bond distances do not match.")
                compare_dict_keys(
                    self._parameters.bond_types,
                    forces,
                    message="Bonds defining force constants and distances do not match bonds in universe.",
                )

                for bond_type in self._parameters.bond_types:
                    self._parameters.bond_types[bond_type] = pmd.BondType(k=forces[bond_type], req=lengths[bond_type])
            except mda.NoDataError as e:
                e.add_note("No bond data found in the universe.")
                logger.exception(e)
                raise
            except ValueError as e:
                e.add_note("Forces or lengths are not equal to the number of bonds.")
                logger.exception(e)
                raise

        return self

    def write(self: Self, filename: Path | str, /) -> None:
        """Write the parameter data to a parameter, topology, or stream file.

        Parameters
        ----------
        filename : Path or str
            Filename to write the parameter data
        """
        par = Path(filename).with_suffix(".prm")
        top = par.with_suffix(".rtf")
        stream = par.with_suffix(".str")

        logger.info(f"Writing parameter file: {par}")
        logger.info(f"Writing topology file: {top}")
        logger.info(f"Writing stream file: {stream}")

        if not self._parameters.atom_types and not self._parameters.bond_types:
            logger.warning("No atom types or bond types were provided. The parameter file will be empty.")

        self._parameters.write(par=par.as_posix(), top=top.as_posix(), stream=stream.as_posix())

    def read(self: Self, filename: str | Path, /) -> Self:
        """Read the parameters from a file.

        Parameters
        ----------
        filename : str or Path
            Parameter file

        Returns
        -------
        CharmmParameter
            Self

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

        return self
