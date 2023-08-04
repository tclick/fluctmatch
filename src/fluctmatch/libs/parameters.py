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
# pyright: reportOptionalIterable=false, reportInvalidTypeVarUse=false, reportGeneralTypeIssues=false
# pyright: reportOptionalSubscript=false
"""Class to store parameters."""

import copy
from typing import ClassVar, TypeVar

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray

TParameters = TypeVar("TParameters", bound="Parameters")


class Parameters:
    """Molecular dynamics parameters."""

    # Headers
    _HEADERS: ClassVar[dict[str, tuple[str, ...]]] = {
        "ATOMS": ("type", "atom", "mass"),
        "BONDS": ("I", "J", "Kb", "b0"),
        "ANGLES": ("I", "J", "K", "Ktheta", "theta0"),
        "DIHEDRALS": ("I", "J", "K", "L", "Kchi", "n", "delta"),
        "IMPROPER": ("I", "J", "K", "L", "Kchi", "n", "delta"),
    }

    def __init__(
        self: TParameters,
        n_atoms: int = 1,
        n_bonds: int = 0,
        n_angles: int = 0,
        n_dihedrals: int = 0,
        n_improper: int = 0,
    ) -> None:
        """Set up the parameters.

        Parameters
        ----------
        n_atoms : int, default=1
            number of atoms
        n_bonds : int, default=0
            number of bonds
        n_angles : int, default=0
            number of angles
        n_dihedrals : int, default=0
            number of dihedral angles
        n_improper : int, default=0
            number of improper dihedral angles
        """
        items: tuple[int, ...] = (n_atoms, n_bonds, n_angles, n_dihedrals, n_improper)
        numbers: dict[str, int] = dict(zip(self._HEADERS.keys(), items, strict=True))
        self._parameters: dict[str, dict[str, list[str | int | float] | NDArray | None]] = dict.fromkeys(
            self._HEADERS.keys()
        )
        for key in self._parameters:
            self._parameters[key] = dict.fromkeys(self._HEADERS[key])
            self._parameters[key] = {key2: [] for key2 in self._parameters[key]} if numbers[key] > 0 else None

    @property
    def atoms(self: TParameters) -> dict[str, list[str | int | float] | NDArray | None]:
        """Return atom information.

        Returns
        -------
        tuple
            atom types, atom names, and masses
        """
        return self._parameters["ATOMS"]

    @atoms.setter
    def atoms(self: TParameters, params: tuple[list[str | int | float] | NDArray, ...]) -> None:
        """Set atom types, names, and masses.

        Parameters
        ----------
        params : tuple
            atom types, names, and masses
        """
        self._parameters["ATOMS"] = {
            key: copy.deepcopy(value) if isinstance(value, list) else value.copy()
            for key, value in zip(self._parameters["ATOMS"].keys(), params, strict=True)
        }

    @property
    def bonds(self: TParameters) -> dict[str, list[str | int | float] | NDArray | None]:
        """Retrieve the bond information.

        Returns
        -------
        tuple
            atom I, atom J, force constant, bond length
        """
        return self._parameters["BONDS"]

    @bonds.setter
    def bonds(self: TParameters, params: tuple[list[str | int | float] | NDArray, ...]) -> None:
        """Set bond atoms, force constants, and distance.

        Parameters
        ----------
        params : tuple or array
            atom names, force constants, and bond distances
        """
        self._parameters["BONDS"] = {
            key: copy.deepcopy(value) if isinstance(value, list) else value.copy()
            for key, value in zip(self._parameters["BONDS"].keys(), params, strict=True)
        }

    @property
    def bond_info(self: TParameters) -> tuple[NDArray, NDArray] | None:
        """Return only the bond information.

        Returns
        -------
        tuple
            force constants and bond distances
        """
        return (
            self._parameters["BONDS"]["Kb"],
            self._parameters["BONDS"]["b0"] if self._parameters["BONDS"] is not None else self._parameters["BONDS"],
        )

    @bond_info.setter
    def bond_info(self: TParameters, params: tuple[NDArray, NDArray]) -> None:
        """Set force constants, and distance.

        Parameters
        ----------
        params : tuple or array
            atom names, force constants, and bond distances

        Raises
        ------
        AttributeError
            if bond definitions aren't defined
        IndexError
            if array lengths don't match
        """
        forces: NDArray
        distances: NDArray
        forces, distances = params

        if self._parameters["BONDS"] is None:
            message = "Bond parameters are undefined."
            logger.error(message)
            raise AttributeError(message)
        if len(self._parameters["BONDS"]["I"]) != forces.size and forces.size != distances.size:
            message = "Array lengths don't match!"
            logger.error(message)
            raise IndexError(message)

        self._parameters["BONDS"]["Kb"] = forces.copy()
        self._parameters["BONDS"]["b0"] = distances.copy()

    @property
    def angles(self: TParameters) -> dict[str, list[str | int | float] | NDArray | None]:
        """Retrieve the angle information.

        Returns
        -------
        tuple
            atom I, atom J, atom K, force constant, angle
        """
        return self._parameters["ANGLES"]

    @angles.setter
    def angles(self: TParameters, params: tuple[list[str | int | float] | NDArray, ...]) -> None:
        """Set angle atoms, force constants, and angles.

        Parameters
        ----------
        params : tuple or array
            atom names, force constants, and angles
        """
        self._parameters["ANGLES"] = {
            key: copy.deepcopy(value) if isinstance(value, list) else value.copy()
            for key, value in zip(self._parameters["ANGLES"].keys(), params, strict=True)
        }

    @property
    def angle_info(self: TParameters) -> tuple[NDArray, NDArray] | None:
        """Return only the angle information.

        Returns
        -------
        tuple
            force constants and angle distances
        """
        return (
            self._parameters["ANGLES"]["Ktheta"],
            self._parameters["ANGLES"]["theta0"]
            if self._parameters["ANGLES"] is not None
            else self._parameters["ANGLES"],
        )

    @angle_info.setter
    def angle_info(self: TParameters, params: tuple[NDArray, NDArray]) -> None:
        """Set force constants, and distance.

        Parameters
        ----------
        params : tuple or array
            atom names, force constants, and angle distances

        Raises
        ------
        AttributeError
            if bond definitions aren't defined
        IndexError
            if array lengths don't match
        """
        forces: NDArray
        distances: NDArray
        forces, distances = params

        if self._parameters["ANGLES"] is None:
            message = "Angle parameters are undefined."
            logger.error(message)
            raise AttributeError(message)
        if len(self._parameters["ANGLES"]["I"]) != forces.size and forces.size != distances.size:
            message = "Array lengths don't match!"
            logger.error(message)
            raise IndexError(message)

        self._parameters["ANGLES"]["theta0"] = distances.copy()
        self._parameters["ANGLES"]["Ktheta"] = forces.copy()

    @property
    def dihedrals(self: TParameters) -> dict[str, list[str | int | float] | NDArray | None]:
        """Retrieve the dihedral angle information.

        Returns
        -------
        tuple
            atom I, atom J, atom K, atom L, force constant, degeneration, angle
        """
        return self._parameters["DIHEDRALS"]

    @dihedrals.setter
    def dihedrals(self: TParameters, params: tuple[list[str | int | float] | NDArray, ...]) -> None:
        """Set dihedral angle atoms, force constants, and angles.

        Parameters
        ----------
        params : tuple or array
            atom names, force constants, and angles
        """
        self._parameters["DIHEDRALS"] = {
            key: copy.deepcopy(value) if isinstance(value, list) else value.copy()
            for key, value in zip(self._parameters["DIHEDRALS"].keys(), params, strict=True)
        }

    @property
    def dihedral_info(self: TParameters) -> tuple[NDArray, NDArray, NDArray] | None:
        """Return only the dihedral information.

        Returns
        -------
        tuple
            force constants and dihedral distances
        """
        return (
            self._parameters["DIHEDRALS"]["KChi"],
            self._parameters["DIHEDRALS"]["n"],
            self._parameters["DIHEDRALS"]["delta"]
            if self._parameters["DIHEDRALS"] is not None
            else self._parameters["DIHEDRALS"],
        )

    @dihedral_info.setter
    def dihedral_info(self: TParameters, params: tuple[NDArray, NDArray, NDArray]) -> None:
        """Set force constants and distance.

        Parameters
        ----------
        params : tuple or array
            atom names, force constants, and dihedral distances

        Raises
        ------
        AttributeError
            if bond definitions aren't defined
        IndexError
            if array lengths don't match
        """
        forces: NDArray
        degen: NDArray
        angle: NDArray
        forces, degen, angle = params

        if self._parameters["DIHEDRALS"] is None:
            message = "Angle parameters are undefined."
            logger.error(message)
            raise AttributeError(message)
        if len(self._parameters["DIHEDRALS"]["I"]) != forces.size and forces.size != degen.size != angle.size:
            message = "Array lengths don't match!"
            logger.error(message)
            raise IndexError(message)

        self._parameters["DIHEDRALS"]["Kchi"] = forces.copy()
        self._parameters["DIHEDRALS"]["n"] = degen.copy()
        self._parameters["DIHEDRALS"]["delta"] = angle.copy()

    @property
    def improper(self: TParameters) -> dict[str, list[str | int | float] | NDArray | None]:
        """Retrieve the improper dihedral angle information.

        Returns
        -------
        tuple
            atom I, atom J, atom K, atom L, force constant, degeneration, angle
        """
        return self._parameters["IMPROPER"]

    @improper.setter
    def improper(self: TParameters, params: tuple[list[str | int | float] | NDArray, ...]) -> None:
        """Set dihedral angle atoms, force constants, and angles.

        Parameters
        ----------
        params : tuple or array
            atom names, force constants, and angles
        """
        self._parameters["IMPROPER"] = {
            key: copy.deepcopy(value) if isinstance(value, list) else value.copy()
            for key, value in zip(self._parameters["IMPROPER"].keys(), params, strict=True)
        }

    @property
    def improper_info(self: TParameters) -> tuple[NDArray, NDArray, NDArray] | None:
        """Return only the improper information.

        Returns
        -------
        tuple
            force constants and improper distances
        """
        return (
            self._parameters["IMPROPER"]["KChi"],
            self._parameters["IMPROPER"]["n"],
            self._parameters["IMPROPER"]["delta"]
            if self._parameters["IMPROPER"] is not None
            else self._parameters["IMPROPER"],
        )

    @improper_info.setter
    def improper_info(self: TParameters, params: tuple[NDArray, NDArray, NDArray]) -> None:
        """Set force constants and distance.

        Parameters
        ----------
        params : tuple or array
            atom names, force constants, and improper distances

        Raises
        ------
        AttributeError
            if bond definitions aren't defined
        IndexError
            if array lengths don't match
        """
        forces: NDArray
        degen: NDArray
        angle: NDArray
        forces, degen, angle = params

        if self._parameters["IMPROPER"] is None:
            message = "Angle parameters are undefined."
            logger.error(message)
            raise AttributeError(message)
        if len(self._parameters["IMPROPER"]["I"]) != forces.size and forces.size != degen.size != angle.size:
            message = "Array lengths don't match!"
            logger.error(message)
            raise IndexError(message)

        self._parameters["IMPROPER"]["Kchi"] = forces.copy()
        self._parameters["IMPROPER"]["n"] = degen.copy()
        self._parameters["IMPROPER"]["delta"] = angle.copy()

    def create_table(self: TParameters) -> dict[str, pd.DataFrame]:
        """Create a dictionary of parameters.

        Returns
        -------
        dict of DataFrames
            dictionary of parameters
        """
        parameters: dict[str, pd.DataFrame] = {
            key: pd.DataFrame.from_dict(value) for key, value in self._parameters.items() if value is not None
        }

        return parameters

    def from_dataframe(self: TParameters, **kwargs: pd.DataFrame) -> None:
        """Load parameters from a dataframe.

        Parameters
        ----------
        kwargs : pd.DataFrame
            Parameters
        """
        for key, value in kwargs.items():
            if self._parameters[key] is not None:
                self._parameters[key].update(value.to_dict(orient="list"))
            else:
                self._parameters[key] = copy.deepcopy(value.to_dict(orient="list"))

            if self._parameters[key] is not None:
                for k, v in self._parameters[key].items():
                    if isinstance(v[0], float):
                        self._parameters[key][k] = np.asarray(v)
