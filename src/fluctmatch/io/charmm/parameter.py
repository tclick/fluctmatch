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
        self._parameters = CharmmParameterSet()

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
            atom_type.epsilon, atom_type.epsilon_14, atom_type.sigma, atom_type.sigma_14 = 4 * (0.0,)
            self._parameters.atom_types[atom.type] = atom_type

        # Setup atom bond types
        try:
            logger.debug("Adding bond types to the parameter list.")
            self._parameters.bond_types.update({k: pmd.BondType(k=0.0, req=0.0) for k in universe.bonds.types()})

            # Set bond force constants
            k = forces if forces is not None else OrderedDict({k: 0.0 for k in universe.bonds.types()})
            compare_dict_keys(
                self._parameters.bond_types,
                k,
                message="Bonds defining force constants do not match bonds in universe.",
            )

            # Set equilibrium bond distances
            req = lengths if lengths is not None else OrderedDict({k: 0.0 for k in universe.bonds.types()})
            compare_dict_keys(
                self._parameters.bond_types,
                req,
                message="Bonds defining bond lengths do not match bonds in universe.",
            )

            if forces is None and lengths is None:
                logger.warning("Both force constants and bond lengths are set to 0.")
            for bond_type in self._parameters.bond_types:
                self._parameters.bond_types[bond_type] = pmd.BondType(k=k[bond_type], req=req[bond_type])
        except mda.NoDataError as exception:
            logger.exception(exception)
            raise
        except ValueError as exception:
            logger.exception(exception)
            raise

        return self

    def write(self: Self, filename: Path | str, /, *, stream: bool = False) -> None:
        """Write the parameter data to a parameter, topology, or stream file.

        Parameters
        ----------
        filename : Path or str
            Filename to write the parameter data
        stream : bool
            Combine the topology and parameter files into one file. If `False`, write the topology and parameter files
            separately.
        """
        if not self._parameters.atom_types and not self._parameters.bond_types:
            logger.warning("No atom types or bond types were provided. The parameter file will be empty.")

        if stream:
            stream_file = Path(filename).with_suffix(".str")
            logger.info(f"Writing stream file: {stream_file}")
            self._parameters.write(stream=stream_file.as_posix())
        else:
            par = Path(filename).with_suffix(".prm")
            top = par.with_suffix(".rtf")
            logger.info(f"Writing topology to {top} and parameters to {par}.")
            self._parameters.write(par=par.as_posix(), top=top.as_posix())

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

        # Remove duplicates with a key (atom2, atom1)
        keys = set(filter(lambda key: key != (max(key), min(key)), self._parameters.bond_types))
        self._parameters.bond_types = OrderedDict({k: self._parameters.bond_types[k] for k in keys})
        return self


class CharmmParameterSet(charmm.CharmmParameterSet):
    """CHARMM parameter set."""

    def __init__(self: Self, *args: str) -> None:
        """Construct a CHARMM parameter set."""
        super().__init__(*args)

    def _write_par_to(self: Self, f: str) -> None:  # noqa: C901
        """Private method to write parameter items to open file object.

        Notes
        -----
        This method is almost identical to its parent. The primary difference is that the BONDS and ANGLES section
        have four decimal places instead of two. This is for increased accuracy for fluctuation matching.
        """
        # Find out what the 1-4 electrostatic scaling factors and the 1-4
        # van der Waals scaling factors are
        scee, scnb = set(), set()
        for _, typ in self.dihedral_types.items():
            for t in typ:
                if t.scee:
                    scee.add(t.scee)
                if t.scnb:
                    scnb.add(t.scnb)
        if len(scee) > 1 or len(scnb) > 1:
            message = "Mixed 1-4 scaling not supported"
            raise ValueError(message)
        scee = 1.0 if not scee else scee.pop()
        scnb = 1.0 if not scnb else scnb.pop()

        f.write("ATOMS\n")
        self._write_top_to(f, False)
        f.write("\nBONDS\n")
        written = set()
        for key, typ in self.bond_types.items():
            if key in written:
                continue
            written.add(key)
            written.add(tuple(reversed(key)))
            f.write(f"{key[0]:<6s} {key[1]:<6s} {typ.k:7.4f} {typ.req:10.4f}\n")
        f.write("\nANGLES\n")
        written = set()
        for key, typ in self.angle_types.items():
            if key in written:
                continue
            written.add(key)
            written.add(tuple(reversed(key)))
            f.write(f"{key[0]:<6s} {key[1]:<6s} {key[2]:<6s} {typ.k:7.4f} {typ.theteq:8.4f}\n")
        f.write("\nDIHEDRALS\n")
        written = set()
        for key, typ in self.dihedral_types.items():
            if key in written:
                continue
            written.add(key)
            written.add(tuple(reversed(key)))
            for tor in typ:
                f.write(
                    f"{key[0]:<6s} {key[1]:<6s} {key[2]:<6s} {key[3]:<6s} {tor.phi_k:11.4f} {tor.per:2d} {tor.phase:8.2f}\n"
                )
        f.write("\nIMPROPERS\n")
        written = set()
        for key, typ in sorted(self.improper_periodic_types.items(), key=lambda x: x[0]):
            f.write(
                f"{key[0]:<6s} {key[1]:<6s} {key[2]:<6s} {key[3]:<6s} {typ.phi_k:11.4f} {int(typ.per):2d} {typ.phase:8.2f}\n"
            )
        for key, typ in self.improper_types.items():
            f.write(f"{key[0]:<6s} {key[1]:<6s} {key[2]:<6s} {key[3]:<6s} {typ.psi_k:11.4f} {0:2d} {typ.psi_eq:8.2f}\n")
        if self.cmap_types:
            f.write("\nCMAPS\n")
            written = set()
            for key, typ in self.cmap_types.items():
                if key in written:
                    continue
                written.add(key)
                written.add(tuple(reversed(key)))
                f.write(
                    f"{key[0]:<6s} {key[1]:<6s} {key[2]:<6s} {key[3]:<6s} {key[4]:<6s} "
                    f"{key[5]:<6s} {key[6]:<6s} {key[7]:<6s} {typ.resolution:5d}\n\n"
                )
                i = 0
                for val in typ.grid:
                    if i:
                        if i % 5 == 0:
                            f.write("\n")
                            if i % typ.resolution == 0:
                                f.write("\n")
                                i = 0
                        elif i % typ.resolution == 0:
                            f.write("\n\n")
                            i = 0
                    i += 1
                    f.write(f" {val:13.6f}")
                f.write("\n\n\n")
        comb_rule = " GEOM" if self.combining_rule == "geometric" else ""
        f.write(
            "\nNONBONDED  nbxmod  5 atom cdiel fshift vatom vdistance vfswitch -\ncutnb 14.0 "
            f"ctofnb 12.0 ctonnb 10.0 eps 1.0 e14fac {1 / scee} wmin 1.5{comb_rule}\n\n"
        )
        for key, typ in self.atom_types.items():
            f.write(f"{key:<6s} {0:14.6f} {-abs(typ.epsilon):10.6f} {typ.rmin:14.6f}")
            if typ.epsilon == typ.epsilon_14 and typ.rmin == typ.rmin_14:
                f.write(f"{0:10.6f} {-abs(typ.epsilon) / scnb:10.6f} {typ.rmin:14.6f}\n")
            else:
                f.write(f"{0:10.6f} {-abs(typ.epsilon_14):10.6f} {typ.rmin_14:14.6f}\n")
        f.write("\nEND\n")
