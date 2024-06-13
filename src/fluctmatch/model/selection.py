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
# pyright: reportInvalidTypeVarUse = false, reportCallIssue = false, reportArgumentType = false
# pyright: reportAssignmentType = false, reportAttributeAccessIssue = false, reportOptionalMemberAccess = false
"""Define additional selection criteria for MDAnalysis."""

from collections.abc import Iterable
from typing import ClassVar, TypeVar

import numpy as np
from MDAnalysis.core import selection
from MDAnalysis.core.groups import AtomGroup
from numpy.typing import NDArray

Self = TypeVar("Self")


class BioIonSelection(selection.Selection):
    """Contains atoms commonly found in proteins."""

    token: ClassVar[str] = "bioion"
    ion_atoms: ClassVar[set[str]] = set("MG CAL MN FE CU ZN AG".split())

    def _apply(self: Self, group: AtomGroup) -> NDArray:
        """Apply selection to atom group.

        Parameters
        ----------
        group : AtomGroup
            Group of atoms for selection

        Returns
        -------
        NDArray
            Selection of atom names
        """
        atomnames = group.universe._topology.names

        # filter by atom names
        name_matches = [ix for (nm, ix) in atomnames.namedict.items() if nm in self.ion_atoms]
        nmidx = atomnames.nmidx[group.ix]
        group = group[np.in1d(nmidx, name_matches)]

        return group.unique


class WaterSelection(selection.Selection):
    """Contains atoms commonly found in water."""

    token: ClassVar[str] = "water"
    water_atoms: ClassVar[set[str]] = set("TIP3 WAT".split())

    def _apply(self: Self, group: AtomGroup) -> NDArray:
        """Apply selection to atom group.

        Parameters
        ----------
        group : AtomGroup
            Group of atoms for selection

        Returns
        -------
        NDArray
            Selection of atom names
        """
        resname_attr = group.universe._topology.resnames
        # which values in resname attr are in prot_res?
        matches = [ix for (nm, ix) in resname_attr.namedict.items() if nm in self.water_atoms]
        # index of each atom's resname
        nmidx = resname_attr.nmidx[group.resindices]
        # intersect atom's resname index and matches to prot_res
        return group[np.isin(nmidx, matches)]


class BackboneSelection(selection.BackboneSelection):
    """Contains all heavy atoms within a protein backbone including C-termini."""

    token: ClassVar[str] = "backbone"
    oxy_atoms: ClassVar[set[str]] = set("OXT OT1 OT2".split())
    bb_atoms: ClassVar[set[str]] = selection.BackboneSelection.bb_atoms.union(oxy_atoms)

    def _apply(self: Self, group: AtomGroup) -> NDArray:
        """Apply selection to atom group.

        Parameters
        ----------
        group : AtomGroup
            Group of atoms for selection

        Returns
        -------
        NDArray
            Selection of atom names
        """
        atomnames = group.universe._topology.names
        resnames = group.universe._topology.resnames

        # filter by atom names
        name_matches = [ix for (nm, ix) in atomnames.namedict.items() if nm in self.bb_atoms]
        nmidx = atomnames.nmidx[group.ix]
        group = group[np.in1d(nmidx, name_matches)]

        # filter by resnames
        resname_matches = [ix for (nm, ix) in resnames.namedict.items() if nm in self.prot_res]
        nmidx = resnames.nmidx[group.resindices]
        group = group[np.in1d(nmidx, resname_matches)]

        return group.unique


class HBackboneSelection(BackboneSelection):
    """Includes all atoms found within a protein backbone including hydrogens."""

    token: ClassVar[str] = "hbackbone"
    hbb_atoms: ClassVar[set[str]] = set("H HN H1 H2 H3 HT1 HT2 HT3 HA HA1 HA2 1HA 2HA".split())
    bb_atoms: ClassVar[set[str]] = BackboneSelection.bb_atoms.union(hbb_atoms)

    def _apply(self: Self, group: AtomGroup) -> NDArray:
        """Apply selection to atom group.

        Parameters
        ----------
        group : AtomGroup
            Group of atoms for selection

        Returns
        -------
        NDArray
            Selection of atom names
        """
        atomnames = group.universe._topology.names
        resnames = group.universe._topology.resnames

        # filter by atom names
        name_matches = [ix for (nm, ix) in atomnames.namedict.items() if nm in self.bb_atoms]
        nmidx = atomnames.nmidx[group.ix]
        group = group[np.in1d(nmidx, name_matches)]

        # filter by resnames
        resname_matches = [ix for (nm, ix) in resnames.namedict.items() if nm in self.prot_res]
        nmidx = resnames.nmidx[group.resindices]
        group = group[np.in1d(nmidx, resname_matches)]

        return group.unique


class CalphaSelection(selection.ProteinSelection):
    """Contains only the alpha-carbon of a protein."""

    token: ClassVar[str] = "calpha"
    calpha: ClassVar[set[str]] = {"CA"}

    def _apply(self: Self, group: AtomGroup) -> NDArray:
        """Apply selection to atom group.

        Parameters
        ----------
        group : AtomGroup
            Group of atoms for selection

        Returns
        -------
        NDArray
            Selection of atom names
        """
        atomnames = group.universe._topology.names
        resnames = group.universe._topology.resnames

        # filter by atom names
        name_matches = [ix for (nm, ix) in atomnames.namedict.items() if nm in self.calpha]
        nmidx = atomnames.nmidx[group.ix]
        group = group[np.in1d(nmidx, name_matches)]

        # filter by resnames
        resname_matches = [ix for (nm, ix) in resnames.namedict.items() if nm in self.prot_res]
        nmidx = resnames.nmidx[group.resindices]
        group = group[np.in1d(nmidx, resname_matches)]

        return group.unique


class HCalphaSelection(CalphaSelection):
    """Contains the alpha-carbon and alpha-hydrogens of a protein."""

    token: ClassVar[str] = "hcalpha"
    hcalpha: ClassVar[set[str]] = set("HA HA1 HA2 1HA 2HA".split())
    calpha: ClassVar[set[str]] = CalphaSelection.calpha.union(hcalpha)


class CbetaSelection(selection.ProteinSelection):
    """Contains only the beta-carbon of a protein."""

    token: ClassVar[str] = "cbeta"
    cbeta: ClassVar[set[str]] = {"CB"}

    def _apply(self: Self, group: AtomGroup) -> NDArray:
        """Apply selection to atom group.

        Parameters
        ----------
        group : AtomGroup
            Group of atoms for selection

        Returns
        -------
        NDArray
            Selection of atom names
        """
        atomnames = group.universe._topology.names
        resnames = group.universe._topology.resnames

        # filter by atom names
        name_matches = [ix for (nm, ix) in atomnames.namedict.items() if nm in self.cbeta]
        nmidx = atomnames.nmidx[group.ix]
        group = group[np.in1d(nmidx, name_matches)]

        # filter by resnames
        resname_matches = [ix for (nm, ix) in resnames.namedict.items() if nm in self.prot_res]
        nmidx = resnames.nmidx[group.resindices]
        group = group[np.in1d(nmidx, resname_matches)]

        return group.unique


class AmineSelection(selection.ProteinSelection):
    """Contains atoms within the amine group of a protein."""

    token: ClassVar[str] = "amine"
    amine: ClassVar[set[str]] = set("N HN H H1 H2 H3 HT1 HT2 HT3".split())

    def _apply(self: Self, group: AtomGroup) -> NDArray:
        """Apply selection to atom group.

        Parameters
        ----------
        group : AtomGroup
            Group of atoms for selection

        Returns
        -------
        NDArray
            Selection of atom names
        """
        atomnames = group.universe._topology.names
        resnames = group.universe._topology.resnames

        # filter by atom names
        name_matches = [ix for (nm, ix) in atomnames.namedict.items() if nm in self.amine]
        nmidx = atomnames.nmidx[group.ix]
        group = group[np.in1d(nmidx, name_matches)]

        # filter by resnames
        resname_matches = [ix for (nm, ix) in resnames.namedict.items() if nm in self.prot_res]
        nmidx = resnames.nmidx[group.resindices]
        group = group[np.in1d(nmidx, resname_matches)]

        return group.unique


class CarboxylSelection(selection.ProteinSelection):
    """Contains atoms within the carboxyl group of a protein."""

    token: ClassVar[str] = "carboxyl"
    carboxyl: ClassVar[set[str]] = set("C O OXT OT1 OT2".split())

    def _apply(self: Self, group: AtomGroup) -> NDArray:
        """Apply selection to atom group.

        Parameters
        ----------
        group : AtomGroup
            Group of atoms for selection

        Returns
        -------
        NDArray
            Selection of atom names
        """
        atomnames = group.universe._topology.names
        resnames = group.universe._topology.resnames

        # filter by atom names
        name_matches = [ix for (nm, ix) in atomnames.namedict.items() if nm in self.carboxyl]
        nmidx = atomnames.nmidx[group.ix]
        group = group[np.in1d(nmidx, name_matches)]

        # filter by resnames
        resname_matches = [ix for (nm, ix) in resnames.namedict.items() if nm in self.prot_res]
        nmidx = resnames.nmidx[group.resindices]
        group = group[np.in1d(nmidx, resname_matches)]

        return group.unique


class HSidechainSelection(HBackboneSelection):
    """Includes hydrogens on the protein sidechain."""

    token: ClassVar[str] = "hsidechain"

    def _apply(self: Self, group: AtomGroup) -> NDArray:
        """Apply selection to atom group.

        Parameters
        ----------
        group : AtomGroup
            Group of atoms for selection

        Returns
        -------
        NDArray
            Selection of atom names
        """
        atomnames = group.universe._topology.names
        resnames = group.universe._topology.resnames

        # filter by atom names
        name_matches = [ix for (nm, ix) in atomnames.namedict.items() if nm not in self.bb_atoms]
        nmidx = atomnames.nmidx[group.ix]
        group = group[np.in1d(nmidx, name_matches)]

        # filter by resnames
        resname_matches = [ix for (nm, ix) in resnames.namedict.items() if nm in self.prot_res]
        nmidx = resnames.nmidx[group.resindices]
        group = group[np.in1d(nmidx, resname_matches)]

        return group.unique


class AdditionalNucleicSelection(selection.NucleicSelection):
    """Contains additional nucleic acid residues."""

    token: ClassVar[str] = "nucleic"

    def __init__(self: Self, parser: str, tokens: Iterable[str]) -> None:
        super().__init__(parser, tokens)
        self.nucl_res = self.nucl_res.union("OXG ABNP HPX DC35".split())


class HNucleicSugarSelection(AdditionalNucleicSelection, selection.NucleicSugarSelection):
    """Contains the additional atoms definitions for the sugar."""

    token: ClassVar[str] = "hnucleicsugar"

    def __init__(self: Self, parser: str, tokens: Iterable[str]) -> None:
        super().__init__(parser, tokens)
        self.sug_atoms = self.sug_atoms.union("H1' O1' O2' H2' H2'' O3' H3' H3T H4'".split())


class HBaseSelection(AdditionalNucleicSelection, selection.BaseSelection):
    """Contains additional atoms on the base region of the nucleic acids."""

    token: ClassVar[str] = "hnucleicbase"

    def __init__(self: Self, parser: str, tokens: Iterable[str]) -> None:
        super().__init__(parser, tokens)
        self.base_atoms = self.base_atoms.union("O8 H8 H21 H22 H2 O6 H6 H61 H62 H41 H42 H5 H51 H52 H53 H3 H7".split())


class NucleicPhosphateSelection(AdditionalNucleicSelection):
    """Contains the nucleic phosphate group including the C5'."""

    token: ClassVar[str] = "nucleicphosphate"
    phos_atoms: ClassVar[set[str]] = set("P O1P O2P O5' C5' H5' H5'' O5T H5T".split())

    def _apply(self: Self, group: AtomGroup) -> NDArray:
        """Apply selection to atom group.

        Parameters
        ----------
        group : AtomGroup
            Group of atoms for selection

        Returns
        -------
        NDArray
            Selection of atom names
        """
        atomnames = group.universe._topology.names
        resnames = group.universe._topology.resnames

        # filter by atom names
        name_matches = [ix for (nm, ix) in atomnames.namedict.items() if nm in self.phos_atoms]
        nmidx = atomnames.nmidx[group.ix]
        group = group[np.in1d(nmidx, name_matches)]

        # filter by resnames
        resname_matches = [ix for (nm, ix) in resnames.namedict.items() if nm in self.nucl_res]
        nmidx = resnames.nmidx[group.resindices]
        group = group[np.in1d(nmidx, resname_matches)]

        return group.unique


class NucleicC2Selection(AdditionalNucleicSelection):
    """Contains the definition for the C3' region."""

    token: ClassVar[str] = "sugarC2"
    c3_atoms: ClassVar[set[str]] = set("C1' H1' C2' O2' H2' H2''".split())

    def _apply(self: Self, group: AtomGroup) -> NDArray:
        """Apply selection to atom group.

        Parameters
        ----------
        group : AtomGroup
            Group of atoms for selection

        Returns
        -------
        NDArray
            Selection of atom names
        """
        atomnames = group.universe._topology.names
        resnames = group.universe._topology.resnames

        # filter by atom names
        name_matches = [ix for (nm, ix) in atomnames.namedict.items() if nm in self.c3_atoms]
        nmidx = atomnames.nmidx[group.ix]
        group = group[np.in1d(nmidx, name_matches)]

        # filter by resnames
        resname_matches = [ix for (nm, ix) in resnames.namedict.items() if nm in self.nucl_res]
        nmidx = resnames.nmidx[group.resindices]
        group = group[np.in1d(nmidx, resname_matches)]

        return group.unique


class NucleicC4Selection(AdditionalNucleicSelection):
    """Contains the definition for the C4' region."""

    token: ClassVar[str] = "sugarC4"
    c3_atoms: ClassVar[set[str]] = set("C3' O3' H3' H3T C4' O4' H4'".split())

    def _apply(self: Self, group: AtomGroup) -> NDArray:
        """Apply selection to atom group.

        Parameters
        ----------
        group : AtomGroup
            Group of atoms for selection

        Returns
        -------
        NDArray
            Selection of atom names
        """
        atomnames = group.universe._topology.names
        resnames = group.universe._topology.resnames

        # filter by atom names
        name_matches = [ix for (nm, ix) in atomnames.namedict.items() if nm in self.c3_atoms]
        nmidx = atomnames.nmidx[group.ix]
        group = group[np.in1d(nmidx, name_matches)]

        # filter by resnames
        resname_matches = [ix for (nm, ix) in resnames.namedict.items() if nm in self.nucl_res]
        nmidx = resnames.nmidx[group.resindices]
        group = group[np.in1d(nmidx, resname_matches)]

        return group.unique


class BaseCenterSelection(AdditionalNucleicSelection):
    """Contains the central atoms (C4 and C5) on the base of the nuleic acid."""

    token: ClassVar[str] = "nucleiccenter"
    center_atoms: ClassVar[set[str]] = set("C4 C5".split())

    def _apply(self: Self, group: AtomGroup) -> NDArray:
        """Apply selection to atom group.

        Parameters
        ----------
        group : AtomGroup
            Group of atoms for selection

        Returns
        -------
        NDArray
            Selection of atom names
        """
        atomnames = group.universe._topology.names
        resnames = group.universe._topology.resnames

        # filter by atom names
        name_matches = [ix for (nm, ix) in atomnames.namedict.items() if nm in self.center_atoms]
        nmidx = atomnames.nmidx[group.ix]
        group = group[np.in1d(nmidx, name_matches)]

        # filter by resnames
        resname_matches = [ix for (nm, ix) in resnames.namedict.items() if nm in self.nucl_res]
        nmidx = resnames.nmidx[group.resindices]
        group = group[np.in1d(nmidx, resname_matches)]

        return group.unique
