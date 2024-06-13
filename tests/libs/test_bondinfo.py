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
"""Test analysis module to collect bond information."""

from collections import OrderedDict

import MDAnalysis as mda
import numpy as np
import pytest
from fluctmatch.io.charmm import Bond, BondData
from fluctmatch.libs.bond_info import BondInfo
from numpy import testing

from tests.datafile import DCD_CG, PSF_ENM

N_RESIDUES = 5
STOP: int = 10


@pytest.fixture(scope="class")
def universe() -> mda.Universe:
    """Return an elastic network model with bonds and a trajectory.

    Returns
    -------
    :class:`MDAnalysis.Universe`
        Elastic network model
    """
    u = mda.Universe(PSF_ENM, DCD_CG)
    return mda.Merge(u.residues[:N_RESIDUES].atoms)


@pytest.fixture(scope="class")
def bond_types(universe: mda.Universe) -> list[Bond]:
    """Return a list of bond types.

    Parameters
    ----------
    universe : :class:`MDAnalysis.Universe`
        Elastic network model

    Returns
    -------
    list[tuple[str, str]]
        List of bond types
    """
    return universe.bonds.types()


@pytest.fixture(scope="class")
def bond_std(universe: mda.Universe, bond_types: list[Bond]) -> BondData:
    """Collect bond information from universe and calculate the standard deviation.

    Parameters
    ----------
    universe : :class:`MDAnalysis.Universe`
        Elastic network model
    bond_types : list[tuple[str, str]]
        List of bond types

    Returns
    -------
    OrderedDict[tuple[str, str], float]
        Standard deviations of bonds
    """
    bonds = OrderedDict({k: [] for k in bond_types})
    for _ in universe.trajectory[:STOP]:
        for k, v in zip(universe.bonds.types(), universe.bonds.values(), strict=True):
            bonds[k].append(v)
    return OrderedDict({k: np.std(v) for k, v in bonds.items()})


@pytest.fixture(scope="class")
def bond_mean(universe: mda.Universe, bond_types: list[Bond]) -> BondData:
    """Collect bond information from universe and calculate the mean.

    Parameters
    ----------
    universe : :class:`MDAnalysis.Universe`
        Elastic network model
    bond_types : list[tuple[str, str]]
        List of bond types

    Returns
    -------
    OrderedDict[tuple[str, str], float]
        Average of bonds
    """
    bonds = OrderedDict({k: [] for k in bond_types})
    for _ in universe.trajectory[:STOP]:
        for k, v in zip(universe.bonds.types(), universe.bonds.values(), strict=True):
            bonds[k].append(v)
    return OrderedDict({k: np.mean(v) for k, v in bonds.items()})


def test_bond_info(universe: mda.Universe, bond_std: BondData, bond_mean: BondData) -> None:
    """Test bond information analysis class.

    Parameters
    ----------
    universe : :class:`MDAnalysis.Universe`
        Elastic network model
    bond_std : OrderedDict[tuple[str, str], float]
        Standard deviations of bonds
    bond_mean : OrderedDict[tuple[str, str], float]
        Average of bonds
    """
    bond_info = BondInfo(universe.atoms)
    bond_info.run(stop=STOP)

    testing.assert_allclose(
        list(bond_info.results.std.values()),
        list(bond_std.values()),
        rtol=1e-4,
        err_msg="Standard deviations of bond lengths do not match.",
    )
    testing.assert_allclose(
        list(bond_info.results.mean.values()),
        list(bond_mean.values()),
        rtol=1e-4,
        err_msg="Means of bond lengths do not match.",
    )
