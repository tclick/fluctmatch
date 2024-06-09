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
