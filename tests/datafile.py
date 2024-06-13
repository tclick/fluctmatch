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
"""Various data files for testing."""

from importlib import resources

_data_ref = resources.files("tests.data")

__all__: list[str] = [
    "DMA",
    "TPR",
    "XTC",
    "NCSC",
    "PSF_CG",
    "IC",
    "COR",
    "PRM",
    "RTF",
    "STR",
    "JSON",
    "PSF_ENM",
    "DCD_CG",
    "IC_FLUCT",
    "IC_AVERAGE",
]

with resources.as_file(_data_ref / "trex1.pdb") as f:
    PDB: str = f.as_posix()

with resources.as_file(_data_ref / "trex1.tpr") as f:
    TPR: str = f.as_posix()

with resources.as_file(_data_ref / "trex2.gro") as f:
    GRO: str = f.as_posix()

with resources.as_file(_data_ref / "dna.pdb") as f:
    DNA: str = f.as_posix()

with resources.as_file(_data_ref / "trex1.xtc") as f:
    XTC: str = f.as_posix()

with resources.as_file(_data_ref / "spc216.gro") as f:
    TIP3P: str = f.as_posix()

with resources.as_file(_data_ref / "tip4p.gro") as f:
    TIP4P: str = f.as_posix()

with resources.as_file(_data_ref / "ions.pdb") as f:
    IONS: str = f.as_posix()

with resources.as_file(_data_ref / "dma.gro") as f:
    DMA: str = f.as_posix()

with resources.as_file(_data_ref / "ncsc.pdb") as f:
    NCSC: str = f.as_posix()

with resources.as_file(_data_ref / "cg.psf") as f:
    PSF_CG: str = f.as_posix()

with resources.as_file(_data_ref / "fluct.ic") as f:
    IC: str = f.as_posix()

with resources.as_file(_data_ref / "fluctmatch.prm") as f:
    PRM: str = f.as_posix()

with resources.as_file(_data_ref / "cg.cor") as f:
    COR: str = f.as_posix()

with resources.as_file(_data_ref / "cg.rtf") as f:
    RTF: str = f.as_posix()

with resources.as_file(_data_ref / "cg.stream") as f:
    STR: str = f.as_posix()

with resources.as_file(_data_ref / "cg.json") as f:
    JSON: str = f.as_posix()

with resources.as_file(_data_ref / "fluctmatch.psf") as f:
    PSF_ENM: str = f.as_posix()

with resources.as_file(_data_ref / "cg.dcd") as f:
    DCD_CG: str = f.as_posix()

with resources.as_file(_data_ref / "fluctmatch.fluct.ic") as f:
    IC_FLUCT: str = f.as_posix()

with resources.as_file(_data_ref / "fluctmatch.average.ic") as f:
    IC_AVERAGE: str = f.as_posix()
