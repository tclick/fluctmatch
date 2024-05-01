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
"""Various data files for testing."""
from importlib import resources

_data_ref = resources.files("tests.data")

__all__: list[str] = [
    "PDB",
    "GRO",
    "DNA",
    "TIP3P",
    "TIP4P",
    "IONS",
    "DMA",
    "TPR",
    "XTC",
    "NCSC",
    "PSF",
    "DCD",
    "IC",
    "COR",
    "PRM",
    "RTF",
    "STR",
    "CSV",
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

with resources.as_file(_data_ref / "cg.xplor.psf") as f:
    PSF: str = f.as_posix()

with resources.as_file(_data_ref / "cg.dcd") as f:
    DCD: str = f.as_posix()

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

with resources.as_file(_data_ref / "setup.csv") as f:
    CSV: str = f.as_posix()
